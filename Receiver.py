import numpy as np
import matplotlib.pyplot as plt
import adi
import time
from collections import deque

# ==================== Configuration ====================
CENTER_FREQ = 2_479_991_600
SAMPLE_RATE = 1_000_000
BUFFER_SIZE = 4096
RX_GAIN = 50
WIDE_SPAN_HZ = 50_000
ZOOM_SPAN_HZ = 5_000

BIT_DURATION_MS = 50
SUBCARRIER_HZ = 1000

SAMPLES_PER_WINDOW = int(SAMPLE_RATE * BIT_DURATION_MS / 1000 / 4)

SYNC_BYTE = 0x7E
PREAMBLE_BYTE = 0xAA

# ==================== SDR Connection ====================
print("Connecting to Pluto...")
try:
    sdr = adi.Pluto("ip:192.168.2.1")
except Exception:
    sdr = adi.Pluto("usb:")
print("Connected.\n")

sdr.sample_rate = SAMPLE_RATE
sdr.rx_lo = CENTER_FREQ
sdr.rx_rf_bandwidth = SAMPLE_RATE
sdr.rx_buffer_size = BUFFER_SIZE
sdr.gain_control_mode_chan0 = "manual"
sdr.rx_hardwaregain_chan0 = RX_GAIN

print("Warming up...")
for i in range(5):
    try:
        sdr.rx()
    except Exception:
        time.sleep(0.2)
print("Ready.\n")

# ==================== Carrier offset ====================
# Skip auto-detection, use known value from observation
DETECTED_CARRIER_OFFSET = -200   # observed from previous runs
# To auto-detect instead, set AUTO_DETECT = True
AUTO_DETECT = False

if AUTO_DETECT:
    print("Detecting carrier frequency...")
    calibration_samples = []
    for _ in range(16):
        try:
            calibration_samples.append(sdr.rx())
        except Exception:
            pass
    
    cal_block = np.concatenate(calibration_samples)
    cal_windowed = cal_block * np.hanning(len(cal_block))
    cal_fft = np.fft.fftshift(np.fft.fft(cal_windowed))
    cal_freqs = np.fft.fftshift(np.fft.fftfreq(len(cal_block), 1/SAMPLE_RATE))
    cal_psd = np.abs(cal_fft)**2
    
    # Narrower search window: ±5 kHz, and ignore DC (±50 Hz)
    search_mask = (np.abs(cal_freqs) < 5000) & (np.abs(cal_freqs) > 50)
    cal_psd_masked = cal_psd.copy()
    cal_psd_masked[~search_mask] = 0
    
    peak_idx_global = np.argmax(cal_psd_masked)
    DETECTED_CARRIER_OFFSET = cal_freqs[peak_idx_global]

print(f"Using carrier offset: {DETECTED_CARRIER_OFFSET:+.1f} Hz")
print(f"Sidebands at: {DETECTED_CARRIER_OFFSET+SUBCARRIER_HZ:+.1f} Hz and {DETECTED_CARRIER_OFFSET-SUBCARRIER_HZ:+.1f} Hz\n")

# ==================== Sideband Power Detector ====================
def detect_subcarrier_power(samples, carrier_offset, fs=SAMPLE_RATE, freq=SUBCARRIER_HZ):
    """
    Measure power at (carrier_offset ± freq) — the actual sideband positions.
    """
    n = len(samples)
    t = np.arange(n) / fs
    
    # Reference at upper sideband (carrier + 1kHz)
    ref_upper = np.exp(-2j * np.pi * (carrier_offset + freq) * t)
    # Reference at lower sideband (carrier - 1kHz)
    ref_lower = np.exp(-2j * np.pi * (carrier_offset - freq) * t)
    
    power_upper = np.abs(np.mean(samples * ref_upper)) ** 2
    power_lower = np.abs(np.mean(samples * ref_lower)) ** 2
    
    return power_upper + power_lower

# ==================== Bit Decoder State Machine ====================
class PacketDecoder:
    def __init__(self):
        self.bit_buffer = deque(maxlen=96)
        self.last_decoded = ""
        self.packet_count = 0
        self.last_packet_time = 0
        
    def add_bit(self, bit):
        self.bit_buffer.append(bit)
        self.try_decode()
    
    def try_decode(self):
        if len(self.bit_buffer) < 48:
            return
        
        bits = list(self.bit_buffer)
        
        for start in range(len(bits) - 48 + 1):
            preamble = bits_to_byte(bits[start:start+8])
            sync = bits_to_byte(bits[start+8:start+16])
            
            if preamble == PREAMBLE_BYTE and sync == SYNC_BYTE:
                message = ""
                for i in range(4):
                    byte_start = start + 16 + i * 8
                    byte_val = bits_to_byte(bits[byte_start:byte_start+8])
                    if 32 <= byte_val < 127:
                        message += chr(byte_val)
                    else:
                        message += "?"
                
                now = time.time()
                if now - self.last_packet_time > 1.0:
                    self.packet_count += 1
                    self.last_decoded = message
                    self.last_packet_time = now
                    print(f"\n{'='*50}")
                    print(f"  PACKET #{self.packet_count} DECODED: \"{message}\"")
                    print(f"{'='*50}\n")
                return

def bits_to_byte(bits):
    if len(bits) != 8:
        return 0
    byte = 0
    for b in bits:
        byte = (byte << 1) | (1 if b else 0)
    return byte

# ==================== Bit Slicer ====================
class BitSlicer:
    def __init__(self):
        self.power_history = deque(maxlen=400)
        self.windows_per_bit = 4
        self.bit_accumulator = []
        
    def add_power(self, power_db):
        self.power_history.append(power_db)
        
        if len(self.power_history) < 20:
            return None
        
        hist = np.array(self.power_history)
        p_low = np.percentile(hist, 25)
        p_high = np.percentile(hist, 75)
        threshold = (p_low + p_high) / 2
        
        # More lenient contrast check (1.5 dB is enough)
        if (p_high - p_low) < 1.5:
            return None
        
        symbol = 1 if power_db > threshold else 0
        self.bit_accumulator.append(symbol)
        
        if len(self.bit_accumulator) >= self.windows_per_bit:
            middle = self.bit_accumulator[1:-1] if len(self.bit_accumulator) >= 3 else self.bit_accumulator
            bit = 1 if sum(middle) > len(middle) / 2 else 0
            self.bit_accumulator = []
            return bit
        
        return None

# ==================== Plot Setup ====================
ACCUM_LEN = 65536
NUM_BUFFERS = ACCUM_LEN // BUFFER_SIZE

freqs = np.fft.fftshift(np.fft.fftfreq(ACCUM_LEN, 1/SAMPLE_RATE))
wide_mask = (freqs >= -WIDE_SPAN_HZ) & (freqs <= WIDE_SPAN_HZ)
zoom_mask = (freqs >= DETECTED_CARRIER_OFFSET - ZOOM_SPAN_HZ) & (freqs <= DETECTED_CARRIER_OFFSET + ZOOM_SPAN_HZ)
freqs_wide = freqs[wide_mask]
freqs_zoom = freqs[zoom_mask]

window_fft = np.hanning(ACCUM_LEN)
window_gain = np.sum(window_fft**2)

plt.ion()
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.8])
ax_wide = fig.add_subplot(gs[0, 0])
ax_zoom = fig.add_subplot(gs[0, 1])
ax_wf = fig.add_subplot(gs[1, :])
ax_power = fig.add_subplot(gs[2, :])

waterfall_rows = 150
wf_data = np.full((waterfall_rows, len(freqs_zoom)), -100.0)
psd_accumulator = None

power_history_plot = deque(maxlen=300)
time_history_plot = deque(maxlen=300)
bit_history_plot = deque(maxlen=100)
bit_time_plot = deque(maxlen=100)

slicer = BitSlicer()
decoder = PacketDecoder()
start_time = time.time()

print(f"Receiver running. Looking for 'OPEN' packets...")
print(f"Detection frequencies: {DETECTED_CARRIER_OFFSET-1000:+.0f} Hz and {DETECTED_CARRIER_OFFSET+1000:+.0f} Hz")
print("Ctrl+C to stop.\n")

frame = 0

try:
    while True:
        buffers = []
        for _ in range(NUM_BUFFERS):
            try:
                buffers.append(sdr.rx())
            except Exception:
                continue
        
        if len(buffers) < NUM_BUFFERS:
            continue
        
        samples = np.concatenate(buffers)
        
        # Bit-level detection
        n_sub = SAMPLES_PER_WINDOW
        for start in range(0, len(samples) - n_sub, n_sub):
            chunk = samples[start:start + n_sub]
            power = detect_subcarrier_power(chunk, DETECTED_CARRIER_OFFSET)
            power_db = 10 * np.log10(power + 1e-20)
            
            power_history_plot.append(power_db)
            time_history_plot.append(time.time() - start_time)
            
            bit = slicer.add_power(power_db)
            if bit is not None:
                bit_history_plot.append(bit)
                bit_time_plot.append(time.time() - start_time)
                decoder.add_bit(bit)
        
        # Plots
        windowed = samples * window_fft
        fft = np.fft.fftshift(np.fft.fft(windowed))
        psd = np.abs(fft)**2 / window_gain
        
        if psd_accumulator is None:
            psd_accumulator = psd.copy()
        else:
            psd_accumulator = 0.7 * psd_accumulator + 0.3 * psd
        
        psd_db = 10 * np.log10(psd_accumulator + 1e-20)
        psd_wide = psd_db[wide_mask]
        psd_zoom = psd_db[zoom_mask]
        
        # Freq axis for zoom relative to carrier
        freqs_zoom_rel = freqs_zoom - DETECTED_CARRIER_OFFSET
        
        peak_power = np.max(psd_zoom)
        frame += 1
        
        # Wide spectrum
        ax_wide.clear()
        ax_wide.plot(freqs_wide / 1000, psd_wide, linewidth=0.6, color='steelblue')
        ax_wide.axvline(DETECTED_CARRIER_OFFSET/1000, color='green', linestyle='--', alpha=0.5)
        ax_wide.set_xlabel('Offset (kHz)')
        ax_wide.set_ylabel('Power (dB)')
        ax_wide.set_title(f'Wide View ±{WIDE_SPAN_HZ/1000:.0f} kHz')
        ax_wide.grid(True, alpha=0.3)
        
        # Zoomed spectrum (x-axis is offset FROM CARRIER now)
        ax_zoom.clear()
        ax_zoom.plot(freqs_zoom_rel, psd_zoom, linewidth=0.8, color='steelblue')
        ax_zoom.axvline(0, color='green', linestyle='--', alpha=0.5, label='Carrier')
        ax_zoom.axvline(1000, color='red', linestyle='--', alpha=0.6, label='±1 kHz sideband')
        ax_zoom.axvline(-1000, color='red', linestyle='--', alpha=0.6)
        ax_zoom.set_xlabel('Offset from carrier (Hz)')
        ax_zoom.set_ylabel('Power (dB)')
        ax_zoom.set_title(f'Zoomed View | Carrier {peak_power:.1f} dB @ {DETECTED_CARRIER_OFFSET:+.0f} Hz')
        ax_zoom.set_xlim(-ZOOM_SPAN_HZ, ZOOM_SPAN_HZ)
        ax_zoom.grid(True, alpha=0.3)
        ax_zoom.legend(loc='upper right', fontsize=8)
        
        # Waterfall
        wf_data = np.roll(wf_data, -1, axis=0)
        wf_data[-1, :] = psd_zoom
        vmin = np.percentile(wf_data, 10)
        vmax = np.percentile(wf_data, 99)
        
        ax_wf.clear()
        ax_wf.imshow(wf_data, aspect='auto', cmap='viridis',
                     extent=[-ZOOM_SPAN_HZ, ZOOM_SPAN_HZ, 0, waterfall_rows],
                     vmin=vmin, vmax=vmax, origin='lower')
        ax_wf.axvline(1000, color='red', linestyle='--', alpha=0.5)
        ax_wf.axvline(-1000, color='red', linestyle='--', alpha=0.5)
        ax_wf.set_xlabel('Offset from carrier (Hz)')
        ax_wf.set_ylabel('Time →')
        ax_wf.set_title('Sideband Waterfall (bright = modulation present)')
        
        # Bit-level plot
        ax_power.clear()
        if len(power_history_plot) > 0:
            ax_power.plot(list(time_history_plot), list(power_history_plot), 
                         linewidth=1, color='darkblue', label='Sideband power')
            
            if len(bit_history_plot) > 0:
                y_top = max(power_history_plot) + 1
                for t_val, b in zip(bit_time_plot, bit_history_plot):
                    color = 'green' if b == 1 else 'red'
                    ax_power.scatter(t_val, y_top, c=color, s=40, zorder=5)
            
            ax_power.set_xlabel('Time (s)')
            ax_power.set_ylabel('Power (dB)')
            ax_power.set_title(f'Bit-level Sideband Power  |  '
                              f'Packets decoded: {decoder.packet_count}  |  '
                              f'Last: "{decoder.last_decoded}"')
            ax_power.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.pause(0.01)

except KeyboardInterrupt:
    print(f"\n\nStopped. Total packets: {decoder.packet_count}")
    print(f"Last message: \"{decoder.last_decoded}\"")