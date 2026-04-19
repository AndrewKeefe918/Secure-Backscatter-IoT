import numpy as np
import matplotlib.pyplot as plt
import adi
import time
from collections import deque

# ==================== Configuration ====================
CENTER_FREQ = 2_479_991_600
SAMPLE_RATE = 521_000
BUFFER_SIZE = 4096
RX_GAIN = 15
WIDE_SPAN_HZ = 50_000
ZOOM_SPAN_HZ = 5_000

BIT_DURATION_MS = 50
SUBCARRIER_HZ = 1000

# Fine envelope resolution: 2 ms -> 25 envelope samples per 50 ms bit.
ENV_WINDOW_MS = 2
ENV_SAMPLES = int(SAMPLE_RATE * ENV_WINDOW_MS / 1000)
ENV_PER_BIT = BIT_DURATION_MS // ENV_WINDOW_MS      # 25

SYNC_BYTE = 0x7E
PREAMBLE_BYTE = 0xAA
PACKET_BITS = 48                                    # preamble+sync+4 payload
PAYLOAD_BYTES = 4

# Build the expected bit pattern of preamble+sync (16 bits, MSB first).
SYNC_PATTERN_BITS = np.array(
    [(PREAMBLE_BYTE >> (7 - i)) & 1 for i in range(8)] +
    [(SYNC_BYTE     >> (7 - i)) & 1 for i in range(8)],
    dtype=np.float32,
)
# Template at envelope resolution: each bit becomes ENV_PER_BIT samples.
SYNC_TEMPLATE = np.repeat(SYNC_PATTERN_BITS, ENV_PER_BIT)   # len = 16*25 = 400
# Zero-mean matched filter so it is insensitive to DC level.
SYNC_TEMPLATE_ZM = SYNC_TEMPLATE - SYNC_TEMPLATE.mean()
SYNC_TEMPLATE_NORM = np.sqrt(np.sum(SYNC_TEMPLATE_ZM ** 2))

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

# Enable AD9361 quadrature + DC-offset tracking. This suppresses the IQ
# image (mirrored peak at -f_carrier) from ~30 dB to >55 dB of rejection.
def _enable_tracking(sdr_):
    tried = []
    for attr in ("rx_quadrature_tracking_en_chan0",
                 "rx_rf_dc_offset_tracking_en_chan0",
                 "rx_bb_dc_offset_tracking_en_chan0"):
        try:
            setattr(sdr_, attr, True)
            tried.append(attr)
        except Exception:
            pass
    try:
        ch = sdr_._ctrl.find_channel("voltage0", False)
        for name in ("quadrature_tracking_en",
                     "rf_dc_offset_tracking_en",
                     "bb_dc_offset_tracking_en"):
            if name in ch.attrs:
                ch.attrs[name].value = "1"
                tried.append(f"ctrl:{name}")
    except Exception:
        pass
    return tried

print(f"Tracking enabled: {_enable_tracking(sdr) or 'none'}")

print("Warming up...")
for i in range(5):
    try:
        sdr.rx()
    except Exception:
        time.sleep(0.2)
print("Ready.\n")

# ==================== Auto-center rx_lo on the carrier ====================
# Pluto TCXO drifts several kHz between power-ups. Retune rx_lo onto the
# measured carrier so it sits at 0 Hz and DETECTED_CARRIER_OFFSET stays small.
print("Auto-centering on carrier...")
cal_bufs = []
for _ in range(16):
    try:
        cal_bufs.append(sdr.rx())
    except Exception:
        pass
if cal_bufs:
    cal = np.concatenate(cal_bufs)
    cal_fft = np.fft.fftshift(np.fft.fft(cal * np.hanning(len(cal))))
    cal_freqs = np.fft.fftshift(np.fft.fftfreq(len(cal), 1/SAMPLE_RATE))
    psd = np.abs(cal_fft) ** 2
    # Search the full capture but ignore residual LO leakage at DC.
    search = (np.abs(cal_freqs) < WIDE_SPAN_HZ) & (np.abs(cal_freqs) > 200)
    psd_masked = np.where(search, psd, 0)
    peak_offset = float(cal_freqs[int(np.argmax(psd_masked))])
    new_lo = int(CENTER_FREQ + peak_offset)
    print(f"  Peak at {peak_offset:+.0f} Hz -> retuning rx_lo to {new_lo} Hz")
    sdr.rx_lo = new_lo
    for _ in range(5):
        try:
            sdr.rx()
        except Exception:
            pass

# ==================== Fine carrier offset (residual, after retune) ====================
DETECTED_CARRIER_OFFSET = 0   # after retune the carrier should be ~0 Hz
AUTO_DETECT = True            # measure residual offset for the decoder

if AUTO_DETECT:
    print("Measuring residual carrier offset...")
    calibration_samples = []
    for _ in range(16):
        try:
            calibration_samples.append(sdr.rx())
        except Exception:
            pass

    if calibration_samples:
        cal_block = np.concatenate(calibration_samples)
        cal_windowed = cal_block * np.hanning(len(cal_block))
        cal_fft = np.fft.fftshift(np.fft.fft(cal_windowed))
        cal_freqs = np.fft.fftshift(np.fft.fftfreq(len(cal_block), 1/SAMPLE_RATE))
        cal_psd = np.abs(cal_fft)**2

        # Narrow search ±2 kHz, ignore DC (±50 Hz).
        search_mask = (np.abs(cal_freqs) < 2000) & (np.abs(cal_freqs) > 50)
        cal_psd_masked = cal_psd.copy()
        cal_psd_masked[~search_mask] = 0
        DETECTED_CARRIER_OFFSET = float(cal_freqs[int(np.argmax(cal_psd_masked))])

print(f"Using carrier offset: {DETECTED_CARRIER_OFFSET:+.1f} Hz")
print(f"Sidebands at: {DETECTED_CARRIER_OFFSET+SUBCARRIER_HZ:+.1f} Hz and {DETECTED_CARRIER_OFFSET-SUBCARRIER_HZ:+.1f} Hz\n")

# ==================== Sideband Envelope ====================
def sideband_envelope(samples, carrier_offset,
                      fs=SAMPLE_RATE, freq=SUBCARRIER_HZ,
                      win=ENV_SAMPLES):
    """
    Return one power value per `win` samples, measuring energy at
    (carrier +/- freq) in dB. Length = len(samples) // win.
    """
    n_full = (len(samples) // win) * win
    if n_full == 0:
        return np.empty(0)
    x = samples[:n_full].reshape(-1, win)
    t = np.arange(win) / fs
    ref_u = np.exp(-2j * np.pi * (carrier_offset + freq) * t)
    ref_l = np.exp(-2j * np.pi * (carrier_offset - freq) * t)
    pu = np.abs(x @ ref_u / win) ** 2
    pl = np.abs(x @ ref_l / win) ** 2
    return 10 * np.log10(pu + pl + 1e-20)


# ==================== Packet decoder (correlation) ====================
class PacketDecoder:
    """
    Holds a rolling envelope buffer. On every update, slides a matched
    filter for the preamble+sync bit pattern across the envelope. The peak
    tells us exactly where bit 0 of the preamble starts, so we can sample
    the 4 payload bytes at the correct bit centers.
    """
    def __init__(self, env_buffer_seconds=6.0):
        max_env = int(env_buffer_seconds * 1000 / ENV_WINDOW_MS)
        self.env = deque(maxlen=max_env)
        self.last_decoded = ""
        self.packet_count = 0
        self.last_packet_time = 0.0
        self.last_corr = 0.0

    def push(self, env_db):
        self.env.extend(env_db.tolist())

    def try_decode(self):
        need = PACKET_BITS * ENV_PER_BIT + len(SYNC_TEMPLATE)
        if len(self.env) < need:
            return

        env = np.asarray(self.env, dtype=np.float32)

        # Binary slice around a robust threshold (midpoint of 20/80 percentile).
        lo, hi = np.percentile(env, [20, 80])
        if (hi - lo) < 2.0:
            self.last_corr = 0.0
            return                              # no modulation on the air
        thresh = 0.5 * (lo + hi)
        binary = (env > thresh).astype(np.float32)

        # Normalized cross-correlation against the preamble+sync template.
        tmpl = SYNC_TEMPLATE_ZM
        # Only search positions where a full 48-bit packet fits after the match.
        search_end = len(binary) - PACKET_BITS * ENV_PER_BIT
        if search_end <= 0:
            return
        # Use np.correlate in 'valid' mode for the template length,
        # then restrict to positions where the payload also fits.
        corr = np.correlate(binary - binary.mean(), tmpl, mode="valid")
        corr = corr[: search_end + 1]
        if corr.size == 0:
            return

        peak = int(np.argmax(corr))
        peak_val = float(corr[peak]) / (SYNC_TEMPLATE_NORM + 1e-12)
        self.last_corr = peak_val

        # Require a strong match (empirically ~8+ for a clean packet).
        if peak_val < 6.0:
            return

        # Sample each of the 48 bits at its center in the envelope.
        bit_offset = peak + ENV_PER_BIT // 2
        bits = np.empty(PACKET_BITS, dtype=np.uint8)
        for i in range(PACKET_BITS):
            idx = bit_offset + i * ENV_PER_BIT
            bits[i] = 1 if env[idx] > thresh else 0

        # Verify preamble+sync survived the bit sampling.
        header = 0
        for b in bits[:16]:
            header = (header << 1) | int(b)
        if header != ((PREAMBLE_BYTE << 8) | SYNC_BYTE):
            return

        # Pack payload MSB-first.
        message = ""
        for byte_i in range(PAYLOAD_BYTES):
            v = 0
            for j in range(8):
                v = (v << 1) | int(bits[16 + byte_i * 8 + j])
            message += chr(v) if 32 <= v < 127 else "?"

        now = time.time()
        if now - self.last_packet_time > 1.0:
            self.packet_count += 1
            self.last_decoded = message
            self.last_packet_time = now
            print(f"\n{'='*50}")
            print(f"  PACKET #{self.packet_count} DECODED: \"{message}\""
                  f"  (corr={peak_val:.1f})")
            print(f"{'='*50}\n")

        # Drop the consumed envelope so we don't re-decode the same packet.
        consume = peak + PACKET_BITS * ENV_PER_BIT
        for _ in range(min(consume, len(self.env))):
            self.env.popleft()

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

PLOT_HISTORY_SEC = 6.0
env_plot = deque(maxlen=int(PLOT_HISTORY_SEC * 1000 / ENV_WINDOW_MS))
env_time_plot = deque(maxlen=int(PLOT_HISTORY_SEC * 1000 / ENV_WINDOW_MS))

decoder = PacketDecoder(env_buffer_seconds=PLOT_HISTORY_SEC)
start_time = time.time()

print(f"Receiver running. Looking for 'OPEN' packets...")
print(f"Detection frequencies: {DETECTED_CARRIER_OFFSET-1000:+.0f} Hz and {DETECTED_CARRIER_OFFSET+1000:+.0f} Hz")
print("Ctrl+C to stop.\n")

frame = 0
# Redraw the spectrum/waterfall only every N frames so rx() stays fed.
PLOT_EVERY = 4

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

        # --- Bit-level detection via matched filter on sideband envelope ---
        env_db = sideband_envelope(samples, DETECTED_CARRIER_OFFSET)
        decoder.push(env_db)
        decoder.try_decode()

        now_rel = time.time() - start_time
        for i, v in enumerate(env_db):
            env_plot.append(float(v))
            # each env sample represents ENV_WINDOW_MS ms
            env_time_plot.append(
                now_rel - (len(env_db) - 1 - i) * ENV_WINDOW_MS / 1000.0
            )

        frame += 1
        if frame % PLOT_EVERY != 0:
            continue

        # --- Spectrum / waterfall (slower cadence) ---
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
        
        freqs_zoom_rel = freqs_zoom - DETECTED_CARRIER_OFFSET
        peak_power = np.max(psd_zoom)

        ax_wide.clear()
        ax_wide.plot(freqs_wide / 1000, psd_wide, linewidth=0.6, color='steelblue')
        ax_wide.axvline(DETECTED_CARRIER_OFFSET/1000, color='green', linestyle='--', alpha=0.5)
        ax_wide.set_xlabel('Offset (kHz)')
        ax_wide.set_ylabel('Power (dB)')
        ax_wide.set_title(f'Wide View ±{WIDE_SPAN_HZ/1000:.0f} kHz')
        ax_wide.grid(True, alpha=0.3)
        
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
        
        ax_power.clear()
        if len(env_plot) > 0:
            ax_power.plot(list(env_time_plot), list(env_plot),
                          linewidth=1, color='darkblue', label='Sideband power (2 ms)')
            ax_power.set_xlabel('Time (s)')
            ax_power.set_ylabel('Power (dB)')
            ax_power.set_title(f'Sideband envelope  |  '
                               f'Packets decoded: {decoder.packet_count}  |  '
                               f'Last: "{decoder.last_decoded}"  |  '
                               f'corr: {decoder.last_corr:.1f}')
            ax_power.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.pause(0.01)

except KeyboardInterrupt:
    print(f"\n\nStopped. Total packets: {decoder.packet_count}")
    print(f"Last message: \"{decoder.last_decoded}\"")