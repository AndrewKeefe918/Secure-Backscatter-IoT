import numpy as np
import matplotlib.pyplot as plt
import adi
import time

# ==================== Configuration ====================
CENTER_FREQ = 2_479_991_500   # nudge ~700 Hz higher to center carrier at 0
SAMPLE_RATE = 1_000_000
BUFFER_SIZE = 4096
RX_GAIN = 50
WIDE_SPAN_HZ = 50_000     # Wide view: ±50 kHz
ZOOM_SPAN_HZ = 5_000      # Zoomed view: ±5 kHz
AVG_ALPHA = 0.2

# ==================== Connect ====================
print("Connecting to Pluto...")
try:
    sdr = adi.Pluto("ip:192.168.2.1")
    print("  Connected via IP.")
except Exception as e:
    print(f"  IP failed ({e}); trying USB...")
    sdr = adi.Pluto("usb:")
    print("  Connected via USB.")

sdr.sample_rate = SAMPLE_RATE
sdr.rx_lo = CENTER_FREQ
sdr.rx_rf_bandwidth = SAMPLE_RATE
sdr.rx_buffer_size = BUFFER_SIZE
sdr.gain_control_mode_chan0 = "manual"
sdr.rx_hardwaregain_chan0 = RX_GAIN

# Warmup
print("\nWarming up...")
for i in range(5):
    try:
        sdr.rx()
    except Exception as e:
        print(f"  Warmup {i}: {e}")
        time.sleep(0.2)
print("Ready.\n")

# ==================== Accumulation setup ====================
ACCUM_LEN = 65536
NUM_BUFFERS = ACCUM_LEN // BUFFER_SIZE

freqs = np.fft.fftshift(np.fft.fftfreq(ACCUM_LEN, 1/SAMPLE_RATE))
wide_mask = (freqs >= -WIDE_SPAN_HZ) & (freqs <= WIDE_SPAN_HZ)
zoom_mask = (freqs >= -ZOOM_SPAN_HZ) & (freqs <= ZOOM_SPAN_HZ)
freqs_wide = freqs[wide_mask]
freqs_zoom = freqs[zoom_mask]

window = np.hanning(ACCUM_LEN)
window_gain = np.sum(window**2)

# ==================== Plot setup ====================
plt.ion()
fig, axes = plt.subplots(2, 2, figsize=(16, 9))
ax_wide = axes[0, 0]       # Top-left: wide spectrum
ax_zoom = axes[0, 1]       # Top-right: zoomed spectrum
ax_wf_wide = axes[1, 0]    # Bottom-left: wide waterfall
ax_wf_zoom = axes[1, 1]    # Bottom-right: zoomed waterfall

waterfall_rows = 120
wf_wide_data = np.full((waterfall_rows, len(freqs_wide)), -100.0)
wf_zoom_data = np.full((waterfall_rows, len(freqs_zoom)), -100.0)

psd_accumulator = None
frame = 0

print("Receiver running. Two views: wide (±50 kHz) and zoomed (±5 kHz).")
print("Ctrl+C to stop.\n")

try:
    while True:
        # Collect buffers
        buffers = []
        for _ in range(NUM_BUFFERS):
            try:
                buffers.append(sdr.rx())
            except Exception as e:
                continue
        
        if len(buffers) < NUM_BUFFERS:
            print(f"  Dropped frame (got {len(buffers)}/{NUM_BUFFERS} buffers)")
            continue
        
        samples = np.concatenate(buffers)
        windowed = samples * window
        fft = np.fft.fftshift(np.fft.fft(windowed))
        psd = np.abs(fft)**2 / window_gain
        
        if psd_accumulator is None:
            psd_accumulator = psd.copy()
        else:
            psd_accumulator = (1 - AVG_ALPHA) * psd_accumulator + AVG_ALPHA * psd
        
        psd_db = 10 * np.log10(psd_accumulator + 1e-20)
        psd_wide = psd_db[wide_mask]
        psd_zoom = psd_db[zoom_mask]
        
        # Analyze zoomed region
        peak_idx = np.argmax(psd_zoom)
        peak_freq = freqs_zoom[peak_idx]
        peak_power = psd_zoom[peak_idx]
        noise_mask = np.abs(freqs_zoom) > 2000
        noise_floor = np.median(psd_zoom[noise_mask])
        snr = peak_power - noise_floor
        
        # Check for sideband peaks specifically
        sb_pos_mask = (freqs_zoom > 800) & (freqs_zoom < 1200)
        sb_neg_mask = (freqs_zoom > -1200) & (freqs_zoom < -800)
        sb_pos_power = np.max(psd_zoom[sb_pos_mask]) if np.any(sb_pos_mask) else -999
        sb_neg_power = np.max(psd_zoom[sb_neg_mask]) if np.any(sb_neg_mask) else -999
        
        frame += 1
        if frame % 3 == 0:
            print(f"Carrier: {peak_power:6.1f} dB @ {peak_freq:+6.0f} Hz | "
                  f"SB+: {sb_pos_power:6.1f} dB | SB-: {sb_neg_power:6.1f} dB | "
                  f"Noise: {noise_floor:6.1f} dB")
        
        # ========== Wide spectrum ==========
        ax_wide.clear()
        ax_wide.plot(freqs_wide / 1000, psd_wide, linewidth=0.6, color='steelblue')
        ax_wide.axvline(0, color='green', linestyle='--', alpha=0.5, label='Carrier')
        ax_wide.axvline(1, color='red', linestyle='--', alpha=0.3)
        ax_wide.axvline(-1, color='red', linestyle='--', alpha=0.3)
        ax_wide.set_xlabel('Offset from carrier (kHz)')
        ax_wide.set_ylabel('Power (dB)')
        ax_wide.set_title(f'Wide View ±{WIDE_SPAN_HZ/1000:.0f} kHz')
        ax_wide.set_xlim(-WIDE_SPAN_HZ/1000, WIDE_SPAN_HZ/1000)
        ax_wide.grid(True, alpha=0.3)
        ax_wide.legend(loc='upper right', fontsize=8)
        
        # ========== Zoomed spectrum ==========
        ax_zoom.clear()
        ax_zoom.plot(freqs_zoom, psd_zoom, linewidth=0.8, color='steelblue')
        ax_zoom.axvline(0, color='green', linestyle='--', alpha=0.5, label='Carrier')
        ax_zoom.axvline(1000, color='red', linestyle='--', alpha=0.6, label='+1 kHz')
        ax_zoom.axvline(-1000, color='red', linestyle='--', alpha=0.6, label='-1 kHz')
        ax_zoom.axvline(3000, color='orange', linestyle=':', alpha=0.4)
        ax_zoom.axvline(-3000, color='orange', linestyle=':', alpha=0.4)
        ax_zoom.set_xlabel('Offset from carrier (Hz)')
        ax_zoom.set_ylabel('Power (dB)')
        ax_zoom.set_title(f'Zoomed View ±{ZOOM_SPAN_HZ/1000:.0f} kHz  '
                         f'(Carrier {peak_power:.1f} dB, SNR {snr:.1f} dB)')
        ax_zoom.set_xlim(-ZOOM_SPAN_HZ, ZOOM_SPAN_HZ)
        ax_zoom.grid(True, alpha=0.3)
        ax_zoom.legend(loc='upper right', fontsize=8)
        
        # ========== Wide waterfall ==========
        wf_wide_data = np.roll(wf_wide_data, -1, axis=0)
        wf_wide_data[-1, :] = psd_wide
        vmin_w = np.percentile(wf_wide_data, 10)
        vmax_w = np.percentile(wf_wide_data, 99)
        
        ax_wf_wide.clear()
        ax_wf_wide.imshow(wf_wide_data, aspect='auto', cmap='viridis',
                          extent=[-WIDE_SPAN_HZ/1000, WIDE_SPAN_HZ/1000, 0, waterfall_rows],
                          vmin=vmin_w, vmax=vmax_w, origin='lower')
        ax_wf_wide.axvline(0, color='white', linestyle='--', alpha=0.4)
        ax_wf_wide.set_xlabel('Offset (kHz)')
        ax_wf_wide.set_ylabel('Time →')
        ax_wf_wide.set_title('Wide Waterfall')
        
        # ========== Zoomed waterfall ==========
        wf_zoom_data = np.roll(wf_zoom_data, -1, axis=0)
        wf_zoom_data[-1, :] = psd_zoom
        vmin_z = np.percentile(wf_zoom_data, 10)
        vmax_z = np.percentile(wf_zoom_data, 99)
        
        ax_wf_zoom.clear()
        ax_wf_zoom.imshow(wf_zoom_data, aspect='auto', cmap='viridis',
                          extent=[-ZOOM_SPAN_HZ, ZOOM_SPAN_HZ, 0, waterfall_rows],
                          vmin=vmin_z, vmax=vmax_z, origin='lower')
        ax_wf_zoom.axvline(0, color='white', linestyle='--', alpha=0.4)
        ax_wf_zoom.axvline(1000, color='red', linestyle='--', alpha=0.5)
        ax_wf_zoom.axvline(-1000, color='red', linestyle='--', alpha=0.5)
        ax_wf_zoom.set_xlabel('Offset (Hz)')
        ax_wf_zoom.set_title('Zoomed Waterfall')
        
        plt.tight_layout()
        plt.pause(0.05)

except KeyboardInterrupt:
    print("\nStopped.")