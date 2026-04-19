import numpy as np
import matplotlib.pyplot as plt
import adi
import time
from collections import deque

# ==================== Configuration ====================
CENTER_FREQ = 2_479_991_600
SAMPLE_RATE = 521_000
BUFFER_SIZE = 4096
RX_GAIN = 30
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

# The MSP430's DCO is only ~1-3% accurate and the software delay_ms loop
# has per-iteration overhead, so the real bit period is 50 ms +/- ~1.5 ms.
# We build sync templates at a small set of bit periods and let the
# decoder pick whichever gives the strongest NCC. Each tuple is
# (env_per_bit, template_zm, template_norm).
BIT_PERIOD_CANDIDATES = list(range(ENV_PER_BIT - 3, ENV_PER_BIT + 4))  # 22..28
SYNC_TEMPLATES = []
for _epb in BIT_PERIOD_CANDIDATES:
    _t = np.repeat(SYNC_PATTERN_BITS, _epb).astype(np.float32)
    _t_zm = _t - _t.mean()
    _n = float(np.sqrt(np.sum(_t_zm ** 2)))
    SYNC_TEMPLATES.append((_epb, _t_zm, _n))

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
# measured carrier so it sits at 0 Hz. We iterate with a progressively
# narrower search window so a far-off spur in the ±50 kHz span cannot
# capture the lock (previously we'd sometimes lock 33 kHz off).
CARRIER_SEARCH_PASSES = [
    (8_000, 300),   # ±8 kHz,  ignore DC ±300 Hz
    (2_000, 150),   # ±2 kHz,  ignore DC ±150 Hz
    (1_000,  80),   # ±1 kHz,  ignore DC ±80 Hz (fine)
]

def _measure_peak_offset(sdr_, n_bufs, span_hz, dc_guard_hz):
    bufs = []
    for _ in range(n_bufs):
        try:
            bufs.append(sdr_.rx())
        except Exception:
            pass
    if not bufs:
        return None, None
    block = np.concatenate(bufs)
    w = np.hanning(len(block))
    spec = np.fft.fftshift(np.fft.fft(block * w))
    fqs = np.fft.fftshift(np.fft.fftfreq(len(block), 1 / SAMPLE_RATE))
    psd = np.abs(spec) ** 2
    search = (np.abs(fqs) < span_hz) & (np.abs(fqs) > dc_guard_hz)
    if not np.any(search):
        return None, None
    psd_m = np.where(search, psd, 0)
    idx = int(np.argmax(psd_m))
    peak_db = 10 * np.log10(psd_m[idx] + 1e-20)
    # Local noise floor: median of the searched band excluding a ±200 Hz
    # notch around the peak itself.
    notch = np.abs(fqs - fqs[idx]) < 200
    floor_mask = search & ~notch
    floor_db = 10 * np.log10(np.median(psd[floor_mask]) + 1e-20)
    return float(fqs[idx]), float(peak_db - floor_db)

print("Auto-centering on carrier...")
for span_hz, dc_guard in CARRIER_SEARCH_PASSES:
    peak_off, peak_snr = _measure_peak_offset(sdr, 12, span_hz, dc_guard)
    if peak_off is None:
        print(f"  [span ±{span_hz} Hz] no buffers")
        continue
    if peak_snr is None or peak_snr < 8.0:
        print(f"  [span ±{span_hz} Hz] peak at {peak_off:+.0f} Hz "
              f"only {peak_snr:.1f} dB above floor, skipping retune")
        continue
    new_lo = int(sdr.rx_lo + peak_off)
    print(f"  [span ±{span_hz} Hz] peak {peak_off:+.0f} Hz "
          f"({peak_snr:.1f} dB SNR) -> rx_lo = {new_lo}")
    sdr.rx_lo = new_lo
    for _ in range(3):
        try:
            sdr.rx()
        except Exception:
            pass

print(f"Sidebands at: {SUBCARRIER_HZ:+.0f} Hz and {-SUBCARRIER_HZ:+.0f} Hz\n")

# ==================== Sideband Envelope ====================
def sideband_envelope(samples, fs=SAMPLE_RATE, freq=SUBCARRIER_HZ,
                      win=ENV_SAMPLES):
    """
    Return one power value (dB) per `win` samples, measuring energy at
    +/-`freq` relative to baseband (i.e. the two OOK sidebands around the
    centered carrier). Length = len(samples) // win.
    """
    n_full = (len(samples) // win) * win
    if n_full == 0:
        return np.empty(0)
    x = samples[:n_full].reshape(-1, win)
    t = np.arange(win) / fs
    ref_u = np.exp(-2j * np.pi * freq * t)
    ref_l = np.exp(+2j * np.pi * freq * t)
    pu = np.abs(x @ ref_u / win) ** 2
    pl = np.abs(x @ ref_l / win) ** 2
    return 10 * np.log10(pu + pl + 1e-20)


# ==================== Packet decoder (correlation) ====================
# NCC threshold: 1.0 is a perfect template match, 0.0 is pure noise.
# With slow 2 ms envelope blocks + MSP DCO jitter a clean packet scores
# ~0.55-0.80, while background correlates at ~0.15-0.25.
CORR_THRESHOLD = 0.35
# When a peak is found, jitter the bit grid by ±TIMING_SEARCH env samples
# to compensate for MSP clock drift over the 2.4 s packet.
TIMING_SEARCH = 3
# Fraction of each bit cell to average when sampling (centered).
BIT_AVG_FRAC = 0.6

class PacketDecoder:
    """
    Holds a rolling envelope buffer. On every update, slides a matched
    filter for the preamble+sync bit pattern across the envelope, then
    fine-tunes bit timing and samples each bit by averaging the middle
    60 % of its cell.
    """
    def __init__(self, env_buffer_seconds=6.0, verbose=True):
        max_env = int(env_buffer_seconds * 1000 / ENV_WINDOW_MS)
        self.env = deque(maxlen=max_env)
        self.last_decoded = ""
        self.packet_count = 0
        self.last_packet_time = 0.0
        self.last_corr = 0.0
        self.verbose = verbose
        self._last_report = 0.0
        self.attempt_count = 0

    def push(self, env_db):
        self.env.extend(env_db.tolist())

    # --- helpers -------------------------------------------------------
    @staticmethod
    def _sample_bits(env, start, bits_n, thresh, epb):
        """Average BIT_AVG_FRAC of each bit cell (width = epb) and slice."""
        half = int(epb * BIT_AVG_FRAC) // 2
        bits = np.empty(bits_n, dtype=np.uint8)
        for i in range(bits_n):
            c = start + i * epb + epb // 2
            a = max(0, c - half)
            b = min(len(env), c + half + 1)
            bits[i] = 1 if env[a:b].mean() > thresh else 0
        return bits

    @staticmethod
    def _header_from_bits(bits16):
        h = 0
        for b in bits16:
            h = (h << 1) | int(b)
        return h

    @staticmethod
    def _best_ncc(env_zm, tmpl_zm, tmpl_norm, max_start):
        """NCC of `tmpl_zm` against `env_zm`, valid mode, clipped to max_start."""
        tlen = len(tmpl_zm)
        if len(env_zm) < tlen + 1 or max_start <= 0:
            return -1.0, 0
        num = np.correlate(env_zm, tmpl_zm, mode="valid")
        num = num[: max_start + 1]
        if num.size == 0:
            return -1.0, 0
        env_sq = env_zm.astype(np.float64) ** 2
        csum = np.concatenate(([0.0], np.cumsum(env_sq)))
        sig_e = csum[tlen:tlen + len(num)] - csum[:len(num)]
        denom = np.sqrt(sig_e * (tmpl_norm ** 2) + 1e-20)
        ncc = num / denom
        idx = int(np.argmax(ncc))
        return float(ncc[idx]), idx

    # --- main decode loop ---------------------------------------------
    def try_decode(self):
        # Need room for widest template + widest packet.
        max_epb = BIT_PERIOD_CANDIDATES[-1]
        need = PACKET_BITS * max_epb + 16 * max_epb
        if len(self.env) < need:
            return

        env = np.asarray(self.env, dtype=np.float32)

        # 1. Activity gate: need at least ~1.5 dB of modulation somewhere.
        lo, hi = np.percentile(env, [15, 85])
        if (hi - lo) < 1.5:
            self.last_corr = 0.0
            return

        env_zm = env - env.mean()

        # 2. Multi-rate matched filter: pick the bit period with the
        #    highest NCC. MSP DCO drift shows up as a non-25 winner.
        best_peak = -1.0
        best_epb = ENV_PER_BIT
        best_pos = 0
        per_rate = []
        for epb, tmpl, tnorm in SYNC_TEMPLATES:
            max_start = len(env) - PACKET_BITS * epb
            val, pos = self._best_ncc(env_zm, tmpl, tnorm, max_start)
            per_rate.append((epb, val))
            if val > best_peak:
                best_peak = val
                best_epb = epb
                best_pos = pos

        self.last_corr = best_peak

        now = time.time()
        if self.verbose and 0.25 < best_peak < CORR_THRESHOLD \
                and now - self._last_report > 1.5:
            self._last_report = now
            rates = " ".join(f"{e}:{v:.2f}" for e, v in per_rate)
            print(f"  near-miss: best ncc={best_peak:.2f} "
                  f"@ epb={best_epb} (span={hi-lo:.1f} dB)  rates[{rates}]")

        if best_peak < CORR_THRESHOLD:
            return

        # Log when we cross the threshold
        if self.verbose and now - self._last_report > 1.0:
            self._last_report = now
            self.attempt_count += 1
            rates = " ".join(f"{e}:{v:.2f}" for e, v in per_rate)
            print(f"  [attempt {self.attempt_count}] ncc={best_peak:.2f} @ epb={best_epb} "
                  f"(span={hi-lo:.1f} dB)  rates[{rates}]  searching ±{TIMING_SEARCH} samples...")

        # 3. Fine timing search: jitter the bit grid around best_pos and
        #    keep the offset whose header matches most bits (allowing 1
        #    bit-error in the 16-bit header).
        epb = best_epb
        packet_len = PACKET_BITS * epb
        best = None
        for dt in range(-TIMING_SEARCH, TIMING_SEARCH + 1):
            start = best_pos + dt
            if start < 0 or start + packet_len > len(env):
                continue
            win = env[start:start + packet_len]
            w_lo, w_hi = np.percentile(win, [15, 85])
            if (w_hi - w_lo) < 1.2:
                continue
            w_th = 0.5 * (w_lo + w_hi)
            bits = self._sample_bits(env, start, PACKET_BITS, w_th, epb)
            header = self._header_from_bits(bits[:16])
            expected = (PREAMBLE_BYTE << 8) | SYNC_BYTE
            match = 16 - bin(header ^ expected).count("1")
            if best is None or match > best[0]:
                best = (match, dt, start, bits)

        if best is None:
            if self.verbose and now - self._last_report > 1.0:
                self._last_report = now
                print(f"    → timing search failed: no valid window (all too noisy or out of bounds)")
            return
        match, dt, start, bits = best

        if match < 15:
            if self.verbose and now - self._last_report > 1.0:
                self._last_report = now
                expected = (PREAMBLE_BYTE << 8) | SYNC_BYTE
                got = self._header_from_bits(bits[:16])
                print(f"    → dt={dt:+d}: got header 0x{got:04X} (bits: {' '.join(str(b) for b in bits[:16])})")
                print(f"       want        0x{expected:04X} (bits: {' '.join(str(int((expected >> (15-i)) & 1)) for i in range(16))})")
                print(f"       match: {match}/16 Hamming distance")
            return

        # 4. Pack payload MSB-first.
        message = ""
        raw = []
        for byte_i in range(PAYLOAD_BYTES):
            v = 0
            for j in range(8):
                v = (v << 1) | int(bits[16 + byte_i * 8 + j])
            raw.append(v)
            message += chr(v) if 32 <= v < 127 else "?"

        if now - self.last_packet_time > 0.5:
            self.packet_count += 1
            self.last_decoded = message
            self.last_packet_time = now
            bit_ms = epb * ENV_WINDOW_MS
            # Format payload bits for display
            payload_bits = " ".join(str(int(bits[16 + i])) for i in range(32))
            print(f"\n{'='*70}")
            print(f"✓ PACKET #{self.packet_count} DECODED")
            print(f"  Message: \"{message}\"")
            print(f"  Bytes: {' '.join('0x%02X' % v for v in raw)}")
            print(f"  Bits:  {payload_bits}")
            print(f"  NCC: {best_peak:.3f}  Bit period: {bit_ms} ms  Timing shift: {dt:+d}  "
                  f"Header match: {match}/16")
            print(f"{'='*70}\n")

        # Drop the consumed envelope so we don't re-decode the same packet.
        consume = start + packet_len
        for _ in range(min(consume, len(self.env))):
            self.env.popleft()

# ==================== Plot Setup ====================
ACCUM_LEN = 65536
NUM_BUFFERS = ACCUM_LEN // BUFFER_SIZE

freqs = np.fft.fftshift(np.fft.fftfreq(ACCUM_LEN, 1/SAMPLE_RATE))
wide_mask = (freqs >= -WIDE_SPAN_HZ) & (freqs <= WIDE_SPAN_HZ)
zoom_mask = (freqs >= -ZOOM_SPAN_HZ) & (freqs <= ZOOM_SPAN_HZ)
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
print(f"Detection frequencies: {-SUBCARRIER_HZ:+.0f} Hz and {+SUBCARRIER_HZ:+.0f} Hz")
print(f"Decode thresholds: NCC ≥ {CORR_THRESHOLD:.2f}, bit period ±{TIMING_SEARCH} env samples")
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
        env_db = sideband_envelope(samples)
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

        peak_power = np.max(psd_zoom)

        ax_wide.clear()
        ax_wide.plot(freqs_wide / 1000, psd_wide, linewidth=0.6, color='steelblue')
        ax_wide.axvline(0, color='green', linestyle='--', alpha=0.5)
        ax_wide.set_xlabel('Offset (kHz)')
        ax_wide.set_ylabel('Power (dB)')
        ax_wide.set_title(f'Wide View ±{WIDE_SPAN_HZ/1000:.0f} kHz')
        ax_wide.grid(True, alpha=0.3)
        
        ax_zoom.clear()
        ax_zoom.plot(freqs_zoom, psd_zoom, linewidth=0.8, color='steelblue')
        ax_zoom.axvline(0, color='green', linestyle='--', alpha=0.5, label='Carrier')
        ax_zoom.axvline(1000, color='red', linestyle='--', alpha=0.6, label='±1 kHz sideband')
        ax_zoom.axvline(-1000, color='red', linestyle='--', alpha=0.6)
        ax_zoom.set_xlabel('Offset from carrier (Hz)')
        ax_zoom.set_ylabel('Power (dB)')
        ax_zoom.set_title(f'Zoomed View | Carrier {peak_power:.1f} dB')
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
                               f'ncc: {decoder.last_corr:.2f}')
            ax_power.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.pause(0.01)

except KeyboardInterrupt:
    print(f"\n\nStopped. Total packets: {decoder.packet_count}")
    print(f"Last message: \"{decoder.last_decoded}\"")