import numpy as np
import matplotlib.pyplot as plt
import adi
import time
from collections import deque

# ==================== Configuration ====================
CENTER_FREQ = 2_479_991_600
SAMPLE_RATE = 521_000
BUFFER_SIZE = 4096
RX_GAIN = 50
WIDE_SPAN_HZ = 50_000
ZOOM_SPAN_HZ = 12_000
CARRIER_COARSE_SEARCH_HZ = 12_000
CARRIER_FINE_SEARCH_HZ = 1_500

BIT_DURATION_MS = 50
SUBCARRIER_HZ = 1000
NOISE_REF_DELTA_HZ = 350
ENV_MEDIAN_KERNEL = 5
MSP_HEALTH_ENABLED = True
MSP_HEALTH_REPORT_SEC = 3.0
MSP_HEALTH_MIN_TRANSITIONS = 10
# If multiple carrier candidates are similarly strong, prefer the one
# closest to DC by allowing this dB margin from the strongest peak.
CARRIER_NEAR_DC_MARGIN_DB = 6.0

# Envelope resolution: 5 ms per sample → 10 envelope samples per 50 ms bit.
# Larger windows reduce noise bandwidth (~200 Hz vs ~500 Hz at 2 ms),
# giving ~4 dB better SNR at the cost of coarser time resolution.
# 10 samples/bit is still more than enough for the matched filter.
ENV_WINDOW_MS = 5
ENV_SAMPLES = int(SAMPLE_RATE * ENV_WINDOW_MS / 1000)
ENV_PER_BIT = BIT_DURATION_MS // ENV_WINDOW_MS      # 10

SYNC_BYTE = 0x7E
PREAMBLE_BYTE = 0xAA
PACKET_BITS = 48                                    # preamble+sync+4 payload
PAYLOAD_BYTES = 4
EXPECTED_PAYLOAD = b"OPEN"
EXPECTED_PACKET_BYTES = bytes([PREAMBLE_BYTE, SYNC_BYTE]) + EXPECTED_PAYLOAD
EXPECTED_PACKET_BITS = np.array(
    [(b >> (7 - i)) & 1 for b in EXPECTED_PACKET_BYTES for i in range(8)],
    dtype=np.uint8,
)

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
BIT_PERIOD_CANDIDATES = list(range(ENV_PER_BIT - 3, ENV_PER_BIT + 4))  # 7..13
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

# ==================== Carrier Location ====================
# Find the CW carrier in baseband and return its offset from 0 Hz.
# Strategy: accumulate a large block for high SNR, use Blackman-Harris
# window for narrow mainlobe, search the valid band excluding DC leakage
# (±100 Hz) and the OOK sideband region (±800-1200 Hz). No hard SNR
# threshold — always trust the best peak found; repeat twice to converge.

def _locate_carrier(sdr_, n_bufs=32, search_hz=12_000):
    """Return the carrier frequency offset in Hz (relative to rx_lo)."""
    bufs = []
    for _ in range(n_bufs):
        try:
            bufs.append(sdr_.rx())
        except Exception:
            pass
    if not bufs:
        return 0.0
    block = np.concatenate(bufs)
    # Blackman-Harris window → very low sidelobes, helps reject sidebands.
    win = np.blackman(len(block))
    spec = np.fft.fftshift(np.fft.fft(block * win))
    fqs  = np.fft.fftshift(np.fft.fftfreq(len(block), 1 / SAMPLE_RATE))
    psd  = np.abs(spec) ** 2
    # Valid search region: exclude DC (±100 Hz) and sideband zone (±800-1200 Hz).
    valid = (np.abs(fqs) > 100) & (np.abs(fqs) < search_hz) & \
            ~((np.abs(fqs) > 800) & (np.abs(fqs) < 1200))
    if not np.any(valid):
        return 0.0
    valid_idx = np.flatnonzero(valid)
    valid_psd = psd[valid]
    # Prefer near-DC if it is within a few dB of the strongest candidate.
    best_i = int(np.argmax(valid_psd))
    best_p = float(valid_psd[best_i])
    strong = valid_psd >= (best_p / (10 ** (CARRIER_NEAR_DC_MARGIN_DB / 10.0)))
    strong_idx = valid_idx[strong]
    idx = int(strong_idx[np.argmin(np.abs(fqs[strong_idx]))])
    offset = float(fqs[idx])
    snr = 10*np.log10(psd[idx] + 1e-20) - 10*np.log10(np.median(valid_psd) + 1e-20)
    print(f"    peak at {offset:+.1f} Hz  ({snr:.1f} dB above median)")
    return offset

print("Locating carrier (pass 1)...")
_off1 = _locate_carrier(sdr, search_hz=CARRIER_COARSE_SEARCH_HZ)
sdr.rx_lo = int(sdr.rx_lo + _off1)
# Drain stale buffers after retune.
for _ in range(4):
    try: sdr.rx()
    except Exception: pass

print("Locating carrier (pass 2)...")
# Fine pass: keep the lock near DC so strong OOK odd harmonics (±3/5/7 kHz)
# cannot be mistaken for the CW carrier.
_off2 = _locate_carrier(sdr, search_hz=CARRIER_FINE_SEARCH_HZ)
sdr.rx_lo = int(sdr.rx_lo + _off2)
for _ in range(4):
    try: sdr.rx()
    except Exception: pass

print("Measuring residual carrier offset...")
CARRIER_OFFSET = _locate_carrier(sdr, n_bufs=16, search_hz=CARRIER_FINE_SEARCH_HZ)
print(f"Carrier at {CARRIER_OFFSET:+.1f} Hz | "
      f"Sidebands at {CARRIER_OFFSET+SUBCARRIER_HZ:+.0f} Hz and "
      f"{CARRIER_OFFSET-SUBCARRIER_HZ:+.0f} Hz\n")


# ==================== Sideband Envelope ====================
def sideband_envelope(samples, carrier_offset=0.0, fs=SAMPLE_RATE,
                      freq=SUBCARRIER_HZ, win=ENV_SAMPLES,
                      noise_delta=NOISE_REF_DELTA_HZ):
    """
    Return a denoised envelope for OOK detection plus diagnostics:
    - env_clean_db: sideband-to-noise ratio (dB) per `win` samples
    - env_sig_db: raw sideband power (dB)
    - env_noise_db: local reference noise floor (dB)

    Noise floor is estimated from four bins adjacent to the sidebands:
    (carrier_offset +/- freq +/- noise_delta). We robustly reject a
    contaminated reference by dropping the strongest bin each window,
    then averaging the remaining three. This prevents a narrow interferer
    on one probe from biasing the floor estimate.
    """
    n_full = (len(samples) // win) * win
    if n_full == 0:
        empty = np.empty(0)
        return empty, empty, empty
    x = samples[:n_full].reshape(-1, win)
    t = np.arange(win) / fs

    ref_u = np.exp(-2j * np.pi * (carrier_offset + freq) * t)
    ref_l = np.exp(-2j * np.pi * (carrier_offset - freq) * t)
    ref_u_i = np.exp(-2j * np.pi * (carrier_offset + freq - noise_delta) * t)
    ref_u_o = np.exp(-2j * np.pi * (carrier_offset + freq + noise_delta) * t)
    ref_l_i = np.exp(-2j * np.pi * (carrier_offset - freq + noise_delta) * t)
    ref_l_o = np.exp(-2j * np.pi * (carrier_offset - freq - noise_delta) * t)

    pu = np.abs(x @ ref_u / win) ** 2
    pl = np.abs(x @ ref_l / win) ** 2
    pui = np.abs(x @ ref_u_i / win) ** 2
    puo = np.abs(x @ ref_u_o / win) ** 2
    pli = np.abs(x @ ref_l_i / win) ** 2
    plo = np.abs(x @ ref_l_o / win) ** 2

    sig = pu + pl
    refs = np.stack((pui, puo, pli, plo), axis=1)
    refs_sorted = np.sort(refs, axis=1)
    noise = np.mean(refs_sorted[:, :3], axis=1)
    env_clean_db = 10 * np.log10((sig + 1e-20) / (noise + 1e-20))
    env_sig_db = 10 * np.log10(sig + 1e-20)
    env_noise_db = 10 * np.log10(noise + 1e-20)
    return env_clean_db, env_sig_db, env_noise_db


def median_filter_1d(x, kernel=ENV_MEDIAN_KERNEL):
    """Robustly suppress impulse spikes with a small sliding median."""
    if len(x) == 0 or kernel <= 1:
        return x
    if kernel % 2 == 0:
        kernel += 1
    pad = kernel // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    y = np.empty_like(x, dtype=np.float32)
    for i in range(len(x)):
        y[i] = np.median(xp[i:i + kernel])
    return y


def _estimate_bit_period_from_transitions_s(dt):
    """Estimate bit period from transition spacings using integer-multiple fit."""
    if len(dt) < MSP_HEALTH_MIN_TRANSITIONS:
        return None
    d = np.asarray(dt, dtype=np.float64)
    d = d[(d > 0.015) & (d < 0.25)]
    if d.size < MSP_HEALTH_MIN_TRANSITIONS:
        return None

    # Search around expected 50 ms and pick the candidate whose transition
    # spacings are closest to integer multiples of that candidate.
    cands = np.linspace(0.035, 0.065, 121)
    best_c = None
    best_err = 1e9
    for c in cands:
        q = d / c
        err = float(np.median(np.abs(q - np.maximum(1.0, np.round(q)))))
        if err < best_err:
            best_err = err
            best_c = c
    if best_c is None or best_err > 0.18:
        return None
    return float(best_c)


class MSPHealthMonitor:
    """Tracks envelope transitions and reports transmitter timing drift."""
    def __init__(self):
        self.prev_state = None
        self.prev_time = None
        self.transitions = deque(maxlen=400)
        self.last_report_t = 0.0

    def push(self, env_db, t0, dt_s):
        if len(env_db) == 0:
            return
        lo, hi = np.percentile(env_db, [15, 85])
        span = float(hi - lo)
        if span < 2.0:
            return
        th = 0.5 * (lo + hi)
        hyst = max(0.35, 0.12 * span)

        t = t0
        for v in env_db:
            if self.prev_state is None:
                self.prev_state = 1 if v >= th else 0
            else:
                if self.prev_state == 0 and v > (th + hyst):
                    self.prev_state = 1
                    self.transitions.append(t)
                elif self.prev_state == 1 and v < (th - hyst):
                    self.prev_state = 0
                    self.transitions.append(t)
            t += dt_s

        if t0 - self.last_report_t < MSP_HEALTH_REPORT_SEC:
            return
        self.last_report_t = t0
        if len(self.transitions) < MSP_HEALTH_MIN_TRANSITIONS + 1:
            return

        edge_t = np.asarray(self.transitions, dtype=np.float64)
        dt = np.diff(edge_t)
        est = _estimate_bit_period_from_transitions_s(dt)
        if est is None:
            return
        target = BIT_DURATION_MS / 1000.0
        err_pct = 100.0 * (est - target) / target
        print(f"MSP health: bit {est*1000.0:5.1f} ms (target {BIT_DURATION_MS} ms, {err_pct:+.1f}%) "
              f"from {len(dt)} transitions")


# ==================== Packet decoder (correlation) ====================
# NCC threshold: 1.0 is a perfect template match, 0.0 is pure noise.
# With 5 ms denoised envelope blocks + MSP DCO jitter a clean packet scores
# ~0.55-0.80, while background correlates at ~0.15-0.25.
CORR_THRESHOLD = 0.35
# When a peak is found, jitter the bit grid by ±TIMING_SEARCH env samples
# to compensate for MSP clock drift over the 2.4 s packet.
TIMING_SEARCH = 3
# Fraction of each bit cell to average when sampling (centered).
BIT_AVG_FRAC = 0.6
# Try these bit-grid offsets after timing lock to absorb residual skew.
BIT_OFFSET_CANDIDATES = (0, 1, 2, 3)
# Minimum modulation span needed before attempting decode.
ACTIVITY_MIN_SPAN_DB = 2.5
# Minimum modulation span inside candidate packet window.
WINDOW_MIN_SPAN_DB = 1.0
# Require at least this many payload bits to match EXPECTED_PAYLOAD.
PAYLOAD_MIN_MATCH_BITS = 24
# Require this many total bits (out of 48) to match expected packet pattern.
PACKET_MIN_MATCH_BITS = 40
# Receiver-side error correction: combine multiple nearby timing/phase
# candidates and vote each bit to repair marginal decodes.
ECC_MIN_PACKET_MATCH_BITS = 24
ECC_MIN_HEADER_MATCH_BITS = 8

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
    def _ecc_vote(candidates):
        """
        Weighted bit voting across candidate packets.
        Candidates that already resemble the expected frame contribute more.
        """
        vote = np.zeros(PACKET_BITS, dtype=np.float64)
        total_w = 0.0
        used = 0
        for pmatch, hmatch, _dt, _start, bits_long, bit_off, invert in candidates:
            if (pmatch < ECC_MIN_PACKET_MATCH_BITS
                    and hmatch < ECC_MIN_HEADER_MATCH_BITS):
                continue
            bits = bits_long[bit_off:bit_off + PACKET_BITS]
            if invert:
                bits = (1 - bits).astype(np.uint8)
            w = 0.2 + (pmatch / PACKET_BITS) + (hmatch / 16.0)
            vote += w * bits
            total_w += w
            used += 1
        if used == 0 or total_w <= 0:
            return None
        return (vote >= (0.5 * total_w)).astype(np.uint8)

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

        # 1. Activity gate: require meaningful modulation before decoding.
        lo, hi = np.percentile(env, [15, 85])
        if (hi - lo) < ACTIVITY_MIN_SPAN_DB:
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

        # Log when we cross the threshold; use a flag so follow-up lines
        # (timing result, header bits) are always printed for this attempt
        # regardless of _last_report throttling.
        logged = False
        if self.verbose and now - self._last_report > 1.0:
            self._last_report = now
            self.attempt_count += 1
            logged = True
            rates = " ".join(f"{e}:{v:.2f}" for e, v in per_rate)
            print(f"  [attempt {self.attempt_count}] ncc={best_peak:.2f} @ epb={best_epb} "
                  f"(span={hi-lo:.1f} dB)  rates[{rates}]  searching ±{TIMING_SEARCH} samples...")

        # 3. Fine timing search: jitter the bit grid around best_pos and
        #    keep the offset whose header matches most bits (allowing 1
        #    bit-error in the 16-bit header).
        epb = best_epb
        expected = (PREAMBLE_BYTE << 8) | SYNC_BYTE
        # Sample extra bits so we can test several bit-grid offsets without
        # changing the envelope timing search.
        packet_len = (PACKET_BITS + max(BIT_OFFSET_CANDIDATES)) * epb
        best = None
        candidates = []
        rejected_noisy = 0
        rejected_header = []
        for dt in range(-TIMING_SEARCH, TIMING_SEARCH + 1):
            start = best_pos + dt
            if start < 0 or start + packet_len > len(env):
                continue
            win = env[start:start + packet_len]
            w_lo, w_hi = np.percentile(win, [15, 85])
            if (w_hi - w_lo) < WINDOW_MIN_SPAN_DB:
                rejected_noisy += 1
                continue
            w_th = 0.5 * (w_lo + w_hi)
            bits_long = self._sample_bits(env, start, PACKET_BITS + max(BIT_OFFSET_CANDIDATES), w_th, epb)

            # Try several alignments and both
            # polarities (normal/inverted). Keep the best matching variant.
            local_best = None
            for bit_off in BIT_OFFSET_CANDIDATES:
                cand_bits = bits_long[bit_off:bit_off + PACKET_BITS]
                hbits = cand_bits[:16]
                for invert in (False, True):
                    hb = (1 - hbits) if invert else hbits
                    header = self._header_from_bits(hb)
                    match = 16 - bin(header ^ expected).count("1")
                    cb = (1 - cand_bits) if invert else cand_bits
                    pkt_match = int(np.sum(cb == EXPECTED_PACKET_BITS))
                    if (local_best is None or
                            pkt_match > local_best[0] or
                            (pkt_match == local_best[0] and match > local_best[1])):
                        local_best = (pkt_match, match, bit_off, invert, header)

            l_pmatch, l_hmatch, l_off, l_inv, l_hdr = local_best
            rejected_header.append((dt, l_hdr, l_hmatch, l_off, l_inv, l_pmatch))
            candidates.append((l_pmatch, l_hmatch, dt, start, bits_long, l_off, l_inv))
            if best is None or l_pmatch > best[0] or (l_pmatch == best[0] and l_hmatch > best[1]):
                best = (l_pmatch, l_hmatch, dt, start, bits_long, l_off, l_inv)

        if best is None:
            if self.verbose and logged:
                print(f"    → timing search: {rejected_noisy} windows too noisy, {len(rejected_header)} tested headers:")
                for dt, hdr, m, boff, inv, pm in rejected_header:
                    pol = "inv" if inv else "norm"
                    print(f"       dt={dt:+d}: 0x{hdr:04X} ({m}/16 hdr, {pm}/48 pkt, off={boff}, {pol})")
            return
        packet_match, match, dt, start, bits_long, bit_off, invert = best
        bits = bits_long[bit_off:bit_off + PACKET_BITS]
        if invert:
            bits = (1 - bits).astype(np.uint8)
        correction_bits = 0

        if packet_match < PACKET_MIN_MATCH_BITS:
            corrected = self._ecc_vote(candidates)
            if corrected is not None:
                corrected_match = int(np.sum(corrected == EXPECTED_PACKET_BITS))
                if corrected_match > packet_match:
                    correction_bits = int(np.sum(corrected != bits))
                    bits = corrected
                    packet_match = corrected_match
                    got = self._header_from_bits(bits[:16])
                    match = 16 - bin(got ^ expected).count("1")
                    if self.verbose and logged and correction_bits > 0:
                        print(f"    → ECC corrected {correction_bits} bits; packet match now {packet_match}/48")

        if packet_match < PACKET_MIN_MATCH_BITS:
            if self.verbose and logged:
                print(f"    → packet rejected: {packet_match}/48 bits match expected pattern")
            return

        # Accept a near header match; correlation + payload checks below still
        # guard against random noise packets.
        if match < 13:
            if self.verbose and logged:
                expected = (PREAMBLE_BYTE << 8) | SYNC_BYTE
                got = self._header_from_bits(bits[:16])
                print(f"    → dt={dt:+d}: got header 0x{got:04X} (bits: {' '.join(str(b) for b in bits[:16])})")
                print(f"       want        0x{expected:04X} (bits: {' '.join(str(int((expected >> (15-i)) & 1)) for i in range(16))})")
                pol = "inverted" if invert else "normal"
                print(f"       match: {match}/16 Hamming distance (off={bit_off}, {pol})")
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

        # Payload sanity gate: this link is expected to send "OPEN".
        # Keep decodes that look sufficiently close and reject random noise.
        exp_bits = np.array(
            [(b >> (7 - i)) & 1 for b in EXPECTED_PAYLOAD for i in range(8)],
            dtype=np.uint8,
        )
        got_bits = bits[16:48]
        payload_match = int(np.sum(got_bits == exp_bits))
        if payload_match < PAYLOAD_MIN_MATCH_BITS:
            if self.verbose and logged:
                print(f"    → payload rejected: {payload_match}/32 bits match expected \"{EXPECTED_PAYLOAD.decode(errors='ignore')}\"")
            return

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
                f"Header match: {match}/16  Header off: {bit_off}  Polarity: {'inv' if invert else 'norm'}  "
                f"Payload match: {payload_match}/32  Packet match: {packet_match}/48  "
                f"ECC corrected: {correction_bits}")
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
health = MSPHealthMonitor()
start_time = time.time()

print(f"Receiver running. Looking for 'OPEN' packets...")
print(f"Detection frequencies: {CARRIER_OFFSET-SUBCARRIER_HZ:+.0f} Hz and {CARRIER_OFFSET+SUBCARRIER_HZ:+.0f} Hz")
print(f"Noise references: ±{NOISE_REF_DELTA_HZ} Hz around each sideband")
print(f"Envelope despike filter: median kernel {ENV_MEDIAN_KERNEL}")
print(f"Decode thresholds: NCC ≥ {CORR_THRESHOLD:.2f}, bit period ±{TIMING_SEARCH} env samples")
print("Ctrl+C to stop.\n")

frame = 0
# Redraw the spectrum/waterfall only every N frames so rx() stays fed.
PLOT_EVERY = 4
# Print compact RF telemetry periodically so peak visibility does not depend
# on the GUI state.
STATUS_EVERY_PLOT_FRAMES = 6
# Re-lock carrier every RELOCK_FRAMES frames when no packet has been decoded
# recently. This compensates for Pluto TCXO drift during long runs.
RELOCK_FRAMES = 300   # ~every ~40 s at 16 buffers/frame × ~8 ms/buffer
last_relock_frame = 0

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

        # Periodic carrier re-lock: only runs when the decoder hasn't seen
        # a strong packet recently (avoids disrupting an active reception).
        if (frame - last_relock_frame >= RELOCK_FRAMES
                and decoder.last_corr < CORR_THRESHOLD * 0.8
                and time.time() - decoder.last_packet_time > 5.0):
            last_relock_frame = frame
            print("Re-locking carrier (drift correction)...")
            _drift = _locate_carrier(sdr, n_bufs=16, search_hz=CARRIER_FINE_SEARCH_HZ)
            # Ignore implausibly large re-lock jumps that usually indicate a spur.
            if abs(_drift) > 100 and abs(_drift) < 600:
                sdr.rx_lo = int(sdr.rx_lo + _drift)
                for _ in range(4):
                    try: sdr.rx()
                    except Exception: pass
                CARRIER_OFFSET = _locate_carrier(sdr, n_bufs=8, search_hz=CARRIER_FINE_SEARCH_HZ)
            elif abs(_drift) <= 100:
                CARRIER_OFFSET = _drift
            else:
                print(f"Re-lock ignored large drift candidate {_drift:+.1f} Hz")
            print(f"Re-locked: carrier at {CARRIER_OFFSET:+.1f} Hz | "
                  f"sidebands at {CARRIER_OFFSET+SUBCARRIER_HZ:+.0f} Hz "
                  f"and {CARRIER_OFFSET-SUBCARRIER_HZ:+.0f} Hz\n")

        # --- Bit-level detection via matched filter on sideband envelope ---
        env_db, _, _ = sideband_envelope(samples, CARRIER_OFFSET)
        env_db = median_filter_1d(env_db)
        if MSP_HEALTH_ENABLED:
            now_rel = time.time() - start_time
            t0 = now_rel - (len(env_db) - 1) * ENV_WINDOW_MS / 1000.0
            health.push(env_db, t0, ENV_WINDOW_MS / 1000.0)
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

        # Sideband telemetry around the tracked carrier.
        sb_bw_hz = 120
        ns_bw_hz = 120
        sb_hi_f = CARRIER_OFFSET + SUBCARRIER_HZ
        sb_lo_f = CARRIER_OFFSET - SUBCARRIER_HZ
        sb_hi = np.max(psd_zoom[np.abs(freqs_zoom - sb_hi_f) <= sb_bw_hz])
        sb_lo = np.max(psd_zoom[np.abs(freqs_zoom - sb_lo_f) <= sb_bw_hz])
        n1 = np.median(psd_zoom[np.abs(freqs_zoom - (sb_hi_f + NOISE_REF_DELTA_HZ)) <= ns_bw_hz])
        n2 = np.median(psd_zoom[np.abs(freqs_zoom - (sb_hi_f - NOISE_REF_DELTA_HZ)) <= ns_bw_hz])
        n3 = np.median(psd_zoom[np.abs(freqs_zoom - (sb_lo_f + NOISE_REF_DELTA_HZ)) <= ns_bw_hz])
        n4 = np.median(psd_zoom[np.abs(freqs_zoom - (sb_lo_f - NOISE_REF_DELTA_HZ)) <= ns_bw_hz])
        sb_noise = float(np.median([n1, n2, n3, n4]))
        if (frame // PLOT_EVERY) % STATUS_EVERY_PLOT_FRAMES == 0:
            print(f"RF: carrier {peak_power:6.1f} dB | "
                  f"SB+ {sb_hi:6.1f} dB SB- {sb_lo:6.1f} dB | "
                  f"noise {sb_noise:6.1f} dB")

        ax_wide.clear()
        ax_wide.plot(freqs_wide / 1000, psd_wide, linewidth=0.6, color='steelblue')
        ax_wide.axvline(0, color='green', linestyle='--', alpha=0.5)
        ax_wide.set_xlabel('Offset (kHz)')
        ax_wide.set_ylabel('Power (dB)')
        ax_wide.set_title(f'Wide View ±{WIDE_SPAN_HZ/1000:.0f} kHz')
        ax_wide.grid(True, alpha=0.3)
        
        ax_zoom.clear()
        ax_zoom.plot(freqs_zoom, psd_zoom, linewidth=0.8, color='steelblue')
        ax_zoom.axvline(CARRIER_OFFSET, color='green', linestyle='--', alpha=0.5, label='Carrier')
        ax_zoom.axvline(CARRIER_OFFSET + SUBCARRIER_HZ, color='red', linestyle='--', alpha=0.6,
                        label=f'\u00b1{SUBCARRIER_HZ} Hz sideband')
        ax_zoom.axvline(CARRIER_OFFSET - SUBCARRIER_HZ, color='red', linestyle='--', alpha=0.6)
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
        ax_wf.axvline(CARRIER_OFFSET + SUBCARRIER_HZ, color='red', linestyle='--', alpha=0.5)
        ax_wf.axvline(CARRIER_OFFSET - SUBCARRIER_HZ, color='red', linestyle='--', alpha=0.5)
        ax_wf.set_xlabel('Offset from carrier (Hz)')
        ax_wf.set_ylabel('Time →')
        ax_wf.set_title('Sideband Waterfall (bright = modulation present)')
        
        ax_power.clear()
        if len(env_plot) > 0:
            ax_power.plot(list(env_time_plot), list(env_plot),
                          linewidth=1, color='darkblue', label=f'Sideband SNR ({ENV_WINDOW_MS} ms)')
            ax_power.set_xlabel('Time (s)')
            ax_power.set_ylabel('SNR (dB)')
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
finally:
    # Explicit cleanup reduces occasional pylibiio buffer destructor crashes
    # on interpreter shutdown on Windows.
    try:
        if hasattr(sdr, "rx_destroy_buffer"):
            sdr.rx_destroy_buffer()
    except Exception:
        pass
    try:
        del sdr
    except Exception:
        pass