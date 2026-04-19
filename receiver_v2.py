"""
Backscatter OOK receiver (clean rewrite).

Pipeline:
  Pluto IQ  ->  lock CW carrier near DC
            ->  measure sideband envelope at +/- SUBCARRIER_HZ
            ->  matched-filter the preamble+sync pattern
            ->  sample payload bits, check against EXPECTED_PAYLOAD
            ->  live spectrum / waterfall / envelope plot
"""

import time
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
import adi

# ==================== Configuration ====================
CENTER_FREQ = 2_479_991_600
SAMPLE_RATE = 521_000
BUFFER_SIZE = 4096
RX_GAIN = 50

WIDE_SPAN_HZ = 50_000
ZOOM_SPAN_HZ = 12_000
CARRIER_COARSE_SEARCH_HZ = 12_000
CARRIER_FINE_SEARCH_HZ = 1_500

SUBCARRIER_HZ = 2000
NOISE_REF_DELTA_HZ = 350
BIT_DURATION_MS = 10
BIT_DURATION_MIN_MS = 5
BIT_DURATION_MAX_MS = 20
ENV_WINDOW_MS = 5
ENV_SAMPLES = SAMPLE_RATE * ENV_WINDOW_MS // 1000
ENV_PER_BIT = BIT_DURATION_MS // ENV_WINDOW_MS

PREAMBLE_BYTE = 0xAA
SYNC_BYTE = 0x7E
EXPECTED_PAYLOAD = b"OPEN"
PAYLOAD_BYTES = len(EXPECTED_PAYLOAD)
PACKET_BITS = 8 * (2 + PAYLOAD_BYTES)

CORR_THRESHOLD = 0.32
TIMING_SEARCH = 3
BIT_AVG_FRAC = 0.6
ACTIVITY_MIN_SPAN_DB = 2.5
WINDOW_MIN_SPAN_DB = 1.0
PAYLOAD_MIN_MATCH = 24
PACKET_MIN_MATCH = 38
NO_ACTIVITY_RESCAN_SEC = 8.0

_min_epb = max(1, int(round(BIT_DURATION_MIN_MS / ENV_WINDOW_MS)))
_max_epb = max(_min_epb, int(round(BIT_DURATION_MAX_MS / ENV_WINDOW_MS)))
BIT_PERIOD_CANDIDATES = tuple(range(_min_epb, _max_epb + 1))

_sync_bits = np.array(
    [(PREAMBLE_BYTE >> (7 - i)) & 1 for i in range(8)] +
    [(SYNC_BYTE     >> (7 - i)) & 1 for i in range(8)],
    dtype=np.float32,
)
_expected_packet_bits = np.array(
    [(b >> (7 - i)) & 1
     for b in bytes([PREAMBLE_BYTE, SYNC_BYTE]) + EXPECTED_PAYLOAD
     for i in range(8)],
    dtype=np.uint8,
)


def _make_sync_template(epb):
    _t = np.repeat(_sync_bits, epb).astype(np.float32)
    _t_zm = _t - _t.mean()
    return _t_zm, float(np.sqrt(np.sum(_t_zm ** 2)))


_sync_template_cache = {epb: _make_sync_template(epb) for epb in BIT_PERIOD_CANDIDATES}


# ==================== SDR ====================
def connect_sdr():
    print("Connecting to Pluto...")
    try:
        sdr = adi.Pluto("ip:192.168.2.1")
    except Exception:
        sdr = adi.Pluto("usb:")
    sdr.sample_rate             = SAMPLE_RATE
    sdr.rx_lo                   = CENTER_FREQ
    sdr.rx_rf_bandwidth         = SAMPLE_RATE
    sdr.rx_buffer_size          = BUFFER_SIZE
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0   = RX_GAIN
    for attr in ("rx_quadrature_tracking_en_chan0",
                 "rx_rf_dc_offset_tracking_en_chan0",
                 "rx_bb_dc_offset_tracking_en_chan0"):
        try: setattr(sdr, attr, True)
        except Exception: pass
    for _ in range(5):
        try: sdr.rx()
        except Exception: time.sleep(0.1)
    print("Connected.\n")
    return sdr


def drain(sdr, n=4):
    for _ in range(n):
        try: sdr.rx()
        except Exception: pass


def scan_sidebands(sdr, carrier_offset, n_bufs=32,
                   freqs_hz=(500, 750, 1000, 1500, 2000, 2500, 3000,
                             3500, 4000, 5000, 6000, 7000, 8000),
                   noise_delta=NOISE_REF_DELTA_HZ):
    """
    One-shot sideband scan: for each candidate f, measure pair power at
    (carrier +/- f) and compare to adjacent-bin noise at +/- noise_delta.
    Prints a table and returns the frequency with the strongest combined
    score of spectral SNR and envelope activity.
    """
    bufs = []
    for _ in range(n_bufs):
        try: bufs.append(sdr.rx())
        except Exception: pass
    if not bufs:
        return SUBCARRIER_HZ
    block = np.concatenate(bufs)
    N = len(block)
    t = np.arange(N) / SAMPLE_RATE

    def pwr(f):
        return float(np.abs(np.sum(block * np.exp(-2j * np.pi * f * t))) ** 2) / (N * N)

    print("Sideband scan:")
    best_f, best_score = None, -1e9
    for f in freqs_hz:
        sig = pwr(carrier_offset + f) + pwr(carrier_offset - f)
        n = 0.5 * (pwr(carrier_offset + f + noise_delta)
                 + pwr(carrier_offset - f - noise_delta))
        snr = 10 * np.log10((sig + 1e-20) / (n + 1e-20))
        env = sideband_envelope(block, carrier_offset, freq=f)
        span = float(np.percentile(env, 85) - np.percentile(env, 15)) if len(env) else 0.0
        # Favor frequencies that are both spectrally strong and OOK-active.
        score = snr + 2.0 * max(0.0, span - 1.0)
        if score > best_score:
            best_score, best_f = score, f
        print(f"  {f:>5} Hz:  snr={snr:+6.1f} dB  env-span={span:4.2f} dB  score={score:6.2f}")
    print(f"  -> using {best_f} Hz  (score {best_score:.2f})\n")
    return int(best_f) if best_f is not None else SUBCARRIER_HZ


def locate_carrier(sdr, n_bufs=32, search_hz=CARRIER_COARSE_SEARCH_HZ):
    """Return CW carrier offset (Hz) relative to rx_lo."""
    bufs = []
    for _ in range(n_bufs):
        try: bufs.append(sdr.rx())
        except Exception: pass
    if not bufs:
        return 0.0
    block = np.concatenate(bufs)
    win  = np.blackman(len(block))
    spec = np.fft.fftshift(np.fft.fft(block * win))
    fqs  = np.fft.fftshift(np.fft.fftfreq(len(block), 1 / SAMPLE_RATE))
    psd  = np.abs(spec) ** 2
    valid = ((np.abs(fqs) > 100) & (np.abs(fqs) < search_hz)
             & ~((np.abs(fqs) > 800) & (np.abs(fqs) < 1200)))
    if not np.any(valid):
        return 0.0
    vidx = np.flatnonzero(valid)
    vpsd = psd[valid]
    strong = vidx[vpsd >= float(np.max(vpsd)) / 10 ** 0.6]
    idx = int(strong[np.argmin(np.abs(fqs[strong]))])
    offset = float(fqs[idx])
    snr = 10 * np.log10(psd[idx] + 1e-20) - 10 * np.log10(np.median(vpsd) + 1e-20)
    print(f"    peak at {offset:+.1f} Hz  ({snr:.1f} dB above median)")
    return offset


# ==================== Sideband envelope ====================
def sideband_envelope(samples, carrier_offset, freq=SUBCARRIER_HZ,
                      win=ENV_SAMPLES, noise_delta=NOISE_REF_DELTA_HZ):
    """Per-window sideband-to-noise ratio in dB."""
    n = (len(samples) // win) * win
    if n == 0:
        return np.empty(0, dtype=np.float32)
    x = samples[:n].reshape(-1, win)
    t = np.arange(win) / SAMPLE_RATE

    def pwr(f):
        return np.abs(x @ np.exp(-2j * np.pi * f * t) / win) ** 2

    sig = pwr(carrier_offset + freq) + pwr(carrier_offset - freq)
    refs = np.stack((
        pwr(carrier_offset + freq - noise_delta),
        pwr(carrier_offset + freq + noise_delta),
        pwr(carrier_offset - freq + noise_delta),
        pwr(carrier_offset - freq - noise_delta),
    ), axis=1)
    noise = np.mean(np.sort(refs, axis=1)[:, :3], axis=1)
    return (10 * np.log10((sig + 1e-20) / (noise + 1e-20))).astype(np.float32)


def estimate_bit_duration_ms(env_db, min_span_db=ACTIVITY_MIN_SPAN_DB):
    """
    Estimate the MSP's actual bit period from envelope edge spacing.

    Threshold-crosses the envelope at its midline, measures the gap (in
    ENV_WINDOW_MS steps) between consecutive rising+falling edges, and
    returns statistics on the shortest half-bit runs (which correspond
    to 0<->1 transitions at the bit-rate).
    """
    if len(env_db) < 20:
        return None
    lo, hi = np.percentile(env_db, [15, 85])
    if (hi - lo) < min_span_db:
        return None
    th = 0.5 * (lo + hi)
    bits = (env_db > th).astype(np.int8)
    edges = np.flatnonzero(np.diff(bits) != 0)
    if len(edges) < 4:
        return None
    gaps = np.diff(edges).astype(np.float32)          # in ENV_WINDOW_MS units
    # The shortest gaps correspond to single-bit runs (alternating 0/1
    # like in the 0xAA preamble). Use the lower quartile as the estimate.
    short = gaps[gaps <= np.percentile(gaps, 40)]
    if len(short) == 0:
        return None
    est_ms   = float(np.median(short)) * ENV_WINDOW_MS
    min_ms   = float(np.min(short))    * ENV_WINDOW_MS
    n_edges  = int(len(edges))
    return est_ms, min_ms, n_edges


def median_filter_1d(x, kernel=5):
    if len(x) == 0 or kernel <= 1:
        return x
    if kernel % 2 == 0:
        kernel += 1
    pad = kernel // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    return np.array([np.median(xp[i:i + kernel]) for i in range(len(x))],
                    dtype=np.float32)


# ==================== Packet decoder ====================
class PacketDecoder:
    def __init__(self, buffer_seconds=6.0):
        self.env = deque(maxlen=int(buffer_seconds * 1000 / ENV_WINDOW_MS))
        self.last_decoded = ""
        self.last_corr = 0.0
        self.last_packet_time = 0.0
        self.packet_count = 0

    def push(self, env_db):
        self.env.extend(env_db.tolist())

    @staticmethod
    def _sample_bits(env, start, n_bits, thresh, epb):
        half = int(epb * BIT_AVG_FRAC) // 2
        out = np.empty(n_bits, dtype=np.uint8)
        for i in range(n_bits):
            c = start + i * epb + epb // 2
            a, b = max(0, c - half), min(len(env), c + half + 1)
            out[i] = 1 if env[a:b].mean() > thresh else 0
        return out

    @staticmethod
    def _best_ncc(env_zm, tmpl_zm, tmpl_norm, max_start):
        tlen = len(tmpl_zm)
        if len(env_zm) < tlen + 1 or max_start <= 0:
            return -1.0, 0
        num = np.correlate(env_zm, tmpl_zm, mode="valid")[:max_start + 1]
        if num.size == 0:
            return -1.0, 0
        csum = np.concatenate(([0.0], np.cumsum(env_zm.astype(np.float64) ** 2)))
        sig_e = csum[tlen:tlen + len(num)] - csum[:len(num)]
        ncc = num / np.sqrt(sig_e * tmpl_norm ** 2 + 1e-20)
        idx = int(np.argmax(ncc))
        return float(ncc[idx]), idx

    def try_decode(self):
        epb_candidates = BIT_PERIOD_CANDIDATES
        env = np.asarray(self.env, dtype=np.float32)
        est = estimate_bit_duration_ms(env, min_span_db=ACTIVITY_MIN_SPAN_DB)
        if est is not None:
            est_ms = est[0]
            est_epb = int(round(est_ms / ENV_WINDOW_MS))
            lo_epb = max(BIT_PERIOD_CANDIDATES[0], est_epb - 1)
            hi_epb = min(BIT_PERIOD_CANDIDATES[-1], est_epb + 1)
            epb_candidates = tuple(range(lo_epb, hi_epb + 1))

        max_epb = epb_candidates[-1]
        if len(self.env) < PACKET_BITS * max_epb + 16 * max_epb:
            return

        lo, hi = np.percentile(env, [15, 85])
        if (hi - lo) < ACTIVITY_MIN_SPAN_DB:
            self.last_corr = 0.0
            return

        env_zm = env - env.mean()

        best_peak, best_epb, best_pos = -1.0, ENV_PER_BIT, 0
        for epb in epb_candidates:
            tmpl, tnorm = _sync_template_cache[epb]
            val, pos = self._best_ncc(env_zm, tmpl, tnorm, len(env) - PACKET_BITS * epb)
            if val > best_peak:
                best_peak, best_epb, best_pos = val, epb, pos
        self.last_corr = best_peak
        if best_peak < CORR_THRESHOLD:
            return

        epb = best_epb
        packet_len = PACKET_BITS * epb
        best = None
        for dt in range(-TIMING_SEARCH, TIMING_SEARCH + 1):
            start = best_pos + dt
            if start < 0 or start + packet_len > len(env):
                continue
            win = env[start:start + packet_len]
            w_lo, w_hi = np.percentile(win, [15, 85])
            if (w_hi - w_lo) < WINDOW_MIN_SPAN_DB:
                continue
            th = 0.5 * (w_lo + w_hi)
            bits = self._sample_bits(env, start, PACKET_BITS, th, epb)
            pkt_match = int(np.sum(bits == _expected_packet_bits))
            if best is None or pkt_match > best[0]:
                best = (pkt_match, dt, start, bits)

        if best is None:
            return
        pkt_match, dt, start, bits = best
        if pkt_match < PACKET_MIN_MATCH:
            return
        payload_match = int(np.sum(bits[16:] == _expected_packet_bits[16:]))
        if payload_match < PAYLOAD_MIN_MATCH:
            return

        raw = []
        message = ""
        for bi in range(PAYLOAD_BYTES):
            v = 0
            for j in range(8):
                v = (v << 1) | int(bits[16 + bi * 8 + j])
            raw.append(v)
            message += chr(v) if 32 <= v < 127 else "?"

        now = time.time()
        if now - self.last_packet_time > 0.5:
            self.packet_count += 1
            self.last_decoded = message
            self.last_packet_time = now
            print(f"\n{'=' * 60}")
            print(f"  PACKET #{self.packet_count}  \"{message}\"  "
                  f"[{' '.join('0x%02X' % v for v in raw)}]")
            print(f"  ncc={best_peak:.3f}  epb={epb}  dt={dt:+d}  "
                  f"match={pkt_match}/48  payload={payload_match}/32")
            print(f"{'=' * 60}\n")

        for _ in range(min(start + packet_len, len(self.env))):
            self.env.popleft()


# ==================== Main ====================
def main():
    sdr = connect_sdr()

    print("Locating carrier (coarse)...")
    sdr.rx_lo = int(sdr.rx_lo + locate_carrier(sdr, search_hz=CARRIER_COARSE_SEARCH_HZ))
    drain(sdr)
    print("Locating carrier (fine)...")
    sdr.rx_lo = int(sdr.rx_lo + locate_carrier(sdr, search_hz=CARRIER_FINE_SEARCH_HZ))
    drain(sdr)
    print("Measuring residual offset...")
    carrier_offset = locate_carrier(sdr, n_bufs=16, search_hz=CARRIER_FINE_SEARCH_HZ)
    print(f"Carrier at {carrier_offset:+.1f} Hz\n")

    subcarrier_hz = scan_sidebands(sdr, carrier_offset)
    print(f"Sidebands at {carrier_offset + subcarrier_hz:+.0f} Hz and "
          f"{carrier_offset - subcarrier_hz:+.0f} Hz\n")

    accum_len = 65536
    n_bufs_per_frame = accum_len // BUFFER_SIZE
    freqs = np.fft.fftshift(np.fft.fftfreq(accum_len, 1 / SAMPLE_RATE))
    wide_mask = (freqs >= -WIDE_SPAN_HZ) & (freqs <= WIDE_SPAN_HZ)
    zoom_mask = (freqs >= -ZOOM_SPAN_HZ) & (freqs <= ZOOM_SPAN_HZ)
    freqs_wide = freqs[wide_mask]
    freqs_zoom = freqs[zoom_mask]
    fft_win = np.hanning(accum_len)
    fft_gain = float(np.sum(fft_win ** 2))

    plt.ion()
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.8])
    ax_wide  = fig.add_subplot(gs[0, 0])
    ax_zoom  = fig.add_subplot(gs[0, 1])
    ax_wf    = fig.add_subplot(gs[1, :])
    ax_power = fig.add_subplot(gs[2, :])

    wf_rows = 150
    wf = np.full((wf_rows, len(freqs_zoom)), -100.0)
    psd_avg = None

    plot_history_sec = 6.0
    env_plot      = deque(maxlen=int(plot_history_sec * 1000 / ENV_WINDOW_MS))
    env_time_plot = deque(maxlen=int(plot_history_sec * 1000 / ENV_WINDOW_MS))

    decoder = PacketDecoder(buffer_seconds=plot_history_sec)
    t_start = time.time()

    print(
        f"Receiver running. Looking for \"{EXPECTED_PAYLOAD.decode()}\" packets. "
        f"Ctrl+C to stop.\n"
        f"Bit timing: nominal={BIT_DURATION_MS} ms, search={BIT_DURATION_MIN_MS}-{BIT_DURATION_MAX_MS} ms, "
        f"epb candidates={BIT_PERIOD_CANDIDATES}\n"
    )

    PLOT_EVERY = 4
    frame = 0
    last_bitdur_print = 0.0
    last_activity_ts = time.time()
    try:
        while True:
            bufs = []
            for _ in range(n_bufs_per_frame):
                try: bufs.append(sdr.rx())
                except Exception: pass
            if len(bufs) < n_bufs_per_frame:
                continue
            samples = np.concatenate(bufs)

            env_db = median_filter_1d(sideband_envelope(samples, carrier_offset, freq=subcarrier_hz))
            decoder.push(env_db)
            decoder.try_decode()

            if len(env_db):
                env_span = float(np.percentile(env_db, 85) - np.percentile(env_db, 15))
                if env_span >= ACTIVITY_MIN_SPAN_DB:
                    last_activity_ts = time.time()
                elif (time.time() - last_activity_ts) > NO_ACTIVITY_RESCAN_SEC:
                    print(f"[retune] low activity for {NO_ACTIVITY_RESCAN_SEC:.0f}s, re-scanning sidebands...")
                    subcarrier_hz = scan_sidebands(sdr, carrier_offset)
                    print(f"[retune] now using +/-{subcarrier_hz} Hz sidebands")
                    last_activity_ts = time.time()

            now_rel = time.time() - t_start
            for i, v in enumerate(env_db):
                env_plot.append(float(v))
                env_time_plot.append(now_rel - (len(env_db) - 1 - i) * ENV_WINDOW_MS / 1000.0)

            # Print measured MSP bit duration once per second when no decodes.
            if now_rel - last_bitdur_print >= 1.0 and len(env_plot) >= 100:
                est = estimate_bit_duration_ms(np.asarray(env_plot, dtype=np.float32))
                last_bitdur_print = now_rel
                if est is None:
                    print(f"[bit-dur] no activity (envelope span < "
                          f"{ACTIVITY_MIN_SPAN_DB:.1f} dB)")
                else:
                    est_ms, min_ms, n_edges = est
                    expected = BIT_DURATION_MS
                    ratio = est_ms / expected if expected else 0.0
                    note = ""
                    if abs(ratio - 1.0) > 0.15:
                        note = f"  <-- MSP is ~{ratio:.2f}x expected"
                    print(f"[bit-dur] median={est_ms:5.1f} ms  "
                          f"min={min_ms:5.1f} ms  edges={n_edges:3d}  "
                          f"(expected {expected} ms){note}")

            frame += 1
            if frame % PLOT_EVERY != 0:
                continue

            fft = np.fft.fftshift(np.fft.fft(samples * fft_win))
            psd = np.abs(fft) ** 2 / fft_gain
            psd_avg = psd.copy() if psd_avg is None else 0.7 * psd_avg + 0.3 * psd
            psd_db   = 10 * np.log10(psd_avg + 1e-20)
            psd_wide = psd_db[wide_mask]
            psd_zoom = psd_db[zoom_mask]
            peak_power = float(np.max(psd_zoom))

            ax_wide.clear()
            ax_wide.plot(freqs_wide / 1000, psd_wide, lw=0.6, color="steelblue")
            ax_wide.axvline(0, color="green", ls="--", alpha=0.5)
            ax_wide.set_xlabel("Offset (kHz)")
            ax_wide.set_ylabel("Power (dB)")
            ax_wide.set_title(f"Wide View +/-{WIDE_SPAN_HZ / 1000:.0f} kHz")
            ax_wide.grid(True, alpha=0.3)

            ax_zoom.clear()
            ax_zoom.plot(freqs_zoom, psd_zoom, lw=0.8, color="steelblue")
            ax_zoom.axvline(carrier_offset, color="green", ls="--", alpha=0.5, label="Carrier")
            ax_zoom.axvline(carrier_offset + subcarrier_hz, color="red", ls="--", alpha=0.6,
                            label=f"+/-{subcarrier_hz} Hz sideband")
            ax_zoom.axvline(carrier_offset - subcarrier_hz, color="red", ls="--", alpha=0.6)
            ax_zoom.set_xlabel("Offset from carrier (Hz)")
            ax_zoom.set_ylabel("Power (dB)")
            ax_zoom.set_title(f"Zoomed View | Carrier {peak_power:.1f} dB")
            ax_zoom.set_xlim(-ZOOM_SPAN_HZ, ZOOM_SPAN_HZ)
            ax_zoom.grid(True, alpha=0.3)
            ax_zoom.legend(loc="upper right", fontsize=8)

            wf = np.roll(wf, -1, axis=0)
            wf[-1, :] = psd_zoom
            ax_wf.clear()
            ax_wf.imshow(wf, aspect="auto", cmap="viridis",
                         extent=[-ZOOM_SPAN_HZ, ZOOM_SPAN_HZ, 0, wf_rows],
                         vmin=np.percentile(wf, 10), vmax=np.percentile(wf, 99),
                         origin="lower")
            ax_wf.axvline(carrier_offset + subcarrier_hz, color="red", ls="--", alpha=0.5)
            ax_wf.axvline(carrier_offset - subcarrier_hz, color="red", ls="--", alpha=0.5)
            ax_wf.set_xlabel("Offset from carrier (Hz)")
            ax_wf.set_ylabel("Time")
            ax_wf.set_title("Sideband Waterfall")

            ax_power.clear()
            if env_plot:
                ax_power.plot(list(env_time_plot), list(env_plot),
                              lw=1, color="darkblue")
                ax_power.set_xlabel("Time (s)")
                ax_power.set_ylabel("SNR (dB)")
                ax_power.set_title(
                    f'Sideband envelope  |  packets: {decoder.packet_count}  |  '
                    f'last: "{decoder.last_decoded}"  |  ncc: {decoder.last_corr:.2f}'
                )
                ax_power.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.pause(0.01)

    except KeyboardInterrupt:
        print(f"\n\nStopped. Total packets: {decoder.packet_count}")
        print(f'Last message: "{decoder.last_decoded}"')
    finally:
        try:
            if hasattr(sdr, "rx_destroy_buffer"):
                sdr.rx_destroy_buffer()
        except Exception:
            pass
        try:
            del sdr
        except Exception:
            pass


if __name__ == "__main__":
    main()
