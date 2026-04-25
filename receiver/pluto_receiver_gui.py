#!/usr/bin/env python3
"""PlutoSDR GUI receiver for raw baseband visualization.

Shows:
- Time-domain I/Q samples near baseband
- Baseband spectrum around DC

Close the window or press Ctrl+C in the terminal to stop.
"""

import signal
import sys
from types import FrameType

import adi
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sp
from matplotlib.animation import FuncAnimation

RX_URI = "ip:192.168.2.1"
FREQ_HZ = 2.48e9
SAMPLE_RATE = 1e6
RX_GAIN_MODE: str = "manual"
RX_GAIN_DB = 40.0
RX_BUFFER_SIZE = 65536       # 3.8 Hz/bin — 263 bins between carrier and ±1 kHz sideband
ADC_FULL_SCALE = 2048.0
TIME_SAMPLES = 1200
SPECTRUM_SPAN_HZ = 150000.0
FFT_AVG_ALPHA = 0.1           # ~10-frame memory — ~1s response, good noise rejection
EXCITER_SEARCH_MIN_HZ = 500.0   # ignore near-DC bins
EXCITER_SEARCH_MAX_HZ = 40000.0 # prevent false lock on far-out noise peaks
CENTERED_SPAN_HZ = 8000.0
WATERFALL_ROWS = 120
WATERFALL_BINS = 512          # lower interpolation cost per frame
WATERFALL_DYN_RANGE_DB = 28.0 # wider range avoids center saturation and preserves discrete lines
DC_BLOCK_ALPHA = 0.9998       # cutoff ≈ 32 Hz — well below 1 kHz signal
RX_RF_BANDWIDTH = 200_000     # AD9361 on-chip analog LPF — rejects noise outside 200 kHz
ENABLE_FOCUSED_FILTER = False # default off: avoids filter transients/blueout artifacts
FOCUSED_PASSBAND_HZ = 2500.0
SIDEBAND_OFFSET_KHZ = 1.0     # 1 kHz subcarrier offset
SIDEBAND_WINDOW_HZ = 30.0     # ±30 Hz at 3.8 Hz/bin = ±8 bins — tight enough to avoid noise peaks
SNR_LOCK_THRESHOLD_DB = 20.0  # minimum SNR for a reliable backscatter decode
ENV_Y_SMOOTH_ALPHA = 0.15     # envelope y-limit smoothing (higher = more responsive)
ENV_Y_MIN_SPAN = 0.01         # minimum y-span for envelope plot stability
CENTERED_FREQ_SMOOTH_BINS = 3 # lighter smoothing to keep sidebands sharp
WATERFALL_ROW_BLEND = 0.75    # favor current frame; less temporal blur
BIT_DURATION_MS = 50.0
BIT_NCC_THRESHOLD = 0.10
PREAMBLE_BYTES = b"\xAA\xAA"
SYNC_BYTES = b"\xD3\x91"
PAYLOAD_BYTES = b"OPEN"
REPETITION_CHIPS = 3
NCC_DISPLAY_ALPHA = 0.2
NCC_ENTER_THRESHOLD = 0.1
NCC_EXIT_THRESHOLD = 0.05
NCC_ENTER_FRAMES = 3
NCC_EXIT_FRAMES = 8
PACKET_STATUS_HOLD_FRAMES = 20
TERMINAL_DEBUG_BITS_EVERY = 20
TERMINAL_DEBUG_BIT_TAIL = 64
BIT_PHASE_STEP_SAMPLES = 5000
PHASE_HISTORY_BITS = 128
HEADER_MAX_BIT_ERRORS = 4


def normalize_iq(raw: object) -> np.ndarray:
    samples = np.asarray(raw, dtype=np.complex64)
    return samples / np.float32(ADC_FULL_SCALE)


def remove_dc(x: np.ndarray) -> np.ndarray:
    return x - np.mean(x)


def dc_block_filter(
    x: np.ndarray,
    x_prev: np.complex64,
    y_prev: np.complex64,
    alpha: float,
) -> tuple[np.ndarray, np.complex64, np.complex64]:
    """Vectorised first-order IIR DC-block: y[n] = x[n] - x[n-1] + alpha*y[n-1].

    Initial condition for scipy direct-form-II transposed:
      zi[0] = b[1]*x[-1] - a[1]*y[-1]  where b=[1,-1], a=[1,-alpha]
            = -x_prev + alpha*y_prev
    """
    b = np.array([1.0, -1.0])
    a = np.array([1.0, -float(alpha)])
    zi = np.array([-x_prev + float(alpha) * y_prev], dtype=np.complex128)
    y, _ = sp.lfilter(b, a, x.astype(np.complex128), zi=zi)
    y = y.astype(np.complex64)
    return y, np.complex64(x[-1]), np.complex64(y[-1])


def compute_sideband_snr(
    centered_freqs_khz: np.ndarray,
    centered_spec_dbfs: np.ndarray,
    sideband_offset_khz: float,
    window_hz: float,
) -> tuple[float, float, float, float]:
    """Return (snr_db, sb_pos_dbfs, sb_neg_dbfs, noise_floor_dbfs, sb_pos_khz, sb_neg_khz).

    Sideband bins: within ±window_hz of ±sideband_offset_khz.
    Noise estimate: median of bins between 0.3–4.0 kHz excluding sideband windows.
    Returns (snr_db, sb_pos_peak, sb_neg_peak, noise_floor).
    """
    w = window_hz / 1000.0
    pos_mask = np.abs(centered_freqs_khz - sideband_offset_khz) <= w
    neg_mask = np.abs(centered_freqs_khz + sideband_offset_khz) <= w
    sb_mask = pos_mask | neg_mask
    # Use only positive-offset bins for noise reference — the negative side often
    # carries exciter phase noise that inflates the noise estimate and suppresses SNR.
    noise_mask = (
        (centered_freqs_khz > 0.3)
        & (centered_freqs_khz < (CENTERED_SPAN_HZ / 1000.0 - 0.5))
        & ~sb_mask
    )
    if not sb_mask.any() or not noise_mask.any():
        return 0.0, -140.0, -140.0, -140.0
    sb_pos = float(np.max(centered_spec_dbfs[pos_mask])) if pos_mask.any() else -140.0
    sb_neg = float(np.max(centered_spec_dbfs[neg_mask])) if neg_mask.any() else -140.0
    noise_floor = float(np.percentile(centered_spec_dbfs[noise_mask], 50))
    snr = max(sb_pos, sb_neg) - noise_floor
    return snr, sb_pos, sb_neg, noise_floor


def coherent_ncc(
    x: np.ndarray,
    carrier_hz: float,
    mod_hz: float,
    sample_rate: float,
) -> tuple[float, np.ndarray]:
    """Coherent AM demodulation + NCC against a square-wave reference.

    Steps:
    1. Mix carrier to DC: multiply by exp(-j*2π*carrier_hz*n/fs)
    2. Take magnitude envelope — this is the AM modulation signal
    3. Remove DC from envelope (mean subtraction)
    4. Build ideal square-wave reference at mod_hz
    5. Compute normalised cross-correlation → NCC in [-1, 1]
       NCC > 0.1 reliably indicates a coherent 1 kHz tone.

    Returns (ncc_value, envelope) so the caller can plot the envelope.
    """
    n = np.arange(len(x), dtype=np.float64)
    # Step 1: mix carrier to DC
    mix = np.exp(-1j * 2.0 * np.pi * carrier_hz * n / sample_rate)
    baseband = x.astype(np.complex128) * mix
    # Step 2: envelope
    envelope = np.abs(baseband).astype(np.float64)
    # Step 3: remove DC
    envelope -= envelope.mean()
    # Step 4: square-wave reference (±1), same length
    ref = np.sign(np.sin(2.0 * np.pi * mod_hz * n / sample_rate))
    # Step 5: NCC
    env_rms = float(np.sqrt(np.mean(envelope ** 2)))
    ref_rms = float(np.sqrt(np.mean(ref ** 2)))
    if env_rms < 1e-12 or ref_rms < 1e-12:
        return 0.0, envelope
    ncc = float(np.mean(envelope * ref) / (env_rms * ref_rms))
    return ncc, envelope


def compute_spectrum_dbfs(x: np.ndarray, sample_rate: float) -> tuple[np.ndarray, np.ndarray]:
    """Blackman-Harris window: -92 dB sidelobes vs -31 dB for Hanning.

    Lower sidelobes prevent the strong carrier from leaking into adjacent
    bins and masking the weaker 1 kHz backscatter sidebands.
    """
    # Restore Blackman window — carrier is 9 kHz from LO, sidebands ±1 kHz from carrier.
    # 8 kHz separation is far beyond Blackman's main lobe; -92 dB sidelobes keep carrier
    # leakage out of the sideband bins.
    window = np.blackman(len(x)).astype(np.float32)
    spectrum = np.fft.fftshift(np.fft.fft(x * window))
    coherent_gain = float(np.mean(window))
    mag = np.abs(spectrum) / max(len(x) * coherent_gain, 1e-12)
    spectrum_dbfs = 20.0 * np.log10(np.maximum(mag, 1e-12))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(x), d=1.0 / sample_rate))
    return freqs, spectrum_dbfs


def bandpass_filter_around_carrier(
    x: np.ndarray,
    carrier_hz: float,
    sample_rate: float,
    passband_hz: float = 2000.0,
) -> np.ndarray:
    """Carrier-centered filter using heterodyne + low-pass.

    1) Mix carrier_hz to DC.
    2) Low-pass to keep ±passband_hz.
    3) Mix back to original center frequency.

    This works for any carrier offset (including nonzero offsets from Pluto LO drift).
    """
    if passband_hz <= 0.0 or len(x) == 0:
        return x

    nyquist = sample_rate / 2.0
    wn = min(passband_hz / nyquist, 0.99)
    if wn <= 0.0:
        return x

    n = np.arange(len(x), dtype=np.float64)
    w = 2.0 * np.pi * float(carrier_hz) / float(sample_rate)
    downmix = np.exp(-1j * w * n)
    upmix = np.exp(1j * w * n)

    x_shifted = x.astype(np.complex128) * downmix
    sos = sp.butter(4, wn, btype="lowpass", output="sos")
    y_shifted = sp.sosfilt(sos, x_shifted)
    y = y_shifted * upmix
    return y.astype(np.complex64)


def bytes_to_bit_list(data: bytes) -> list[int]:
    bits: list[int] = []
    for byte in data:
        for shift in range(7, -1, -1):
            bits.append((byte >> shift) & 1)
    return bits


def bits_to_bytes(bits: list[int]) -> bytes:
    if len(bits) % 8 != 0:
        return b""
    out = bytearray()
    for i in range(0, len(bits), 8):
        value = 0
        for bit in bits[i : i + 8]:
            value = (value << 1) | int(bit)
        out.append(value)
    return bytes(out)


def find_subsequence(bits: list[int], pattern: list[int], start: int = 0) -> int:
    if not pattern or len(bits) < len(pattern):
        return -1
    max_idx = len(bits) - len(pattern)
    for i in range(max(0, start), max_idx + 1):
        if bits[i : i + len(pattern)] == pattern:
            return i
    return -1


def find_header_match(
    bits: list[int],
    pattern: list[int],
    start: int = 0,
    max_errors: int = 0,
) -> tuple[int, int]:
    if not pattern or len(bits) < len(pattern):
        return -1, len(pattern)
    best_idx = -1
    best_errors = len(pattern) + 1
    max_idx = len(bits) - len(pattern)
    for i in range(max(0, start), max_idx + 1):
        errors = sum(1 for left, right in zip(bits[i : i + len(pattern)], pattern) if left != right)
        if errors < best_errors:
            best_idx = i
            best_errors = errors
        if errors <= max_errors:
            return i, errors
    if best_errors <= max_errors:
        return best_idx, best_errors
    return -1, best_errors


def bits_to_text(bits: list[int]) -> str:
    return "".join("1" if b else "0" for b in bits)


def safe_ascii(data: bytes) -> str:
    return data.decode("ascii", errors="replace") if data else ""


def majority_decode_triplets(chips: list[int], start_offset: int) -> list[int]:
    decoded: list[int] = []
    idx = start_offset
    while idx + REPETITION_CHIPS <= len(chips):
        triplet = chips[idx : idx + REPETITION_CHIPS]
        decoded.append(1 if sum(triplet) >= 2 else 0)
        idx += REPETITION_CHIPS
    return decoded


def main() -> int:
    sdr = adi.Pluto(RX_URI)
    sdr.rx_enabled_channels = [0]
    sdr.sample_rate = int(SAMPLE_RATE)
    sdr.rx_lo = int(FREQ_HZ)
    sdr.rx_rf_bandwidth = int(SAMPLE_RATE)  # full 1 MHz — let sideband spread through
    sdr.rx_buffer_size = int(RX_BUFFER_SIZE)
    sdr.gain_control_mode_chan0 = RX_GAIN_MODE
    if RX_GAIN_MODE == "manual":
        sdr.rx_hardwaregain_chan0 = float(RX_GAIN_DB)

    stop_requested = False

    def _request_stop(_sig: int, _frame: FrameType | None) -> None:
        nonlocal stop_requested
        stop_requested = True
        plt.close("all")

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    fig, (ax_time, ax_fft) = plt.subplots(2, 1, figsize=(11, 7))
    fig.canvas.manager.set_window_title("Pluto Baseband Receiver")
    fig.suptitle("PlutoSDR Raw Baseband View")

    carrier_fig, (ax_centered, ax_waterfall) = plt.subplots(2, 1, figsize=(10, 7))
    carrier_fig.canvas.manager.set_window_title("Pluto Carrier Detail")
    carrier_fig.suptitle("Auto-Centered Carrier View")

    # Coherent demodulator window — NCC time trace + AM envelope
    NCC_HISTORY = 200
    ncc_fig, (ax_ncc, ax_env) = plt.subplots(2, 1, figsize=(10, 5))
    ncc_fig.canvas.manager.set_window_title("Coherent 1 kHz Demodulator")
    ncc_fig.suptitle("Coherent AM Demodulation — 1 kHz Square-Wave NCC")
    ncc_history = np.zeros(NCC_HISTORY, dtype=np.float64)
    ncc_time_axis = np.arange(NCC_HISTORY)
    (line_ncc,) = ax_ncc.plot(ncc_time_axis, ncc_history, lw=1.2, color="C2")
    ax_ncc.axhline(0.0,  color="0.5", lw=0.8, linestyle="--")
    ax_ncc.axhline( 0.1, color="lime",  lw=0.8, linestyle=":", label="detect threshold")
    ax_ncc.axhline(-0.1, color="lime",  lw=0.8, linestyle=":")
    ax_ncc.set_ylim(-1.0, 1.0)
    ax_ncc.set_ylabel("NCC")
    ax_ncc.set_xlabel("Frame")
    ax_ncc.set_title("Normalised Cross-Correlation vs 1 kHz Square Wave")
    ax_ncc.legend(loc="upper right", fontsize=8)
    ax_ncc.grid(True, alpha=0.3)
    env_plot_len = min(int(SAMPLE_RATE * (BIT_DURATION_MS / 1000.0)), RX_BUFFER_SIZE)  # 50 ms envelope window
    (line_env,) = ax_env.plot(np.arange(env_plot_len), np.zeros(env_plot_len), lw=0.8, color="C0")
    ax_env.set_ylabel("AM Envelope (AC)")
    ax_env.set_xlabel("Sample")
    ax_env.set_title("Demodulated Envelope (50 ms window)")
    ax_env.set_ylim(-0.02, 0.02)
    ax_env.grid(True, alpha=0.3)
    ncc_status = ncc_fig.text(0.01, 0.01, "Waiting...", ha="left", va="bottom")
    ncc_fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    env_ylim_low = -0.02
    env_ylim_high = 0.02


    time_axis = np.arange(TIME_SAMPLES)
    (line_i,) = ax_time.plot(time_axis, np.zeros(TIME_SAMPLES), label="I", lw=1.0)
    (line_q,) = ax_time.plot(time_axis, np.zeros(TIME_SAMPLES), label="Q", lw=1.0)
    ax_time.set_title("Time Domain (first samples)")
    ax_time.set_xlabel("Sample")
    ax_time.set_ylabel("Amplitude (normalized)")
    ax_time.set_ylim(-1.1, 1.1)
    ax_time.grid(True, alpha=0.3)
    ax_time.legend(loc="upper right")

    fft_axis = np.linspace(-SPECTRUM_SPAN_HZ / 1000.0, SPECTRUM_SPAN_HZ / 1000.0, 512)
    (line_fft_raw,) = ax_fft.plot(
        fft_axis,
        np.full_like(fft_axis, -140.0),
        lw=0.9,
        color="0.65",
        label="Raw spectrum",
    )
    (line_fft_dc_blocked,) = ax_fft.plot(
        fft_axis,
        np.full_like(fft_axis, -140.0),
        lw=1.2,
        color="C0",
        label="Processed spectrum",
    )
    exciter_marker = ax_fft.axvline(0.0, color="C3", lw=1.0, alpha=0.8, label="Exciter peak")
    ax_fft.axvline(0.0, color="0.3", lw=0.8, alpha=0.35, linestyle="--")
    ax_fft.set_title("Spectrum Near Baseband")
    ax_fft.set_xlabel("Frequency Offset (kHz)")
    ax_fft.set_ylabel("Magnitude (dBFS)")
    ax_fft.set_ylim(-140.0, 5.0)
    ax_fft.grid(True, alpha=0.3)
    ax_fft.legend(loc="lower left")

    centered_axis = np.linspace(-CENTERED_SPAN_HZ / 1000.0, CENTERED_SPAN_HZ / 1000.0, WATERFALL_BINS)
    (line_centered,) = ax_centered.plot(
        centered_axis,
        np.full_like(centered_axis, -140.0),
        lw=1.2,
        color="C0",
    )
    ax_centered.axvline(0.0, color="C3", lw=1.0, alpha=0.8)
    # Mark the expected ±1 kHz backscatter sidebands
    ax_centered.axvline( SIDEBAND_OFFSET_KHZ, color="lime", lw=0.9, alpha=0.7, linestyle="--", label="±1 kHz SB")
    ax_centered.axvline(-SIDEBAND_OFFSET_KHZ, color="lime", lw=0.9, alpha=0.7, linestyle="--")
    ax_centered.legend(loc="upper left", fontsize=8)
    ax_centered.set_title("Carrier-Centered Spectrum")
    ax_centered.set_xlabel("Offset From Carrier (kHz)")
    ax_centered.set_ylabel("Magnitude (dBFS)")
    ax_centered.set_ylim(-120.0, -60.0)
    ax_centered.grid(True, alpha=0.3)
    # Scatter markers: show measured sideband peak power each frame
    # White outline diamonds — fill colour switches red/green based on SNR threshold
    sideband_scatter = ax_centered.scatter(
        [-SIDEBAND_OFFSET_KHZ, SIDEBAND_OFFSET_KHZ],
        [-140.0, -140.0],
        c=["#ff4444", "#ff4444"], s=80, zorder=5,
        marker="D", edgecolors="white", linewidths=0.8,
        label="SB peaks"
    )
    # Horizontal threshold line: noise_floor + 20 dB target
    snr_threshold_line = ax_centered.axhline(
        -140.0, color="#ff4444", lw=1.0, linestyle=":", alpha=0.85, label=f"SNR={SNR_LOCK_THRESHOLD_DB:.0f}dB target"
    )
    ax_centered.legend(loc="upper left", fontsize=8)

    waterfall_data = np.full((WATERFALL_ROWS, centered_axis.size), -140.0, dtype=np.float64)
    waterfall_img = ax_waterfall.imshow(
        waterfall_data,
        aspect="auto",
        origin="lower",
        extent=[centered_axis[0], centered_axis[-1], 0, WATERFALL_ROWS],
        cmap="plasma",      # better perceptual contrast for weak signals than viridis
        vmin=-140.0,
        vmax=-20.0,
    )
    ax_waterfall.set_title("Carrier-Centered Waterfall")
    ax_waterfall.set_xlabel("Offset From Carrier (kHz)")
    ax_waterfall.set_ylabel("Frame")
    # Sideband markers on waterfall
    ax_waterfall.axvline( SIDEBAND_OFFSET_KHZ, color="lime", lw=0.8, alpha=0.6, linestyle="--")
    ax_waterfall.axvline(-SIDEBAND_OFFSET_KHZ, color="lime", lw=0.8, alpha=0.6, linestyle="--")

    status = fig.text(0.01, 0.01, "Connecting...", ha="left", va="bottom")
    carrier_status = carrier_fig.text(0.01, 0.01, "Centering...", ha="left", va="bottom")
    smoothed_raw_spec_dbfs = np.array([], dtype=np.float64)
    smoothed_dc_blocked_spec_dbfs = np.array([], dtype=np.float64)
    dc_prev_x = np.complex64(0.0 + 0.0j)
    dc_prev_y = np.complex64(0.0 + 0.0j)
    prev_peak_dbfs: float | None = None
    prev_peak_hz = 0.0
    bit_samples = int(SAMPLE_RATE * (BIT_DURATION_MS / 1000.0))
    phase_offsets = list(range(0, bit_samples, BIT_PHASE_STEP_SAMPLES))
    phase_sample_buffer = np.array([], dtype=np.complex64)
    packet_header_bits = bytes_to_bit_list(PREAMBLE_BYTES + SYNC_BYTES)
    payload_bits_len = len(PAYLOAD_BYTES) * 8
    packet_status_text = "Waiting for preamble+sync"
    packet_status_hold = 0
    ncc_abs_ema = 0.0
    ncc_lock = False
    ncc_enter_count = 0
    ncc_exit_count = 0
    total_bits = 0
    next_debug_bits = TERMINAL_DEBUG_BITS_EVERY
    decoded_packets = 0
    phase_state: dict[int, dict[str, object]] = {
        phase: {
            "next_sample": phase,
            "chips": [],
            "base_chip_index": 0,
            "search_start_by_offset": [0, 0, 0],
            "last_header_abs_by_offset": [-1, -1, -1],
            "chips_seen": 0,
        }
        for phase in phase_offsets
    }

    def _close_event(_event: object) -> None:
        nonlocal stop_requested
        stop_requested = True

    fig.canvas.mpl_connect("close_event", _close_event)
    carrier_fig.canvas.mpl_connect("close_event", _close_event)
    ncc_fig.canvas.mpl_connect("close_event", _close_event)

    def update(_frame: int):
        if stop_requested:
            return (
                line_i,
                line_q,
                line_fft_raw,
                line_fft_dc_blocked,
                exciter_marker,
                line_centered,
                waterfall_img,
                status,
                carrier_status,
            )

        nonlocal smoothed_raw_spec_dbfs, smoothed_dc_blocked_spec_dbfs, dc_prev_x, dc_prev_y, prev_peak_dbfs, prev_peak_hz, ncc_history, env_ylim_low, env_ylim_high, phase_sample_buffer, packet_status_text, packet_status_hold, ncc_abs_ema, ncc_lock, ncc_enter_count, ncc_exit_count, total_bits, next_debug_bits, decoded_packets

        x_raw = normalize_iq(sdr.rx())
        # IIR DC block — maintains state across frames, ~32 Hz cutoff, unity gain at 1 kHz.
        x_dc_blocked, dc_prev_x, dc_prev_y = dc_block_filter(
            x_raw, dc_prev_x, dc_prev_y, DC_BLOCK_ALPHA
        )

        shown = x_raw[:TIME_SAMPLES]
        line_i.set_ydata(np.real(shown))
        line_q.set_ydata(np.imag(shown))

        freqs_hz, raw_spectrum_dbfs = compute_spectrum_dbfs(x_raw, SAMPLE_RATE)
        _, dc_unfiltered_spectrum_dbfs = compute_spectrum_dbfs(x_dc_blocked, SAMPLE_RATE)

        in_view = np.abs(freqs_hz) <= SPECTRUM_SPAN_HZ
        rms = float(np.sqrt(np.mean(np.abs(x_raw) ** 2)))

        # EMA in linear POWER domain — avoids the log-domain bias that suppresses weak signals.
        # Jensen's inequality: E[log(x)] < log(E[x]), so dBFS averaging biases weak signals low.
        # Average |FFT|^2 (power), then convert back to dBFS only for display.
        raw_pow = np.power(10.0, raw_spectrum_dbfs * 0.1)          # dBFS → linear power
        dc_pow  = np.power(10.0, dc_unfiltered_spectrum_dbfs * 0.1)
        if smoothed_raw_spec_dbfs.size == 0:
            smoothed_raw_pow = raw_pow
            smoothed_dc_pow  = dc_pow
        else:
            smoothed_raw_pow = np.power(10.0, smoothed_raw_spec_dbfs * 0.1)
            smoothed_dc_pow  = np.power(10.0, smoothed_dc_blocked_spec_dbfs * 0.1)
            smoothed_raw_pow = FFT_AVG_ALPHA * raw_pow + (1.0 - FFT_AVG_ALPHA) * smoothed_raw_pow
            smoothed_dc_pow  = FFT_AVG_ALPHA * dc_pow  + (1.0 - FFT_AVG_ALPHA) * smoothed_dc_pow
        smoothed_raw_spec_dbfs          = 10.0 * np.log10(np.maximum(smoothed_raw_pow, 1e-20))
        smoothed_dc_blocked_spec_dbfs   = 10.0 * np.log10(np.maximum(smoothed_dc_pow,  1e-20))

        # Carrier search on EMA-smoothed spectrum every frame — full allowed range.
        # EMA (~20-frame memory) already rejects single-frame noise spikes.
        # Require 10 dB above the noise floor to accept a new lock position.
        # If no prominent peak found, hold last known position.
        search_mask = (
            (np.abs(freqs_hz) >= EXCITER_SEARCH_MIN_HZ)
            & (np.abs(freqs_hz) <= EXCITER_SEARCH_MAX_HZ)
        )
        search_view = in_view & search_mask
        exciter_freqs_hz = freqs_hz[search_view]
        exciter_spec_view = smoothed_dc_blocked_spec_dbfs[search_view]
        if len(exciter_spec_view):
            exciter_idx = int(np.argmax(exciter_spec_view))
            cand_peak_hz = float(exciter_freqs_hz[exciter_idx])
            cand_peak_dbfs = float(exciter_spec_view[exciter_idx])
            noise_ref_dbfs = float(np.percentile(smoothed_dc_blocked_spec_dbfs[in_view], 50))
            # Accept lock if peak stands at least 10 dB above noise, OR on first acquisition.
            if (cand_peak_dbfs - noise_ref_dbfs) >= 10.0 or prev_peak_hz == 0.0:
                peak_hz = cand_peak_hz
                peak_dbfs = cand_peak_dbfs
            else:
                peak_hz = prev_peak_hz
                prev_idx = int(np.argmin(np.abs(freqs_hz - peak_hz)))
                peak_dbfs = float(smoothed_dc_blocked_spec_dbfs[prev_idx])
        else:
            peak_hz = prev_peak_hz
            prev_idx = int(np.argmin(np.abs(freqs_hz - peak_hz)))
            peak_dbfs = float(smoothed_dc_blocked_spec_dbfs[prev_idx])

        prev_peak_hz = peak_hz
        exciter_marker.set_xdata([peak_hz / 1000.0, peak_hz / 1000.0])

        if ENABLE_FOCUSED_FILTER:
            x_processed = bandpass_filter_around_carrier(
                x_dc_blocked,
                peak_hz,
                SAMPLE_RATE,
                FOCUSED_PASSBAND_HZ,
            )
            _, processed_spectrum_dbfs = compute_spectrum_dbfs(x_processed, SAMPLE_RATE)
            smoothed_dc_blocked_spec_dbfs = (
                FFT_AVG_ALPHA * processed_spectrum_dbfs
                + (1.0 - FFT_AVG_ALPHA) * smoothed_dc_blocked_spec_dbfs
            )

        freqs_khz = freqs_hz[in_view] / 1000.0
        raw_spec_view = smoothed_raw_spec_dbfs[in_view]
        dc_blocked_spec_view = smoothed_dc_blocked_spec_dbfs[in_view]
        line_fft_raw.set_data(freqs_khz, raw_spec_view)
        line_fft_dc_blocked.set_data(freqs_khz, dc_blocked_spec_view)

        dc_idx = int(np.argmin(np.abs(freqs_hz)))
        dc_dbfs = float(smoothed_raw_spec_dbfs[dc_idx])
        dc_to_carrier_db = peak_dbfs - dc_dbfs

        if prev_peak_dbfs is None or abs(peak_dbfs - prev_peak_dbfs) > 5.0:
            ax_centered.set_ylim(peak_dbfs - 40.0, peak_dbfs + 10.0)
            prev_peak_dbfs = peak_dbfs

        # Centered spectrum: full-buffer FFT (15 Hz/bin) with EMA — sidebands at ±67 bins
        centered_freqs_hz = freqs_hz - peak_hz
        centered_view = np.abs(centered_freqs_hz) <= CENTERED_SPAN_HZ
        centered_freqs_khz = centered_freqs_hz[centered_view] / 1000.0
        centered_spec = smoothed_dc_blocked_spec_dbfs[centered_view]
        centered_spec_display = centered_spec
        if centered_spec.size >= CENTERED_FREQ_SMOOTH_BINS:
            kernel = np.ones(CENTERED_FREQ_SMOOTH_BINS, dtype=np.float64) / float(CENTERED_FREQ_SMOOTH_BINS)
            centered_spec_display = np.convolve(centered_spec, kernel, mode="same")
        line_centered.set_data(centered_freqs_khz, centered_spec_display)

        # Auto-scale Y: keep both carrier and sidebands visible without over-zooming.
        # Exclude a wider carrier core so skirts do not bias the estimated floor.
        carrier_mask = np.abs(centered_freqs_khz) > 0.5
        if carrier_mask.any():
            noise_floor_est_y = float(np.percentile(centered_spec_display[carrier_mask], 20))
            ax_centered.set_ylim(noise_floor_est_y - 6.0, peak_dbfs + 4.0)
            prev_peak_dbfs = peak_dbfs

        # Sideband SNR: measure peak power at ±1 kHz vs noise floor
        snr_db, sb_pos, sb_neg, noise_floor_sb = compute_sideband_snr(
            centered_freqs_khz, centered_spec_display,
            SIDEBAND_OFFSET_KHZ, SIDEBAND_WINDOW_HZ,
        )
        sideband_scatter.set_offsets(
            np.array([[-SIDEBAND_OFFSET_KHZ, sb_neg], [SIDEBAND_OFFSET_KHZ, sb_pos]])
        )
        # Colour: green = locked (≥20 dB SNR), red = too weak
        dot_color = "#44ff88" if snr_db >= SNR_LOCK_THRESHOLD_DB else "#ff4444"
        sideband_scatter.set_facecolor([dot_color, dot_color])
        # Move threshold line to noise_floor + target
        snr_threshold_line.set_ydata([noise_floor_sb + SNR_LOCK_THRESHOLD_DB])

        if centered_spec.size:
            remapped_centered = np.interp(
                centered_axis,
                centered_freqs_khz,
                centered_spec_display,
                left=-140.0,
                right=-140.0,
            )
            # Clamp floor to avoid color-scale collapse from transient deep nulls.
            remapped_centered = np.maximum(remapped_centered, -140.0)

            # Compress only the central carrier core in the waterfall so sidebands
            # stay visually discrete instead of being hidden by center saturation.
            frame_noise_mask = np.abs(centered_axis) > 0.5
            if frame_noise_mask.any():
                frame_noise = float(np.percentile(remapped_centered[frame_noise_mask], 20))
                carrier_core = np.abs(centered_axis) <= 0.20
                remapped_centered[carrier_core] = np.minimum(
                    remapped_centered[carrier_core],
                    frame_noise + 18.0,
                )

            waterfall_data[:-1] = waterfall_data[1:]
            if WATERFALL_ROWS > 1:
                remapped_centered = (
                    WATERFALL_ROW_BLEND * remapped_centered
                    + (1.0 - WATERFALL_ROW_BLEND) * waterfall_data[-2]
                )
            waterfall_data[-1] = remapped_centered
            waterfall_img.set_data(waterfall_data)
            # Estimate colour scale excluding the carrier core (±0.5 kHz).
            wf_axis_mask = np.abs(centered_axis) > 0.5
            valid_wf = waterfall_data[:, wf_axis_mask]
            valid_wf = valid_wf[valid_wf > -139.0]
            if valid_wf.size:
                noise_floor_est = float(np.percentile(valid_wf, 15))
                waterfall_img.set_clim(
                    vmin=noise_floor_est - 1.0,
                    vmax=noise_floor_est + WATERFALL_DYN_RANGE_DB,
                )

        status.set_text(
            f"RX={RX_URI} | LO={FREQ_HZ/1e9:.3f} GHz | RMS={rms:.4f} FS | Exciter={peak_hz:,.1f} Hz @ {peak_dbfs:.1f} dBFS | DC={dc_dbfs:.1f} dBFS | Carrier-DC={dc_to_carrier_db:.1f} dB"
        )
        carrier_status.set_text(
            f"Carrier={peak_hz:,.1f} Hz | Span=±{CENTERED_SPAN_HZ/1000.0:.0f} kHz | SB SNR={snr_db:+.1f} dB (+{SIDEBAND_OFFSET_KHZ:.0f}kHz={sb_pos:.1f} -{SIDEBAND_OFFSET_KHZ:.0f}kHz={sb_neg:.1f} Noise={noise_floor_sb:.1f} dBFS)"
        )

        carrier_fig.canvas.draw_idle()

        # --- Coherent AM demodulation ---
        if peak_hz != 0.0:
            ncc_val, envelope = coherent_ncc(x_raw, peak_hz, 1000.0, SAMPLE_RATE)
            ncc_history = np.roll(ncc_history, -1)
            ncc_history[-1] = ncc_val
            line_ncc.set_ydata(ncc_history)
            env_show = envelope[:env_plot_len]
            line_env.set_ydata(env_show)

            # Robust autoscale: use percentiles to ignore spikes and smooth the limits.
            lo_p = float(np.percentile(env_show, 2.0))
            hi_p = float(np.percentile(env_show, 98.0))
            mid = 0.5 * (lo_p + hi_p)
            span = max(hi_p - lo_p, ENV_Y_MIN_SPAN)
            target_low = mid - 0.6 * span
            target_high = mid + 0.6 * span
            env_ylim_low = (1.0 - ENV_Y_SMOOTH_ALPHA) * env_ylim_low + ENV_Y_SMOOTH_ALPHA * target_low
            env_ylim_high = (1.0 - ENV_Y_SMOOTH_ALPHA) * env_ylim_high + ENV_Y_SMOOTH_ALPHA * target_high
            ax_env.set_ylim(env_ylim_low, env_ylim_high)

            ncc_abs_ema = (1.0 - NCC_DISPLAY_ALPHA) * ncc_abs_ema + NCC_DISPLAY_ALPHA * abs(ncc_val)
            if not ncc_lock:
                if ncc_abs_ema >= NCC_ENTER_THRESHOLD:
                    ncc_enter_count += 1
                else:
                    ncc_enter_count = 0
                if ncc_enter_count >= NCC_ENTER_FRAMES:
                    ncc_lock = True
                    ncc_exit_count = 0
            else:
                if ncc_abs_ema <= NCC_EXIT_THRESHOLD:
                    ncc_exit_count += 1
                else:
                    ncc_exit_count = 0
                if ncc_exit_count >= NCC_EXIT_FRAMES:
                    ncc_lock = False
                    ncc_enter_count = 0

            # Multi-phase 50 ms slicer: scan several timing offsets instead of trusting
            # the arbitrary frame start as the packet bit boundary.
            phase_sample_buffer = np.concatenate((phase_sample_buffer, x_raw))
            phase_chips_added = 0
            for phase, state in phase_state.items():
                next_sample = int(state["next_sample"])
                phase_chips: list[int] = state["chips"]  # type: ignore[assignment]
                while next_sample + bit_samples <= len(phase_sample_buffer):
                    bit_chunk = phase_sample_buffer[next_sample : next_sample + bit_samples]
                    bit_ncc, _ = coherent_ncc(bit_chunk, peak_hz, 1000.0, SAMPLE_RATE)
                    phase_chips.append(1 if abs(bit_ncc) >= BIT_NCC_THRESHOLD else 0)
                    next_sample += bit_samples
                    state["chips_seen"] = int(state["chips_seen"]) + 1
                    phase_chips_added += 1
                state["next_sample"] = next_sample

                if len(phase_chips) > PHASE_HISTORY_BITS * REPETITION_CHIPS:
                    drop = len(phase_chips) - PHASE_HISTORY_BITS * REPETITION_CHIPS
                    del phase_chips[:drop]
                    state["base_chip_index"] = int(state["base_chip_index"]) + drop

                search_start_by_offset: list[int] = state["search_start_by_offset"]  # type: ignore[assignment]
                last_header_abs_by_offset: list[int] = state["last_header_abs_by_offset"]  # type: ignore[assignment]

                for decode_offset in range(REPETITION_CHIPS):
                    decoded_bits = majority_decode_triplets(phase_chips, decode_offset)
                    base_bit_index = (int(state["base_chip_index"]) + decode_offset) // REPETITION_CHIPS
                    while True:
                        search_start = search_start_by_offset[decode_offset]
                        header_idx, header_errors = find_header_match(
                            decoded_bits,
                            packet_header_bits,
                            search_start,
                            HEADER_MAX_BIT_ERRORS,
                        )
                        if header_idx < 0:
                            search_start_by_offset[decode_offset] = max(
                                0,
                                len(decoded_bits) - len(packet_header_bits) + 1,
                            )
                            break
                        header_abs = base_bit_index + header_idx
                        if header_abs <= last_header_abs_by_offset[decode_offset]:
                            search_start_by_offset[decode_offset] = header_idx + 1
                            continue
                        payload_start = header_idx + len(packet_header_bits)
                        payload_end = payload_start + payload_bits_len
                        if payload_end > len(decoded_bits):
                            search_start_by_offset[decode_offset] = header_idx
                            break
                        payload = bits_to_bytes(decoded_bits[payload_start:payload_end])
                        payload_hex = payload.hex().upper() if payload else ""
                        payload_ascii = safe_ascii(payload)
                        print(
                            f"[RX HEADER] phase={phase} chip_offset={decode_offset} bit={header_abs} header_errors={header_errors} payload_hex={payload_hex} payload_ascii={payload_ascii!r}",
                            flush=True,
                        )
                        if payload == PAYLOAD_BYTES:
                            decoded_packets += 1
                            packet_status_text = f"Packet: OPEN detected at phase {phase}, offset {decode_offset}"
                            packet_status_hold = PACKET_STATUS_HOLD_FRAMES
                            print(
                                f"[RX PACKET {decoded_packets}] phase={phase} chip_offset={decode_offset} bit={header_abs} header_errors={header_errors} PREAMBLE+SYNC+OPEN",
                                flush=True,
                            )
                        else:
                            packet_status_text = (
                                f"Packet: header phase {phase}, offset {decode_offset}, errors={header_errors}, payload={payload!r}"
                            )
                            packet_status_hold = PACKET_STATUS_HOLD_FRAMES
                        last_header_abs_by_offset[decode_offset] = header_abs
                        search_start_by_offset[decode_offset] = header_idx + 1

            if phase_chips_added > 0:
                total_bits += phase_chips_added
                while total_bits >= next_debug_bits:
                    best_phase = max(
                        phase_offsets,
                        key=lambda phase: int(phase_state[phase]["chips_seen"]),
                    )
                    best_chips: list[int] = phase_state[best_phase]["chips"]  # type: ignore[assignment]
                    best_decoded = majority_decode_triplets(best_chips, 0)
                    chip_tail = bits_to_text(best_chips[-TERMINAL_DEBUG_BIT_TAIL:])
                    bit_tail = bits_to_text(best_decoded[-(TERMINAL_DEBUG_BIT_TAIL // REPETITION_CHIPS):])
                    print(
                        f"[RX DEBUG] phase={best_phase} chips={int(phase_state[best_phase]['chips_seen'])} ncc_ema={ncc_abs_ema:.3f} lock={'1' if ncc_lock else '0'} chip_tail={chip_tail} bit_tail={bit_tail}",
                        flush=True,
                    )
                    next_debug_bits += TERMINAL_DEBUG_BITS_EVERY

                min_next_sample = min(int(state["next_sample"]) for state in phase_state.values())
                if min_next_sample > bit_samples:
                    trim = min_next_sample - bit_samples
                    phase_sample_buffer = phase_sample_buffer[trim:]
                    for state in phase_state.values():
                        state["next_sample"] = int(state["next_sample"]) - trim

            if packet_status_hold > 0:
                packet_status_hold -= 1
            else:
                packet_status_text = "Waiting for preamble+sync"

            ncc_status.set_text(
                f"NCC={ncc_val:+.3f} | EMA={ncc_abs_ema:.3f} | {'LOCKED' if ncc_lock else 'searching'} | Carrier={peak_hz:,.0f} Hz | {packet_status_text}"
            )
            ncc_fig.canvas.draw_idle()

        return (
            line_i,
            line_q,
            line_fft_raw,
            line_fft_dc_blocked,
            exciter_marker,
            line_centered,
            waterfall_img,
            status,
            carrier_status,
        )

    animation = FuncAnimation(fig, update, interval=100, blit=False, cache_frame_data=False)
    print("Starting Pluto GUI receiver...")
    print(f"  RX URI    : {RX_URI}")
    print(f"  Freq (Hz) : {int(FREQ_HZ)}")
    print(f"  SR (SPS)  : {int(SAMPLE_RATE)}")
    print(f"  Gain mode : {RX_GAIN_MODE}")
    if RX_GAIN_MODE == "manual":
        print(f"  RX gain   : {RX_GAIN_DB} dB")
    fig.tight_layout()
    carrier_fig.tight_layout()
    _ = animation
    plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())