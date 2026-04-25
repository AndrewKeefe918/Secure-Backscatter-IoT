"""DSP utility functions for backscatter signal processing."""

import numpy as np
import scipy.signal as sp

from . import config


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------

def normalize_iq(raw: object) -> np.ndarray:
    samples = np.asarray(raw, dtype=np.complex64)
    return samples / np.float32(config.ADC_FULL_SCALE)


def ema_scalar(prev: float, new: float, alpha: float) -> float:
    """Scalar exponential moving average helper shared by runtime pipelines."""
    return (1.0 - alpha) * prev + alpha * new


def _am_envelope(
    x: np.ndarray, carrier_hz: float, sample_rate: float
) -> tuple[np.ndarray, np.ndarray]:
    """Mix carrier to DC and return (mean-removed envelope, sample-index vector).

    Shared between :func:`coherent_ncc` and :func:`coherent_chip_metric` so
    the two stay perfectly in sync.
    """
    n = np.arange(x.size, dtype=np.float64)
    mix = np.exp(-1j * 2.0 * np.pi * float(carrier_hz) * n / float(sample_rate))
    envelope = np.abs(x.astype(np.complex128) * mix)
    envelope -= envelope.mean()
    return envelope, n


# ---------------------------------------------------------------------------
# Front-end filtering
# ---------------------------------------------------------------------------

def dc_block_filter(
    x: np.ndarray,
    x_prev: np.complex64,
    y_prev: np.complex64,
    alpha: float,
) -> tuple[np.ndarray, np.complex64, np.complex64]:
    """Vectorised first-order IIR DC-block: y[n] = x[n] - x[n-1] + alpha*y[n-1]."""
    b = np.array([1.0, -1.0])
    a = np.array([1.0, -float(alpha)])
    zi = np.array([-x_prev + float(alpha) * y_prev], dtype=np.complex128)
    y, _ = sp.lfilter(b, a, x.astype(np.complex128), zi=zi)
    y = y.astype(np.complex64)
    return y, np.complex64(x[-1]), np.complex64(y[-1])


def bandpass_filter_around_carrier(
    x: np.ndarray,
    carrier_hz: float,
    sample_rate: float,
    passband_hz: float,
) -> np.ndarray:
    """Carrier-centered filter via heterodyne + low-pass (mix down, LPF, mix back)."""
    if passband_hz <= 0.0 or len(x) == 0:
        return x
    nyquist = sample_rate / 2.0
    wn = min(passband_hz / nyquist, config.BUTTER_WN_MAX)
    if wn <= 0.0:
        return x
    n = np.arange(len(x), dtype=np.float64)
    w = 2.0 * np.pi * float(carrier_hz) / float(sample_rate)
    downmix = np.exp(-1j * w * n)
    upmix = np.exp(1j * w * n)
    x_shifted = x.astype(np.complex128) * downmix
    sos = sp.butter(config.BUTTER_ORDER, wn, btype="lowpass", output="sos")
    y_shifted = sp.sosfilt(sos, x_shifted)
    return (y_shifted * upmix).astype(np.complex64)


# ---------------------------------------------------------------------------
# Spectrum / SNR
# ---------------------------------------------------------------------------

def compute_spectrum_dbfs(x: np.ndarray, sample_rate: float) -> tuple[np.ndarray, np.ndarray]:
    """Blackman-windowed FFT → dBFS spectrum.

    Blackman's -92 dB sidelobes (vs -31 dB for Hanning) keep the strong
    carrier from leaking into adjacent bins and masking the weaker 1 kHz
    backscatter sidebands.
    """
    window = np.blackman(len(x)).astype(np.float32)
    spectrum = np.fft.fftshift(np.fft.fft(x * window))
    coherent_gain = float(np.mean(window))
    mag = np.abs(spectrum) / max(len(x) * coherent_gain, config.MIN_FFT_NORM)
    spectrum_dbfs = 20.0 * np.log10(np.maximum(mag, config.MIN_FFT_NORM))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(x), d=1.0 / sample_rate))
    return freqs, spectrum_dbfs


def ema_spectrum_power_domain(
    current_dbfs: np.ndarray,
    previous_dbfs: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """EMA in linear power, returned in dB.

    Operating in power space (not dB) avoids Jensen's-inequality bias that
    otherwise pushes averages below the true mean.
    """
    current_pow = np.power(10.0, current_dbfs * 0.1)
    if previous_dbfs.size == 0:
        smoothed_pow = current_pow
    else:
        previous_pow = np.power(10.0, previous_dbfs * 0.1)
        smoothed_pow = alpha * current_pow + (1.0 - alpha) * previous_pow
    return 10.0 * np.log10(np.maximum(smoothed_pow, config.MIN_SMOOTHED_POWER))


def compute_sideband_snr(
    centered_freqs_khz: np.ndarray,
    centered_spec_dbfs: np.ndarray,
    sideband_offset_khz: float,
    window_hz: float,
) -> tuple[float, float, float, float]:
    """Return (snr_db, sb_pos_dbfs, sb_neg_dbfs, noise_floor_dbfs).

    Noise reference uses positive-offset bins only — the negative side often
    carries exciter phase noise that inflates the noise estimate and
    suppresses SNR.
    """
    w = window_hz / 1000.0
    pos_mask = np.abs(centered_freqs_khz - sideband_offset_khz) <= w
    neg_mask = np.abs(centered_freqs_khz + sideband_offset_khz) <= w
    sb_mask = pos_mask | neg_mask
    span_upper_khz = config.CENTERED_SPAN_HZ / 1000.0 - config.NOISE_EXCLUDE_CARRIER_KHZ
    noise_mask = (
        (centered_freqs_khz > config.SIDEBAND_NOISE_LOW_KHZ)
        & (centered_freqs_khz < span_upper_khz)
        & ~sb_mask
    )
    if not sb_mask.any() or not noise_mask.any():
        return 0.0, config.DBFS_FLOOR, config.DBFS_FLOOR, config.DBFS_FLOOR
    sb_pos = float(np.max(centered_spec_dbfs[pos_mask])) if pos_mask.any() else config.DBFS_FLOOR
    sb_neg = float(np.max(centered_spec_dbfs[neg_mask])) if neg_mask.any() else config.DBFS_FLOOR
    noise_floor = float(
        np.percentile(centered_spec_dbfs[noise_mask], config.SIDEBAND_NOISE_PERCENTILE)
    )
    snr = max(sb_pos, sb_neg) - noise_floor
    return snr, sb_pos, sb_neg, noise_floor


# ---------------------------------------------------------------------------
# Coherent NCC / chip metric
# ---------------------------------------------------------------------------

def coherent_ncc(
    x: np.ndarray, carrier_hz: float, mod_hz: float, sample_rate: float
) -> float:
    """Coherent AM demod + NCC against a square-wave reference at ``mod_hz``.

    Returns NCC in [-1, 1].
    """
    if x.size == 0:
        return 0.0
    envelope, n = _am_envelope(x, carrier_hz, sample_rate)
    ref = np.sign(np.sin(2.0 * np.pi * mod_hz * n / sample_rate))
    env_rms = float(np.sqrt(np.mean(envelope ** 2)))
    ref_rms = float(np.sqrt(np.mean(ref ** 2)))
    if env_rms < config.MIN_RMS_GUARD or ref_rms < config.MIN_RMS_GUARD:
        return 0.0
    return float(np.mean(envelope * ref) / (env_rms * ref_rms))


def coherent_chip_metric(
    x: np.ndarray, carrier_hz: float, mod_hz: float, sample_rate: float
) -> float:
    """Phase-invariant subcarrier-presence metric in [0, ~1].

    Magnitude of the discrete Fourier coefficient at ``mod_hz`` over the AM
    envelope, normalised by envelope RMS. Independent of the chip's starting
    phase, so 50 ms chips don't suffer the cos(phase) attenuation that the
    real-valued square-wave NCC has on short observations.
    """
    if x.size == 0:
        return 0.0
    envelope, n = _am_envelope(x, carrier_hz, sample_rate)
    env_rms = float(np.sqrt(np.mean(envelope ** 2)))
    if env_rms < config.MIN_RMS_GUARD:
        return 0.0
    ref = np.exp(-1j * 2.0 * np.pi * float(mod_hz) * n / float(sample_rate))
    coeff = complex(np.mean(envelope * ref))
    # sqrt(2) compensates the half-power split between +/- frequency bins.
    return float(np.sqrt(2.0) * abs(coeff) / env_rms)


# ---------------------------------------------------------------------------
# Carrier tracking and waterfall remap
# ---------------------------------------------------------------------------

def find_exciter_peak(
    freqs_hz: np.ndarray,
    spectrum_dbfs: np.ndarray,
    in_view_mask: np.ndarray,
    prev_peak_hz: float,
    search_min_hz: float,
    search_max_hz: float,
) -> tuple[float, float]:
    """Locate the exciter carrier with sticky tracking.

    If the new candidate is not at least ``EXCITER_SNR_KEEP_DB`` above the
    median noise, the previous lock is kept to avoid hopping.
    """
    search_mask = (np.abs(freqs_hz) >= search_min_hz) & (np.abs(freqs_hz) <= search_max_hz)
    search_view = in_view_mask & search_mask
    exciter_freqs = freqs_hz[search_view]
    exciter_spec = spectrum_dbfs[search_view]

    if exciter_spec.size == 0:
        idx = int(np.argmin(np.abs(freqs_hz - prev_peak_hz)))
        return prev_peak_hz, float(spectrum_dbfs[idx])

    best = int(np.argmax(exciter_spec))
    cand_hz = float(exciter_freqs[best])
    cand_dbfs = float(exciter_spec[best])
    noise_ref = float(
        np.percentile(spectrum_dbfs[in_view_mask], config.SIDEBAND_NOISE_PERCENTILE)
    )

    if (cand_dbfs - noise_ref) >= config.EXCITER_SNR_KEEP_DB or prev_peak_hz == 0.0:
        return cand_hz, cand_dbfs

    idx = int(np.argmin(np.abs(freqs_hz - prev_peak_hz)))
    return prev_peak_hz, float(spectrum_dbfs[idx])


def remap_and_compress_centered(
    target_axis_khz: np.ndarray,
    source_axis_khz: np.ndarray,
    source_spec_dbfs: np.ndarray,
) -> np.ndarray:
    """Interpolate centered spectrum onto waterfall axis and clip the carrier core."""
    remapped = np.interp(
        target_axis_khz,
        source_axis_khz,
        source_spec_dbfs,
        left=config.DBFS_FLOOR,
        right=config.DBFS_FLOOR,
    )
    remapped = np.maximum(remapped, config.DBFS_FLOOR)

    noise_mask = np.abs(target_axis_khz) > config.NOISE_EXCLUDE_CARRIER_KHZ
    if noise_mask.any():
        frame_noise = float(
            np.percentile(remapped[noise_mask], config.SPECTRUM_NOISE_PERCENTILE)
        )
        core_mask = np.abs(target_axis_khz) <= config.WATERFALL_CARRIER_CORE_HALF_KHZ
        remapped[core_mask] = np.minimum(
            remapped[core_mask],
            frame_noise + config.WATERFALL_CARRIER_CORE_HEADROOM_DB,
        )
    return remapped


def smooth_1d(x: np.ndarray, window_size: int) -> np.ndarray:
    """Moving-average smoothing. Returns x unchanged for short vectors."""
    if window_size <= 1 or x.size < window_size:
        return x
    kernel = np.ones(window_size, dtype=np.float64) / float(window_size)
    return np.convolve(x, kernel, mode="same")
