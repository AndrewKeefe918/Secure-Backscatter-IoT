"""DSP utility functions for backscatter signal processing â€” FSK version.

Adds coherent_fsk_metrics() to the OOK toolkit; everything else is
preserved verbatim from the OOK code so the spectrum, waterfall, and
carrier-tracking machinery continues to work.
"""

import numpy as np
import scipy.signal as sp

from . import config as config


def normalize_iq(raw: object) -> np.ndarray:
    samples = np.asarray(raw, dtype=np.complex64)
    return samples / np.float32(config.ADC_FULL_SCALE)


def remove_dc(x: np.ndarray) -> np.ndarray:
    return x - np.mean(x)


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


def compute_sideband_snr(
    centered_freqs_khz: np.ndarray,
    centered_spec_dbfs: np.ndarray,
    sideband_offset_khz: float,
    window_hz: float,
) -> tuple[float, float, float, float]:
    """Return (snr_db, sb_pos_dbfs, sb_neg_dbfs, noise_floor_dbfs)."""
    w = window_hz / 1000.0
    pos_mask = np.abs(centered_freqs_khz - sideband_offset_khz) <= w
    neg_mask = np.abs(centered_freqs_khz + sideband_offset_khz) <= w
    sb_mask = pos_mask | neg_mask
    noise_mask = (
        (centered_freqs_khz > 0.3)
        & (centered_freqs_khz < (config.CENTERED_SPAN_HZ / 1000.0 - 0.5))
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

    Used for the carrier-tracking display in the GUI; the FSK chip
    decisions use coherent_fsk_metrics() below.
    """
    n = np.arange(len(x), dtype=np.float64)
    mix = np.exp(-1j * 2.0 * np.pi * carrier_hz * n / sample_rate)
    baseband = x.astype(np.complex128) * mix
    envelope = np.abs(baseband).astype(np.float64)
    envelope -= envelope.mean()
    ref = np.sign(np.sin(2.0 * np.pi * mod_hz * n / sample_rate))
    env_rms = float(np.sqrt(np.mean(envelope ** 2)))
    ref_rms = float(np.sqrt(np.mean(ref ** 2)))
    if env_rms < 1e-12 or ref_rms < 1e-12:
        return 0.0, envelope
    ncc = float(np.mean(envelope * ref) / (env_rms * ref_rms))
    return ncc, envelope


def coherent_chip_metric(
    x: np.ndarray,
    carrier_hz: float,
    mod_hz: float,
    sample_rate: float,
) -> float:
    """Phase-invariant subcarrier presence metric in [0, ~1].

    Magnitude of the discrete Fourier coefficient at mod_hz, normalised
    by the envelope RMS. Used as a building block by coherent_fsk_metrics.
    """
    if x.size == 0:
        return 0.0
    n = np.arange(x.size, dtype=np.float64)
    mix = np.exp(-1j * 2.0 * np.pi * float(carrier_hz) * n / float(sample_rate))
    envelope = np.abs(x.astype(np.complex128) * mix)
    envelope -= envelope.mean()
    env_rms = float(np.sqrt(np.mean(envelope ** 2)))
    if env_rms < 1e-12:
        return 0.0
    ref = np.exp(-1j * 2.0 * np.pi * float(mod_hz) * n / float(sample_rate))
    coeff = complex(np.mean(envelope * ref))
    return float(np.sqrt(2.0) * abs(coeff) / env_rms)


def coherent_fsk_metrics(
    x: np.ndarray,
    carrier_hz: float,
    f1_hz: float,
    f0_hz: float,
    sample_rate: float,
) -> tuple[float, float, float]:
    """Coherent dual-frequency FSK detection.

    Mixes the carrier to DC, takes the AM envelope, and computes the
    magnitude of the discrete Fourier coefficient at each of the two
    subcarrier frequencies. Returns (m_f1, m_f0, decision) where
    decision = m_f1 - m_f0; positive means '1', negative means '0'.

    The decision is self-normalising â€” comparing m_f1 to m_f0 eliminates
    the need for an absolute threshold. This is the main reason FSK is
    more robust than OOK on a marginal-SNR channel.
    """
    if x.size == 0:
        return 0.0, 0.0, 0.0
    n = np.arange(x.size, dtype=np.float64)
    mix = np.exp(-1j * 2.0 * np.pi * float(carrier_hz) * n / float(sample_rate))
    envelope = np.abs(x.astype(np.complex128) * mix)
    envelope -= envelope.mean()
    env_rms = float(np.sqrt(np.mean(envelope ** 2)))
    if env_rms < 1e-12:
        return 0.0, 0.0, 0.0

    ref_f1 = np.exp(-1j * 2.0 * np.pi * float(f1_hz) * n / float(sample_rate))
    ref_f0 = np.exp(-1j * 2.0 * np.pi * float(f0_hz) * n / float(sample_rate))

    # sqrt(2) compensates for the half-power split between +/- frequency bins.
    m_f1 = float(np.sqrt(2.0) * abs(complex(np.mean(envelope * ref_f1))) / env_rms)
    m_f0 = float(np.sqrt(2.0) * abs(complex(np.mean(envelope * ref_f0))) / env_rms)

    decision = m_f1 - m_f0
    return m_f1, m_f0, decision


def estimate_residual_cfo_hz(x: np.ndarray, sample_rate: float) -> float:
    """Estimate residual CFO from average adjacent-sample phase advance.

    The input is expected to be near-baseband complex IQ. The estimate is
    robust to amplitude scaling and provides a coarse per-buffer residual
    frequency offset in Hz.
    """
    if x.size < 2:
        return 0.0
    z = x[1:].astype(np.complex128) * np.conjugate(x[:-1].astype(np.complex128))
    c = complex(np.mean(z))
    if abs(c) < 1e-15:
        return 0.0
    return float(np.angle(c) * float(sample_rate) / (2.0 * np.pi))


def derotate_frequency(
    x: np.ndarray,
    freq_hz: float,
    sample_rate: float,
    start_phase_rad: float,
) -> tuple[np.ndarray, float]:
    """Apply phase-continuous derotation for a constant frequency offset.

    Returns the corrected signal and the ending phase to carry into the
    next buffer for continuity.
    """
    if x.size == 0 or abs(freq_hz) < 1e-12:
        return x, float(start_phase_rad)
    n = np.arange(x.size, dtype=np.float64)
    w = 2.0 * np.pi * float(freq_hz) / float(sample_rate)
    phase = float(start_phase_rad) + w * n
    rot = np.exp(-1j * phase)
    y = x.astype(np.complex128) * rot
    end_phase = float((float(start_phase_rad) + w * float(x.size)) % (2.0 * np.pi))
    return y.astype(np.complex64), end_phase


def compute_spectrum_dbfs(x: np.ndarray, sample_rate: float) -> tuple[np.ndarray, np.ndarray]:
    """Blackman window FFT -> dBFS spectrum.

    Blackman: -92 dB sidelobes vs -31 dB for Hanning. Lower sidelobes
    keep the strong carrier from leaking into adjacent bins and masking
    the weaker subcarrier sidebands.
    """
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
    """Carrier-centered filter using heterodyne + low-pass + heterodyne."""
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


def ema_spectrum_power_domain(
    current_dbfs: np.ndarray,
    previous_dbfs: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """EMA in linear power, returned in dB. Avoids Jensen's-inequality bias."""
    current_pow = np.power(10.0, current_dbfs * 0.1)
    if previous_dbfs.size == 0:
        smoothed_pow = current_pow
    else:
        previous_pow = np.power(10.0, previous_dbfs * 0.1)
        smoothed_pow = alpha * current_pow + (1.0 - alpha) * previous_pow
    return 10.0 * np.log10(np.maximum(smoothed_pow, 1e-20))


def find_exciter_peak(
    freqs_hz: np.ndarray,
    spectrum_dbfs: np.ndarray,
    in_view_mask: np.ndarray,
    prev_peak_hz: float,
    search_min_hz: float,
    search_max_hz: float,
    snr_keep_db: float = 10.0,
) -> tuple[float, float]:
    """Locate the exciter carrier with sticky tracking."""
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
    noise_ref = float(np.percentile(spectrum_dbfs[in_view_mask], 50))

    if (cand_dbfs - noise_ref) >= snr_keep_db or prev_peak_hz == 0.0:
        return cand_hz, cand_dbfs

    idx = int(np.argmin(np.abs(freqs_hz - prev_peak_hz)))
    return prev_peak_hz, float(spectrum_dbfs[idx])


def remap_and_compress_centered(
    target_axis_khz: np.ndarray,
    source_axis_khz: np.ndarray,
    source_spec_dbfs: np.ndarray,
    carrier_core_half_width_khz: float = 0.20,
    carrier_core_headroom_db: float = 18.0,
) -> np.ndarray:
    """Interpolate centered spectrum onto waterfall axis and clip carrier core."""
    remapped = np.interp(
        target_axis_khz, source_axis_khz, source_spec_dbfs, left=-140.0, right=-140.0
    )
    remapped = np.maximum(remapped, -140.0)

    noise_mask = np.abs(target_axis_khz) > 0.5
    if noise_mask.any():
        frame_noise = float(np.percentile(remapped[noise_mask], 20))
        core_mask = np.abs(target_axis_khz) <= carrier_core_half_width_khz
        remapped[core_mask] = np.minimum(remapped[core_mask], frame_noise + carrier_core_headroom_db)
    return remapped


def smooth_1d(x: np.ndarray, window_size: int) -> np.ndarray:
    """Moving-average smoothing. Returns x unchanged for short vectors."""
    if window_size <= 1 or x.size < window_size:
        return x
    kernel = np.ones(window_size, dtype=np.float64) / float(window_size)
    return np.convolve(x, kernel, mode="same")

