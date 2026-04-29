"""DSP utility functions for backscatter signal processing â€” FSK version.

Kept lean for the realtime path. The chip-level decision is
coherent_fsk_metrics_cached(); higher-level helpers feed peak tracking,
spectrum display, and CFO correction.
"""

import numpy as np
import scipy.signal as sp

from . import config as config


def normalize_iq(raw: object) -> np.ndarray:
    samples = np.asarray(raw, dtype=np.complex64)
    return samples / np.float32(config.ADC_FULL_SCALE)


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


def coherent_fsk_metrics_cached(
    x: np.ndarray,
    mix_c64: np.ndarray,
    ref_f1_c64: np.ndarray,
    ref_f0_c64: np.ndarray,
) -> tuple[float, float, float]:
    """Coherent FSK chip metric with caller-supplied complex64 references.

    Caller supplies the carrier->DC mix and the f1/f0 reference vectors;
    all inputs share the chip length. Avoids three np.exp() rebuilds per
    chip and stays in complex64 to halve memory bandwidth vs float64.
    """
    if x.size == 0:
        return 0.0, 0.0, 0.0
    # Carrier mix to DC, then real envelope.
    bb = x.astype(np.complex64, copy=False) * mix_c64
    envelope = np.abs(bb).astype(np.float32)
    envelope -= envelope.mean()
    env_rms = float(np.sqrt(np.mean(envelope * envelope)))
    if env_rms < 1e-9:
        return 0.0, 0.0, 0.0
    # complex64 dot products instead of mean()-of-elementwise-product.
    inv_n = 1.0 / float(envelope.size)
    coeff_f1 = complex(np.dot(envelope, ref_f1_c64)) * inv_n
    coeff_f0 = complex(np.dot(envelope, ref_f0_c64)) * inv_n
    m_f1 = float(np.sqrt(2.0) * abs(coeff_f1) / env_rms)
    m_f0 = float(np.sqrt(2.0) * abs(coeff_f0) / env_rms)
    return m_f1, m_f0, m_f1 - m_f0


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
    prev_peak_hz: float | None,
    search_min_hz: float,
    search_max_hz: float,
    snr_keep_db: float = 6.0,
    expected_hz: float | None = None,
    expected_tol_hz: float = 0.0,
    strict_expected_band: bool = False,
    max_step_hz: float = 0.0,
    switch_margin_db: float = 0.0,
) -> tuple[float, float]:
    """Locate the exciter carrier with sticky tracking.

    Only positive-frequency bins are searched: the exciter always transmits
    at a positive LO offset (TONE_HZ > 0), so the negative-frequency mirror
    image is noise / interference and must not win.
    """
    search_mask = (freqs_hz >= search_min_hz) & (freqs_hz <= search_max_hz)
    search_view = in_view_mask & search_mask
    if expected_hz is not None and expected_tol_hz > 0.0:
        expected_view = np.abs(freqs_hz - float(expected_hz)) <= float(expected_tol_hz)
        band_view = search_view & expected_view
        if np.any(band_view):
            search_view = band_view
        elif strict_expected_band:
            if prev_peak_hz is None:
                return 0.0, -140.0
            idx = int(np.argmin(np.abs(freqs_hz - prev_peak_hz)))
            return prev_peak_hz, float(spectrum_dbfs[idx])

    exciter_freqs = freqs_hz[search_view]
    exciter_spec = spectrum_dbfs[search_view]

    if exciter_spec.size == 0:
        if prev_peak_hz is None:
            return 0.0, -140.0
        idx = int(np.argmin(np.abs(freqs_hz - prev_peak_hz)))
        return prev_peak_hz, float(spectrum_dbfs[idx])

    best = int(np.argmax(exciter_spec))
    cand_hz = float(exciter_freqs[best])
    cand_dbfs = float(exciter_spec[best])
    noise_ref = float(np.percentile(spectrum_dbfs[in_view_mask], 50))

    # Anti-jump hysteresis: if the candidate is far from the tracked peak,
    # only allow switching when it is clearly stronger.
    if prev_peak_hz is not None and max_step_hz > 0.0:
        prev_idx = int(np.argmin(np.abs(freqs_hz - prev_peak_hz)))
        prev_dbfs = float(spectrum_dbfs[prev_idx])
        if abs(cand_hz - float(prev_peak_hz)) > float(max_step_hz):
            if cand_dbfs < (prev_dbfs + float(switch_margin_db)):
                return float(prev_peak_hz), prev_dbfs

    if (cand_dbfs - noise_ref) >= snr_keep_db or prev_peak_hz is None:
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

