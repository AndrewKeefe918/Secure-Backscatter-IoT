"""DSP utility functions for backscatter signal processing."""

import numpy as np
import scipy.signal as sp

from . import config


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
    """Return (snr_db, sb_pos_dbfs, sb_neg_dbfs, noise_floor_dbfs).

    Sideband bins: within ±window_hz of ±sideband_offset_khz.
    Noise estimate: median of bins between 0.3–4.0 kHz excluding sideband windows.
    """
    w = window_hz / 1000.0
    pos_mask = np.abs(centered_freqs_khz - sideband_offset_khz) <= w
    neg_mask = np.abs(centered_freqs_khz + sideband_offset_khz) <= w
    sb_mask = pos_mask | neg_mask
    # Use only positive-offset bins for noise reference — the negative side often
    # carries exciter phase noise that inflates the noise estimate and suppresses SNR.
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

    Steps:
    1. Mix carrier to DC: multiply by exp(-j*2π*carrier_hz*n/fs)
    2. Take magnitude envelope — this is the AM modulation signal
    3. Remove DC from envelope (mean subtraction)
    4. Build ideal square-wave reference at mod_hz
    5. Compute normalised cross-correlation → NCC in [-1, 1]

    Returns (ncc_value, envelope) so the caller can plot the envelope.
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

    Computes the AM envelope (after carrier downmix), removes DC, and returns
    the magnitude of the discrete Fourier coefficient at ``mod_hz`` normalised
    by the envelope RMS. Independent of the subcarrier starting phase, so
    short (50 ms) chips no longer suffer the cos(phase) attenuation that the
    real-valued square-wave NCC suffers from.
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
    # sqrt(2) compensates for the half-power split between +/- frequency bins.
    return float(np.sqrt(2.0) * abs(coeff) / env_rms)


def compute_spectrum_dbfs(x: np.ndarray, sample_rate: float) -> tuple[np.ndarray, np.ndarray]:
    """Blackman-Harris window FFT → dBFS spectrum.

    Blackman window: -92 dB sidelobes vs -31 dB for Hanning.
    Lower sidelobes prevent the strong carrier from leaking into adjacent
    bins and masking the weaker 1 kHz backscatter sidebands.
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
    """Carrier-centered filter using heterodyne + low-pass.

    1) Mix carrier_hz to DC.
    2) Low-pass to keep ±passband_hz.
    3) Mix back to original center frequency.
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


def ema_spectrum_power_domain(
    current_dbfs: np.ndarray,
    previous_dbfs: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Exponential moving average in linear power, returned in dB.

    Operating in power space (not dB) avoids Jensen's-inequality bias that
    otherwise pushes averages below the true mean.
    """
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
    """Locate the exciter carrier with sticky tracking.

    Returns (peak_hz, peak_dbfs). If the new candidate is not sufficiently
    above the median noise, the previous lock is kept to avoid hopping.
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
