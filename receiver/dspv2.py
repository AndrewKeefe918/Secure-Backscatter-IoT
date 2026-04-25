# Add to dsp.py

def ema_spectrum_power_domain(
    current_dbfs: np.ndarray,
    previous_dbfs: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Exponential moving average in linear power, returned in dB.

    Operating in power space (not dB) avoids Jensen's-inequality bias
    that would otherwise make the averaged trace sit below the true mean.
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
    """Locate the exciter carrier, with sticky tracking.

    Returns (peak_hz, peak_dbfs). If the new candidate isn't at least
    snr_keep_db above the median noise, we keep the previous peak to
    avoid bouncing into noise bins during fades.
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
    """Interpolate centered spectrum onto waterfall axis and clip the carrier.

    The carrier is typically 40 dB above the sidebands; leaving it at true
    amplitude in the waterfall saturates the colormap around DC and hides
    the sidebands. We cap the bins near DC at (frame noise + headroom).
    """
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
    """Moving-average smoothing. Returns x unchanged if too short or window <= 1."""
    if window_size <= 1 or x.size < window_size:
        return x
    kernel = np.ones(window_size, dtype=np.float64) / float(window_size)
    return np.convolve(x, kernel, mode="same")
