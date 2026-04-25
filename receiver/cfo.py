"""Satori-inspired residual CFO estimation and correction helpers."""

import numpy as np

from . import config


class SatoriCfoCorrector:
    """Stateful two-step residual CFO estimator/corrector.

    Step 1: slow average rCFO tracking over each received chunk.
    Step 2: faster fluctuation tracking over even-index symbol windows.
    """

    def __init__(self, sample_rate: float) -> None:
        self.sample_rate = float(sample_rate)
        self.avg_hz = 0.0
        self.track_hz = 0.0
        self.phase_acc = 0.0
        self.low_conf_streak = 0

    @property
    def rcfo_hz(self) -> float:
        """Current combined rCFO estimate in Hz."""
        return self.avg_hz + self.track_hz

    def _estimate_rcfo_hz(self, x: np.ndarray, lag: int) -> float | None:
        if x.size <= lag or lag <= 0:
            return None
        corr = np.sum(np.conj(x[:-lag]) * x[lag:])
        if float(np.abs(corr)) <= config.MIN_RMS_GUARD:
            return None
        phase = float(np.angle(corr))
        return phase * self.sample_rate / (2.0 * np.pi * float(lag))

    def correct_chunk(self, demod: np.ndarray, confidence: float) -> np.ndarray:
        """Return rCFO-corrected demodulated samples."""
        if not config.SATORI_CFO_ENABLE or demod.size == 0:
            return demod

        lag = max(1, int(config.SATORI_CFO_LAG_SAMPLES))
        max_hz = float(config.SATORI_CFO_MAX_HZ)

        avg_est = self._estimate_rcfo_hz(demod, lag)
        if avg_est is not None and abs(avg_est) <= max_hz:
            self.avg_hz = (1.0 - config.SATORI_CFO_AVG_ALPHA) * self.avg_hz + (
                config.SATORI_CFO_AVG_ALPHA * avg_est
            )

        sym = max(lag + 1, int(config.SATORI_CFO_SYMBOL_SAMPLES))
        sym_count = demod.size // sym
        track_enabled = confidence >= config.SATORI_CFO_TRACK_CONF_THRESHOLD
        if sym_count > 0 and track_enabled:
            for idx in range(sym_count):
                if idx % 2 != 0:
                    continue
                seg = demod[idx * sym : (idx + 1) * sym]
                est = self._estimate_rcfo_hz(seg, lag)
                if est is None or abs(est) > max_hz:
                    continue
                self.track_hz = (
                    (1.0 - config.SATORI_CFO_TRACK_RHO) * self.track_hz
                    + config.SATORI_CFO_TRACK_RHO * est
                )

        if track_enabled:
            self.low_conf_streak = 0
        else:
            self.low_conf_streak += 1

        if self.low_conf_streak >= config.SATORI_CFO_DECAY_AFTER_UNLOCK_FRAMES:
            self.track_hz *= config.SATORI_CFO_DECAY_WHEN_UNLOCKED

        rcfo_hz = float(np.clip(self.rcfo_hz, -max_hz, max_hz))
        n = np.arange(demod.size, dtype=np.float64)
        phase = self.phase_acc + (2.0 * np.pi * rcfo_hz * n / self.sample_rate)
        rot = np.exp(-1j * phase)
        corrected = demod.astype(np.complex128) * rot
        self.phase_acc = float((phase[-1] + 2.0 * np.pi * rcfo_hz / self.sample_rate) % (2.0 * np.pi))
        return corrected.astype(np.complex64)
