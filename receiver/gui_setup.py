"""Matplotlib figure and artist initialisation for the receiver GUI â€” FSK version.

Difference from the OOK GUI:
  - The carrier-detail spectrum and waterfall both show TWO subcarrier
    markers (1 kHz for '1' bits, 1.7 kHz for '0' bits).
  - The chip-decision panel plots BOTH metrics m_f1 and m_f0 in addition
    to the per-chip decision, so visual inspection is easy.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import matplotlib.pyplot as plt

from . import config as config


# ---------------------------------------------------------------------------
# Dataclasses â€” typed containers for plot handles passed to ReceiverLoop
# ---------------------------------------------------------------------------

@dataclass
class BasebandWindow:
    fig: Any
    line_i: Any
    line_q: Any
    line_fft_raw: Any
    line_fft_dc_blocked: Any
    exciter_marker: Any
    status: Any


@dataclass
class CarrierWindow:
    fig: Any
    ax_centered: Any
    line_centered: Any
    waterfall_img: Any
    waterfall_data: np.ndarray
    centered_axis: np.ndarray
    sideband_scatter: Any
    snr_threshold_line: Any
    status: Any


@dataclass
class FskWindow:
    """Replaces the NCC window from the OOK build.

    Top panel: rolling history of m_f1, m_f0, and the decision metric.
    Bottom panel: per-chip decisions over time with the two metric traces.
    """
    fig: Any
    ax_metrics: Any
    ax_chips: Any
    line_m_f1: Any
    line_m_f0: Any
    line_decision: Any
    line_chip_decision: Any
    line_chip_m_f1: Any
    line_chip_m_f0: Any
    status: Any
    metric_history_f1: np.ndarray
    metric_history_f0: np.ndarray
    metric_history_decision: np.ndarray
    chip_history_len: int


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def setup_baseband_window() -> BasebandWindow:
    """Time-domain + wide-spectrum view (unchanged from OOK)."""
    fig, (ax_time, ax_fft) = plt.subplots(2, 1, figsize=(11, 7))
    fig.canvas.manager.set_window_title("Pluto Baseband Receiver â€” FSK")
    fig.suptitle("PlutoSDR Raw Baseband View (FSK Mode)")

    time_axis = np.arange(config.TIME_SAMPLES)
    (line_i,) = ax_time.plot(time_axis, np.zeros(config.TIME_SAMPLES), label="I", lw=1.0)
    (line_q,) = ax_time.plot(time_axis, np.zeros(config.TIME_SAMPLES), label="Q", lw=1.0)
    ax_time.set_title("Time Domain (first samples)")
    ax_time.set_xlabel("Sample")
    ax_time.set_ylabel("Amplitude (normalized)")
    ax_time.set_ylim(-1.1, 1.1)
    ax_time.grid(True, alpha=0.3)
    ax_time.legend(loc="upper right")

    fft_axis = np.linspace(
        -config.SPECTRUM_SPAN_HZ / 1000.0, config.SPECTRUM_SPAN_HZ / 1000.0, 512
    )
    (line_fft_raw,) = ax_fft.plot(
        fft_axis, np.full_like(fft_axis, -140.0), lw=0.9, color="0.65", label="Raw spectrum",
    )
    (line_fft_dc_blocked,) = ax_fft.plot(
        fft_axis, np.full_like(fft_axis, -140.0), lw=1.2, color="C0", label="Processed spectrum",
    )
    exciter_marker = ax_fft.axvline(0.0, color="C3", lw=1.0, alpha=0.8, label="Exciter peak")
    ax_fft.axvline(0.0, color="0.3", lw=0.8, alpha=0.35, linestyle="--")
    ax_fft.set_title("Spectrum Near Baseband")
    ax_fft.set_xlabel("Frequency Offset (kHz)")
    ax_fft.set_ylabel("Magnitude (dBFS)")
    ax_fft.set_ylim(-140.0, 5.0)
    ax_fft.grid(True, alpha=0.3)
    ax_fft.legend(loc="lower left")

    status = fig.text(0.01, 0.01, "Connecting...", ha="left", va="bottom")

    return BasebandWindow(
        fig=fig,
        line_i=line_i,
        line_q=line_q,
        line_fft_raw=line_fft_raw,
        line_fft_dc_blocked=line_fft_dc_blocked,
        exciter_marker=exciter_marker,
        status=status,
    )


def setup_carrier_window() -> CarrierWindow:
    """Carrier-centered spectrum + waterfall, with markers for BOTH FSK frequencies."""
    carrier_fig, (ax_centered, ax_waterfall) = plt.subplots(2, 1, figsize=(10, 7))
    carrier_fig.canvas.manager.set_window_title("Pluto Carrier Detail â€” FSK")
    carrier_fig.suptitle(
        f"Auto-Centered Carrier â€” FSK markers at "
        f"{config.FSK_F1_HZ/1000:.1f} kHz ('1') and {config.FSK_F0_HZ/1000:.1f} kHz ('0')"
    )

    centered_axis = np.linspace(
        -config.CENTERED_SPAN_HZ / 1000.0, config.CENTERED_SPAN_HZ / 1000.0, config.WATERFALL_BINS
    )
    (line_centered,) = ax_centered.plot(
        centered_axis, np.full_like(centered_axis, -140.0), lw=1.2, color="C0",
    )
    ax_centered.axvline(0.0, color="C3", lw=1.0, alpha=0.8)

    # Markers for the '1' frequency (1 kHz)
    ax_centered.axvline(
        config.SIDEBAND_OFFSET_KHZ, color="lime", lw=0.9, alpha=0.7,
        linestyle="--", label=f"Â±{config.FSK_F1_HZ/1000:.1f} kHz ('1')",
    )
    ax_centered.axvline(-config.SIDEBAND_OFFSET_KHZ, color="lime", lw=0.9, alpha=0.7, linestyle="--")
    # Markers for the '0' frequency (1.7 kHz)
    ax_centered.axvline(
        config.SIDEBAND_OFFSET_F0_KHZ, color="orange", lw=0.9, alpha=0.7,
        linestyle="--", label=f"Â±{config.FSK_F0_HZ/1000:.1f} kHz ('0')",
    )
    ax_centered.axvline(-config.SIDEBAND_OFFSET_F0_KHZ, color="orange", lw=0.9, alpha=0.7, linestyle="--")
    ax_centered.set_title("Carrier-Centered Spectrum")
    ax_centered.set_xlabel("Offset From Carrier (kHz)")
    ax_centered.set_ylabel("Magnitude (dBFS)")
    ax_centered.set_ylim(-120.0, -60.0)
    ax_centered.grid(True, alpha=0.3)

    sideband_scatter = ax_centered.scatter(
        [-config.SIDEBAND_OFFSET_KHZ, config.SIDEBAND_OFFSET_KHZ],
        [-140.0, -140.0],
        c=["#ff4444", "#ff4444"], s=80, zorder=5,
        marker="D", edgecolors="white", linewidths=0.8,
        label="'1' SB peaks",
    )
    snr_threshold_line = ax_centered.axhline(
        -140.0, color="#ff4444", lw=1.0, linestyle=":", alpha=0.85,
        label=f"SNR={config.SNR_LOCK_THRESHOLD_DB:.0f}dB target",
    )
    ax_centered.legend(loc="upper left", fontsize=7)

    waterfall_data = np.full(
        (config.WATERFALL_ROWS, centered_axis.size), -140.0, dtype=np.float64
    )
    waterfall_img = ax_waterfall.imshow(
        waterfall_data,
        aspect="auto",
        origin="lower",
        extent=[centered_axis[0], centered_axis[-1], 0, config.WATERFALL_ROWS],
        cmap="plasma",
        vmin=-140.0,
        vmax=-20.0,
    )
    ax_waterfall.set_title("Carrier-Centered Waterfall")
    ax_waterfall.set_xlabel("Offset From Carrier (kHz)")
    ax_waterfall.set_ylabel("Frame")
    # Both frequency markers in the waterfall too
    ax_waterfall.axvline(config.SIDEBAND_OFFSET_KHZ, color="lime", lw=0.8, alpha=0.6, linestyle="--")
    ax_waterfall.axvline(-config.SIDEBAND_OFFSET_KHZ, color="lime", lw=0.8, alpha=0.6, linestyle="--")
    ax_waterfall.axvline(config.SIDEBAND_OFFSET_F0_KHZ, color="orange", lw=0.8, alpha=0.6, linestyle="--")
    ax_waterfall.axvline(-config.SIDEBAND_OFFSET_F0_KHZ, color="orange", lw=0.8, alpha=0.6, linestyle="--")

    carrier_status = carrier_fig.text(0.01, 0.01, "Centering...", ha="left", va="bottom")

    return CarrierWindow(
        fig=carrier_fig,
        ax_centered=ax_centered,
        line_centered=line_centered,
        waterfall_img=waterfall_img,
        waterfall_data=waterfall_data,
        centered_axis=centered_axis,
        sideband_scatter=sideband_scatter,
        snr_threshold_line=snr_threshold_line,
        status=carrier_status,
    )


def setup_fsk_window() -> FskWindow:
    """FSK demodulator window â€” replaces the OOK NCC window.

    Top panel: rolling history of m_f1, m_f0, and decision metric per buffer.
    Bottom panel: per-chip decisions (binary trace) plus the two metrics
    that produced them, so you can SEE which frequency dominated each chip.
    """
    METRIC_HISTORY = 200
    fsk_fig, (ax_metrics, ax_chips) = plt.subplots(2, 1, figsize=(10, 5))
    fsk_fig.canvas.manager.set_window_title("FSK Demodulator")
    fsk_fig.suptitle(
        f"FSK Demodulation â€” m({config.FSK_F1_HZ/1000:.1f} kHz) vs m({config.FSK_F0_HZ/1000:.1f} kHz)"
    )

    metric_history_f1 = np.zeros(METRIC_HISTORY, dtype=np.float64)
    metric_history_f0 = np.zeros(METRIC_HISTORY, dtype=np.float64)
    metric_history_decision = np.zeros(METRIC_HISTORY, dtype=np.float64)
    metric_axis = np.arange(METRIC_HISTORY)

    (line_m_f1,) = ax_metrics.plot(
        metric_axis, metric_history_f1, lw=1.0, color="lime",
        label=f"m({config.FSK_F1_HZ/1000:.1f} kHz)  ['1']",
    )
    (line_m_f0,) = ax_metrics.plot(
        metric_axis, metric_history_f0, lw=1.0, color="orange",
        label=f"m({config.FSK_F0_HZ/1000:.1f} kHz)  ['0']",
    )
    (line_decision,) = ax_metrics.plot(
        metric_axis, metric_history_decision, lw=1.2, color="C0",
        label="decision (m_f1 - m_f0)", alpha=0.85,
    )
    ax_metrics.axhline(0.0, color="0.5", lw=0.8, linestyle="--")
    ax_metrics.axhline(
        config.FSK_DECISION_DEAD_ZONE, color="0.5", lw=0.6, linestyle=":", alpha=0.6,
    )
    ax_metrics.axhline(
        -config.FSK_DECISION_DEAD_ZONE, color="0.5", lw=0.6, linestyle=":", alpha=0.6,
    )
    ax_metrics.set_ylabel("Metric")
    ax_metrics.set_xlabel("Frame")
    ax_metrics.set_title("Per-Buffer FSK Metrics")
    ax_metrics.set_ylim(-0.5, 0.8)
    ax_metrics.legend(loc="upper right", fontsize=8)
    ax_metrics.grid(True, alpha=0.3)

    chip_history_len = config.CHIP_VIEW_HISTORY
    chip_axis = np.arange(chip_history_len)
    (line_chip_m_f1,) = ax_chips.plot(
        chip_axis, np.zeros(chip_history_len), lw=0.9, color="lime", alpha=0.75,
        label=f"m({config.FSK_F1_HZ/1000:.1f} kHz) per chip",
    )
    (line_chip_m_f0,) = ax_chips.plot(
        chip_axis, np.zeros(chip_history_len), lw=0.9, color="orange", alpha=0.75,
        label=f"m({config.FSK_F0_HZ/1000:.1f} kHz) per chip",
    )
    (line_chip_decision,) = ax_chips.step(
        chip_axis, np.zeros(chip_history_len), where="mid", lw=1.1, color="C3", alpha=0.9,
        label="chip decision (scaled)",
    )
    ax_chips.set_ylabel("Metric / Decision")
    ax_chips.set_xlabel("Recent Chip Index")
    ax_chips.set_title(
        f"Per-Chip FSK Metrics ({config.BIT_DURATION_MS:.0f} ms/chip, {chip_history_len} chips history)"
    )
    ax_chips.set_ylim(0.0, 1.0)
    ax_chips.legend(loc="upper right", fontsize=8)
    ax_chips.grid(True, alpha=0.3)

    status = fsk_fig.text(0.01, 0.01, "Waiting...", ha="left", va="bottom")
    fsk_fig.tight_layout(rect=(0, 0.04, 1, 0.95))

    return FskWindow(
        fig=fsk_fig,
        ax_metrics=ax_metrics,
        ax_chips=ax_chips,
        line_m_f1=line_m_f1,
        line_m_f0=line_m_f0,
        line_decision=line_decision,
        line_chip_decision=line_chip_decision,
        line_chip_m_f1=line_chip_m_f1,
        line_chip_m_f0=line_chip_m_f0,
        status=status,
        metric_history_f1=metric_history_f1,
        metric_history_f0=metric_history_f0,
        metric_history_decision=metric_history_decision,
        chip_history_len=chip_history_len,
    )

