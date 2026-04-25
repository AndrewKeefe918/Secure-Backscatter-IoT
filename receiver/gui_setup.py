"""Matplotlib figure and artist initialisation for the receiver GUI.

Each setup_* function builds one window and returns a dataclass holding
all the artist handles needed by ReceiverRuntime.update().
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import matplotlib.pyplot as plt

from . import config


# ---------------------------------------------------------------------------
# Dataclasses — typed containers for plot handles passed to ReceiverRuntime
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
    ax_iq: Any
    iq_scatter: Any           # recent raw IQ samples (subsampled)
    tuning_circle: Any        # Line2D — carrier-locus circle (Satori reference)
    triangle_line: Any        # Line2D — closed triangle: carrier + ±1 kHz sideband phasors
    triangle_vertices: Any    # PathCollection — 3 dots at triangle vertices
    iq_centroid: Any          # PathCollection — single centroid marker


@dataclass
class CarrierWindow:
    fig: Any
    ax_centered: Any
    line_centered: Any
    waterfall_img: Any
    waterfall_data: np.ndarray
    centered_axis: np.ndarray   # reused by ReceiverRuntime for interpolation / masking
    sideband_scatter: Any
    snr_threshold_line: Any
    status: Any


@dataclass
class NccWindow:
    fig: Any
    ax_ncc: Any
    ax_env: Any
    line_ncc: Any
    line_env: Any
    line_env_bits: Any
    env_threshold_line: Any
    ncc_status: Any
    ncc_history: np.ndarray
    env_plot_len: int


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def setup_baseband_window() -> BasebandWindow:
    """Create the baseband window: time domain + IQ plane (top), spectrum (bottom).

    The IQ panel shows a Satori-style tuning circle (carrier-locus) and the
    “tuning triangle” formed by the reference (carrier) phasor and the two
    ±1 kHz backscatter sideband phasors — the three points whose geometry
    Satori uses to recover residual CFO and pilot weighting.
    """
    fig = plt.figure(figsize=(13, 7))
    fig.canvas.manager.set_window_title("Pluto Baseband Receiver")
    fig.suptitle("PlutoSDR Raw Baseband View")

    gs = fig.add_gridspec(2, 3, width_ratios=[2, 2, 1.6], height_ratios=[1, 1])
    ax_time = fig.add_subplot(gs[0, 0:2])
    ax_iq = fig.add_subplot(gs[0, 2])
    ax_fft = fig.add_subplot(gs[1, :])

    time_axis = np.arange(config.TIME_SAMPLES)
    (line_i,) = ax_time.plot(time_axis, np.zeros(config.TIME_SAMPLES), label="I", lw=1.0)
    (line_q,) = ax_time.plot(time_axis, np.zeros(config.TIME_SAMPLES), label="Q", lw=1.0)
    ax_time.set_title("Time Domain (first samples)")
    ax_time.set_xlabel("Sample")
    ax_time.set_ylabel("Amplitude (normalized)")
    ax_time.set_ylim(-config.TIME_PLOT_Y_LIMIT, config.TIME_PLOT_Y_LIMIT)
    ax_time.grid(True, alpha=0.3)
    ax_time.legend(loc="upper right")

    fft_axis = np.linspace(
        -config.SPECTRUM_SPAN_HZ / 1000.0,
        config.SPECTRUM_SPAN_HZ / 1000.0,
        config.BASEBAND_FFT_BINS,
    )
    (line_fft_raw,) = ax_fft.plot(
        fft_axis,
        np.full_like(fft_axis, config.DBFS_FLOOR),
        lw=0.9,
        color="0.65",
        label="Raw spectrum",
    )
    (line_fft_dc_blocked,) = ax_fft.plot(
        fft_axis,
        np.full_like(fft_axis, config.DBFS_FLOOR),
        lw=1.2,
        color="C0",
        label="Processed spectrum",
    )
    exciter_marker = ax_fft.axvline(0.0, color="C3", lw=1.0, alpha=0.8, label="Exciter peak")
    ax_fft.axvline(0.0, color="0.3", lw=0.8, alpha=0.35, linestyle="--")
    ax_fft.set_title("Spectrum Near Baseband")
    ax_fft.set_xlabel("Frequency Offset (kHz)")
    ax_fft.set_ylabel("Magnitude (dBFS)")
    ax_fft.set_ylim(config.DBFS_FLOOR, config.BASEBAND_FFT_YMAX_DBFS)
    ax_fft.grid(True, alpha=0.3)
    ax_fft.legend(loc="lower left")

    # ---- IQ-plane panel: tuning circle + Satori triangle -----------------
    ax_iq.set_title("IQ Plane — Tuning Circle & Satori Triangle")
    ax_iq.set_xlabel("I")
    ax_iq.set_ylabel("Q")
    ax_iq.set_aspect("equal", adjustable="box")
    ax_iq.grid(True, alpha=0.3)
    ax_iq.axhline(0.0, color="0.5", lw=0.6, alpha=0.5)
    ax_iq.axvline(0.0, color="0.5", lw=0.6, alpha=0.5)
    ax_iq.set_xlim(-1.0, 1.0)
    ax_iq.set_ylim(-1.0, 1.0)

    # Subsampled raw IQ scatter (gray cloud).
    iq_scatter = ax_iq.scatter(
        np.zeros(1), np.zeros(1),
        s=4, c="#888888", alpha=0.35, label="IQ samples",
    )
    # Tuning circle — Satori reference: carrier-locus where the reference symbol lives.
    theta = np.linspace(0.0, 2.0 * np.pi, 256)
    (tuning_circle,) = ax_iq.plot(
        np.cos(theta), np.sin(theta),
        color="C0", lw=1.2, alpha=0.85, label="Tuning circle (carrier)",
    )
    # Triangle: 3 vertices = carrier phasor, +1 kHz SB phasor, -1 kHz SB phasor.
    (triangle_line,) = ax_iq.plot(
        [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],
        color="C3", lw=1.4, alpha=0.9, label="Tuning triangle",
    )
    triangle_vertices = ax_iq.scatter(
        [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
        s=55, c=["#ffaa00", "#44ff88", "#44aaff"],
        edgecolors="black", linewidths=0.6, zorder=5,
        label="ref / +SB / -SB",
    )
    iq_centroid = ax_iq.scatter(
        [0.0], [0.0], s=30, facecolors="none", edgecolors="black",
        linewidths=0.9, marker="o", zorder=6,
    )
    ax_iq.legend(loc="upper right", fontsize=7)

    status = fig.text(0.01, 0.01, "Connecting...", ha="left", va="bottom")

    return BasebandWindow(
        fig=fig,
        line_i=line_i,
        line_q=line_q,
        line_fft_raw=line_fft_raw,
        line_fft_dc_blocked=line_fft_dc_blocked,
        exciter_marker=exciter_marker,
        status=status,
        ax_iq=ax_iq,
        iq_scatter=iq_scatter,
        tuning_circle=tuning_circle,
        triangle_line=triangle_line,
        triangle_vertices=triangle_vertices,
        iq_centroid=iq_centroid,
    )


def setup_carrier_window() -> CarrierWindow:
    """Create the two-panel carrier-detail window (centered spectrum + waterfall)."""
    carrier_fig, (ax_centered, ax_waterfall) = plt.subplots(2, 1, figsize=(10, 7))
    carrier_fig.canvas.manager.set_window_title("Pluto Carrier Detail")
    carrier_fig.suptitle("Auto-Centered Carrier View")

    centered_axis = np.linspace(
        -config.CENTERED_SPAN_HZ / 1000.0, config.CENTERED_SPAN_HZ / 1000.0, config.WATERFALL_BINS
    )
    (line_centered,) = ax_centered.plot(
        centered_axis, np.full_like(centered_axis, config.DBFS_FLOOR), lw=1.2, color="C0",
    )
    ax_centered.axvline(0.0, color="C3", lw=1.0, alpha=0.8)
    ax_centered.axvline(
        config.SIDEBAND_OFFSET_KHZ, color="lime", lw=0.9, alpha=0.7,
        linestyle="--", label="±1 kHz SB",
    )
    ax_centered.axvline(-config.SIDEBAND_OFFSET_KHZ, color="lime", lw=0.9, alpha=0.7, linestyle="--")
    ax_centered.set_title("Carrier-Centered Spectrum")
    ax_centered.set_xlabel("Offset From Carrier (kHz)")
    ax_centered.set_ylabel("Magnitude (dBFS)")
    ax_centered.set_ylim(config.CARRIER_VIEW_YMIN_DBFS, config.CARRIER_VIEW_YMAX_DBFS)
    ax_centered.grid(True, alpha=0.3)

    sideband_scatter = ax_centered.scatter(
        [-config.SIDEBAND_OFFSET_KHZ, config.SIDEBAND_OFFSET_KHZ],
        [config.DBFS_FLOOR, config.DBFS_FLOOR],
        c=["#ff4444", "#ff4444"], s=80, zorder=5,
        marker="D", edgecolors="white", linewidths=0.8,
        label="SB peaks",
    )
    snr_threshold_line = ax_centered.axhline(
        config.DBFS_FLOOR, color="#ff4444", lw=1.0, linestyle=":", alpha=0.85,
        label=f"SNR={config.SNR_LOCK_THRESHOLD_DB:.0f}dB target",
    )
    ax_centered.legend(loc="upper left", fontsize=8)

    waterfall_data = np.full(
        (config.WATERFALL_ROWS, centered_axis.size), config.DBFS_FLOOR, dtype=np.float64
    )
    waterfall_img = ax_waterfall.imshow(
        waterfall_data,
        aspect="auto",
        origin="lower",
        extent=[centered_axis[0], centered_axis[-1], 0, config.WATERFALL_ROWS],
        cmap="plasma",
        vmin=config.DBFS_FLOOR,
        vmax=config.WATERFALL_VMAX_DBFS,
    )
    ax_waterfall.set_title("Carrier-Centered Waterfall")
    ax_waterfall.set_xlabel("Offset From Carrier (kHz)")
    ax_waterfall.set_ylabel("Frame")
    ax_waterfall.axvline(config.SIDEBAND_OFFSET_KHZ, color="lime", lw=0.8, alpha=0.6, linestyle="--")
    ax_waterfall.axvline(-config.SIDEBAND_OFFSET_KHZ, color="lime", lw=0.8, alpha=0.6, linestyle="--")

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


def setup_ncc_window() -> NccWindow:
    """Create the two-panel NCC demodulator window (NCC history + AM envelope)."""
    ncc_fig, (ax_ncc, ax_env) = plt.subplots(2, 1, figsize=(10, 5))
    ncc_fig.canvas.manager.set_window_title("Coherent 1 kHz Demodulator")
    ncc_fig.suptitle("Coherent AM Demodulation — 1 kHz Square-Wave NCC")

    ncc_history = np.zeros(config.NCC_HISTORY_FRAMES, dtype=np.float64)
    ncc_time_axis = np.arange(config.NCC_HISTORY_FRAMES)
    (line_ncc,) = ax_ncc.plot(ncc_time_axis, ncc_history, lw=1.2, color="C2")
    ax_ncc.axhline(0.0, color="0.5", lw=0.8, linestyle="--")
    ax_ncc.axhline(
        config.NCC_ENTER_THRESHOLD,
        color="lime",
        lw=0.8,
        linestyle=":",
        label="detect threshold",
    )
    ax_ncc.axhline(-config.NCC_ENTER_THRESHOLD, color="lime", lw=0.8, linestyle=":")
    ax_ncc.set_ylim(-1.0, 1.0)
    ax_ncc.set_ylabel("NCC")
    ax_ncc.set_xlabel("Frame")
    ax_ncc.set_title("Normalised Cross-Correlation vs 1 kHz Square Wave")
    ax_ncc.legend(loc="upper right", fontsize=8)
    ax_ncc.grid(True, alpha=0.3)

    env_plot_len = config.CHIP_VIEW_HISTORY
    chip_axis = np.arange(env_plot_len)
    (line_env,) = ax_env.plot(
        chip_axis, np.zeros(env_plot_len), lw=1.1, color="C0", label="|NCC| per chip",
    )
    (line_env_bits,) = ax_env.step(
        chip_axis, np.zeros(env_plot_len), where="mid", lw=0.9, color="C3", alpha=0.8,
        label="chip decision (scaled)",
    )
    env_threshold_line = ax_env.axhline(
        config.BIT_NCC_THRESHOLD, color="lime", lw=0.9, linestyle="--", label="chip threshold"
    )
    ax_env.set_ylabel("|NCC|")
    ax_env.set_xlabel("Recent Chip Index")
    ax_env.set_title(
        f"Chip Decisions ({config.BIT_DURATION_MS:.0f} ms/chip, {env_plot_len} chips history)"
    )
    ax_env.set_ylim(0.0, 1.0)
    ax_env.legend(loc="upper right", fontsize=8)
    ax_env.grid(True, alpha=0.3)

    ncc_status = ncc_fig.text(0.01, 0.01, "Waiting...", ha="left", va="bottom")
    ncc_fig.tight_layout(rect=(0, 0.04, 1, 0.95))

    return NccWindow(
        fig=ncc_fig,
        ax_ncc=ax_ncc,
        ax_env=ax_env,
        line_ncc=line_ncc,
        line_env=line_env,
        line_env_bits=line_env_bits,
        env_threshold_line=env_threshold_line,
        ncc_status=ncc_status,
        ncc_history=ncc_history,
        env_plot_len=env_plot_len,
    )
