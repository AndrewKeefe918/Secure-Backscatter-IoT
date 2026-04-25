"""Matplotlib figure and artist initialisation for the receiver GUI.

Each setup_* function builds one window and returns a dataclass holding
all the artist handles needed by ReceiverLoop.update().
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import matplotlib.pyplot as plt

from . import config


# ---------------------------------------------------------------------------
# Dataclasses — typed containers for plot handles passed to ReceiverLoop
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
    centered_axis: np.ndarray   # reused by ReceiverLoop for interpolation / masking
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
    ncc_status: Any
    ncc_history: np.ndarray
    env_plot_len: int


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def setup_baseband_window() -> BasebandWindow:
    """Create the two-panel baseband window (time domain + spectrum)."""
    fig, (ax_time, ax_fft) = plt.subplots(2, 1, figsize=(11, 7))
    fig.canvas.manager.set_window_title("Pluto Baseband Receiver")
    fig.suptitle("PlutoSDR Raw Baseband View")

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
    """Create the two-panel carrier-detail window (centered spectrum + waterfall)."""
    carrier_fig, (ax_centered, ax_waterfall) = plt.subplots(2, 1, figsize=(10, 7))
    carrier_fig.canvas.manager.set_window_title("Pluto Carrier Detail")
    carrier_fig.suptitle("Auto-Centered Carrier View")

    centered_axis = np.linspace(
        -config.CENTERED_SPAN_HZ / 1000.0, config.CENTERED_SPAN_HZ / 1000.0, config.WATERFALL_BINS
    )
    (line_centered,) = ax_centered.plot(
        centered_axis, np.full_like(centered_axis, -140.0), lw=1.2, color="C0",
    )
    ax_centered.axvline(0.0, color="C3", lw=1.0, alpha=0.8)
    ax_centered.axvline(
        config.SIDEBAND_OFFSET_KHZ, color="lime", lw=0.9, alpha=0.7,
        linestyle="--", label="±1 kHz SB",
    )
    ax_centered.axvline(-config.SIDEBAND_OFFSET_KHZ, color="lime", lw=0.9, alpha=0.7, linestyle="--")
    ax_centered.legend(loc="upper left", fontsize=8)
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
        label="SB peaks",
    )
    snr_threshold_line = ax_centered.axhline(
        -140.0, color="#ff4444", lw=1.0, linestyle=":", alpha=0.85,
        label=f"SNR={config.SNR_LOCK_THRESHOLD_DB:.0f}dB target",
    )
    ax_centered.legend(loc="upper left", fontsize=8)

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
    NCC_HISTORY = 200
    ncc_fig, (ax_ncc, ax_env) = plt.subplots(2, 1, figsize=(10, 5))
    ncc_fig.canvas.manager.set_window_title("Coherent 1 kHz Demodulator")
    ncc_fig.suptitle("Coherent AM Demodulation — 1 kHz Square-Wave NCC")

    ncc_history = np.zeros(NCC_HISTORY, dtype=np.float64)
    ncc_time_axis = np.arange(NCC_HISTORY)
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

    env_plot_len = min(config.SAMPLES_PER_CHIP * config.ENV_WINDOW_CHIPS, config.RX_BUFFER_SIZE)
    (line_env,) = ax_env.plot(
        np.arange(env_plot_len), np.zeros(env_plot_len), lw=0.8, color="C0",
    )
    ax_env.set_ylabel("AM Envelope (AC)")
    ax_env.set_xlabel("Sample")
    ax_env.set_title(
        f"Demodulated Envelope ({env_plot_len / config.SAMPLE_RATE * 1000.0:.0f} ms window)"
    )
    ax_env.set_ylim(-0.02, 0.02)
    ax_env.grid(True, alpha=0.3)

    ncc_status = ncc_fig.text(0.01, 0.01, "Waiting...", ha="left", va="bottom")
    ncc_fig.tight_layout(rect=(0, 0.04, 1, 0.95))

    return NccWindow(
        fig=ncc_fig,
        ax_ncc=ax_ncc,
        ax_env=ax_env,
        line_ncc=line_ncc,
        line_env=line_env,
        ncc_status=ncc_status,
        ncc_history=ncc_history,
        env_plot_len=env_plot_len,
    )
