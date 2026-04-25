"""UI helpers for chip debug logging and per-frame GUI updates."""

from typing import Any

import numpy as np

from . import config
from .gui_setup import BasebandWindow
from .packet import majority_decode_triplets, bits_to_text


_IQ_THETA = np.linspace(0.0, 2.0 * np.pi, 256)
_IQ_SCATTER_MAX_PTS = 600


def _phasor_at(x: np.ndarray, freq_hz: float, n_idx: np.ndarray, inv_fs: float) -> complex:
    """Single-bin DFT of x at freq_hz, length-normalised."""
    n_len = max(len(x), 1)
    return complex(np.sum(x * np.exp(-1j * 2.0 * np.pi * freq_hz * n_idx * inv_fs)) / n_len)


def update_iq_panel(
    bb: BasebandWindow, x_raw: np.ndarray, x_dc: np.ndarray, peak_hz: float
) -> None:
    """Refresh IQ scatter, tuning circle, and Satori triangle on the baseband window."""
    # Use DC-blocked / focused samples for the IQ panel to suppress wideband haze.
    iq_src = x_dc if x_dc.size else x_raw
    step = max(1, len(iq_src) // _IQ_SCATTER_MAX_PTS)
    iq_pts = iq_src[::step][:_IQ_SCATTER_MAX_PTS]
    bb.iq_scatter.set_offsets(np.column_stack((np.real(iq_pts), np.imag(iq_pts))))
    # Robust centroid from clipped central cloud to avoid rare spikes dragging it.
    i_vals = np.real(iq_src)
    q_vals = np.imag(iq_src)
    i_lo, i_hi = np.percentile(
        i_vals,
        [config.IQ_CENTROID_CLIP_LOW_PERCENTILE, config.IQ_CENTROID_CLIP_HIGH_PERCENTILE],
    )
    q_lo, q_hi = np.percentile(
        q_vals,
        [config.IQ_CENTROID_CLIP_LOW_PERCENTILE, config.IQ_CENTROID_CLIP_HIGH_PERCENTILE],
    )
    i_c = float(np.mean(np.clip(i_vals, i_lo, i_hi)))
    q_c = float(np.mean(np.clip(q_vals, q_lo, q_hi)))
    bb.iq_centroid.set_offsets(np.array([[i_c, q_c]]))
    centroid = complex(i_c, q_c)

    n_idx = np.arange(len(x_dc), dtype=np.float64)
    inv_fs = 1.0 / float(config.SAMPLE_RATE)
    x_dc_c = x_dc.astype(np.complex128) - centroid
    sb = float(config.SUBCARRIER_HZ)
    c0 = _phasor_at(x_dc_c, peak_hz, n_idx, inv_fs)
    cp = _phasor_at(x_dc_c, peak_hz + sb, n_idx, inv_fs)
    cn = _phasor_at(x_dc_c, peak_hz - sb, n_idx, inv_fs)

    radius = float(abs(c0))
    bb.tuning_circle.set_data(
        i_c + radius * np.cos(_IQ_THETA),
        q_c + radius * np.sin(_IQ_THETA),
    )

    c0_plot = c0 + centroid
    cp_plot = cp + centroid
    cn_plot = cn + centroid
    verts_x = [c0_plot.real, cp_plot.real, cn_plot.real]
    verts_y = [c0_plot.imag, cp_plot.imag, cn_plot.imag]
    bb.triangle_line.set_data(verts_x + [verts_x[0]], verts_y + [verts_y[0]])
    bb.triangle_vertices.set_offsets(np.column_stack((verts_x, verts_y)))

    extent = max(radius, abs(cp), abs(cn), 1e-3) * config.IQ_PANEL_EXTENT_SCALE
    bb.ax_iq.set_xlim(i_c - extent, i_c + extent)
    bb.ax_iq.set_ylim(q_c - extent, q_c + extent)


def emit_chip_debug(
    *,
    best_phase: int,
    phase_state: dict[int, Any],
    ncc_abs_ema: float,
    ncc_lock: bool,
    repetition_chips: int,
    debug_bit_tail: int,
) -> None:
    """Emit terminal debug line for the selected phase."""
    st = phase_state[best_phase]

    # Try all chip offsets; pick the one with the most 0<->1 transitions for display.
    best_decoded = majority_decode_triplets(st.chips, 0)
    for off in range(1, repetition_chips):
        cand = majority_decode_triplets(st.chips, off)
        cand_transitions = sum(a != b for a, b in zip(cand, cand[1:]))
        best_transitions = sum(a != b for a, b in zip(best_decoded, best_decoded[1:]))
        if cand_transitions > best_transitions:
            best_decoded = cand

    chip_tail = bits_to_text(st.chips[-debug_bit_tail:])
    bit_tail = bits_to_text(best_decoded[-(debug_bit_tail // repetition_chips):])
    print(
        f"[RX DEBUG] phase={best_phase} chips={st.chips_seen} "
        f"ncc_ema={ncc_abs_ema:.3f} lock={int(ncc_lock)} "
        f"bit_thr={st.bit_ncc_threshold:.3f} noise_ema={st.bit_ncc_noise_ema:.3f} "
        f"chip_tail={chip_tail} bit_tail={bit_tail}",
        flush=True,
    )


def update_chip_view(
    *,
    nw: Any,
    ranked: list[tuple[int, float]],
    phase_state: dict[int, Any],
    env_chip_metric_history_by_phase: dict[int, Any],
    env_chip_decision_history_by_phase: dict[int, Any],
) -> tuple[int, str]:
    """Update NCC chip-view artists and return selected phase + summary text."""
    display_phase = ranked[0][0]
    phase_score_summary = " ".join(
        f"p{p}:{s:.2f}" for p, s in ranked[: config.PHASE_SCORE_DISPLAY_TOP]
    )
    st = phase_state[display_phase]

    nw.line_env.set_ydata(env_chip_metric_history_by_phase[display_phase])
    nw.line_env_bits.set_ydata(
        config.CHIP_VIEW_DECISION_SCALE * env_chip_decision_history_by_phase[display_phase]
    )
    nw.env_threshold_line.set_ydata([st.bit_ncc_threshold])

    return display_phase, phase_score_summary
