"""Per-frame receiver flow functions for frontend processing and demod/decode."""

from typing import Any

import numpy as np

from . import config
from .dsp import (
    normalize_iq,
    dc_block_filter,
    compute_spectrum_dbfs,
    bandpass_filter_around_carrier,
    coherent_ncc,
    compute_sideband_snr,
    ema_spectrum_power_domain,
    find_exciter_peak,
    remap_and_compress_centered,
    smooth_1d,
)
from .chips import slice_chips_for_phase, update_lock_hysteresis, rank_phases
from .lock import apply_lock_policy
from .packet import PacketCandidate, accept_best_packet_candidate, try_decode_packets
from .ui import update_iq_panel, emit_chip_debug, update_chip_view


def process_frontend_frame(runtime: Any) -> tuple[np.ndarray, float]:
    """Run front-end processing for one frame and return (focused_iq, peak_hz)."""
    bb, cw = runtime.bb, runtime.cw

    x_raw = normalize_iq(runtime.sdr.rx())
    x_dc, runtime.dc_prev_x, runtime.dc_prev_y = dc_block_filter(
        x_raw, runtime.dc_prev_x, runtime.dc_prev_y, config.DC_BLOCK_ALPHA
    )

    shown = x_raw[: config.TIME_SAMPLES]
    bb.line_i.set_ydata(np.real(shown))
    bb.line_q.set_ydata(np.imag(shown))

    freqs_hz, raw_dbfs = compute_spectrum_dbfs(x_raw, config.SAMPLE_RATE)
    _, dc_dbfs_full = compute_spectrum_dbfs(x_dc, config.SAMPLE_RATE)
    runtime.smoothed_raw = ema_spectrum_power_domain(raw_dbfs, runtime.smoothed_raw, config.FFT_AVG_ALPHA)
    runtime.smoothed_dc = ema_spectrum_power_domain(dc_dbfs_full, runtime.smoothed_dc, config.FFT_AVG_ALPHA)

    in_view = np.abs(freqs_hz) <= config.SPECTRUM_SPAN_HZ
    rms = float(np.sqrt(np.mean(np.abs(x_raw) ** 2)))

    peak_hz, peak_dbfs = find_exciter_peak(
        freqs_hz,
        runtime.smoothed_dc,
        in_view,
        runtime.prev_peak_hz,
        config.EXCITER_SEARCH_MIN_HZ,
        config.EXCITER_SEARCH_MAX_HZ,
    )
    runtime.prev_peak_hz = peak_hz

    bb.exciter_marker.set_xdata([peak_hz / 1000.0, peak_hz / 1000.0])
    x_focus = x_dc
    if config.ENABLE_FOCUSED_FILTER and peak_hz != 0.0:
        x_focus = bandpass_filter_around_carrier(
            x_dc, peak_hz, config.SAMPLE_RATE, config.FOCUSED_PASSBAND_HZ
        )
    update_iq_panel(bb, x_raw, x_focus, peak_hz)

    freqs_khz = freqs_hz[in_view] / 1000.0
    bb.line_fft_raw.set_data(freqs_khz, runtime.smoothed_raw[in_view])
    bb.line_fft_dc_blocked.set_data(freqs_khz, runtime.smoothed_dc[in_view])

    dc_idx = int(np.argmin(np.abs(freqs_hz)))
    dc_level = float(runtime.smoothed_raw[dc_idx])

    centered_freqs_hz = freqs_hz - peak_hz
    cm = np.abs(centered_freqs_hz) <= config.CENTERED_SPAN_HZ
    centered_freqs_khz = centered_freqs_hz[cm] / 1000.0
    centered_spec = smooth_1d(runtime.smoothed_dc[cm], config.CENTERED_FREQ_SMOOTH_BINS)
    cw.line_centered.set_data(centered_freqs_khz, centered_spec)

    off_carrier = np.abs(centered_freqs_khz) > config.NOISE_EXCLUDE_CARRIER_KHZ
    if off_carrier.any():
        noise_y = float(np.percentile(centered_spec[off_carrier], config.SPECTRUM_NOISE_PERCENTILE))
        cw.ax_centered.set_ylim(noise_y - 6.0, peak_dbfs + 4.0)

    snr_db, sb_pos, sb_neg, noise_floor_sb = compute_sideband_snr(
        centered_freqs_khz,
        centered_spec,
        config.SIDEBAND_OFFSET_KHZ,
        config.SIDEBAND_WINDOW_HZ,
    )
    cw.sideband_scatter.set_offsets(
        np.array([[-config.SIDEBAND_OFFSET_KHZ, sb_neg], [config.SIDEBAND_OFFSET_KHZ, sb_pos]])
    )
    dot_color = "#44ff88" if snr_db >= config.SNR_LOCK_THRESHOLD_DB else "#ff4444"
    cw.sideband_scatter.set_facecolor([dot_color, dot_color])
    cw.snr_threshold_line.set_ydata([noise_floor_sb + config.SNR_LOCK_THRESHOLD_DB])

    if centered_spec.size:
        remapped = remap_and_compress_centered(cw.centered_axis, centered_freqs_khz, centered_spec)
        cw.waterfall_data[:-1] = cw.waterfall_data[1:]
        if config.WATERFALL_ROWS > 1:
            remapped = (
                config.WATERFALL_ROW_BLEND * remapped
                + (1.0 - config.WATERFALL_ROW_BLEND) * cw.waterfall_data[-2]
            )
        cw.waterfall_data[-1] = remapped
        cw.waterfall_img.set_data(cw.waterfall_data)
        wf_mask = np.abs(cw.centered_axis) > config.NOISE_EXCLUDE_CARRIER_KHZ
        valid = cw.waterfall_data[:, wf_mask]
        valid = valid[valid > (config.DBFS_FLOOR + 1.0)]
        if valid.size:
            nf = float(np.percentile(valid, config.WATERFALL_NOISE_PERCENTILE))
            cw.waterfall_img.set_clim(
                vmin=nf - config.WATERFALL_CLIM_HEADROOM_DB,
                vmax=nf + config.WATERFALL_DYN_RANGE_DB,
            )

    bb.status.set_text(
        f"RX={config.RX_URI} | LO={config.FREQ_HZ/1e9:.3f} GHz | RMS={rms:.4f} FS | "
        f"Exciter={peak_hz:,.1f} Hz @ {peak_dbfs:.1f} dBFS | DC={dc_level:.1f} dBFS | "
        f"Carrier-DC={peak_dbfs - dc_level:.1f} dB"
    )
    cw.status.set_text(
        f"Carrier={peak_hz:,.1f} Hz | Span=±{config.CENTERED_SPAN_HZ/1000.0:.0f} kHz | "
        f"SB SNR={snr_db:+.1f} dB (+{config.SIDEBAND_OFFSET_KHZ:.0f}kHz={sb_pos:.1f} "
        f"-{config.SIDEBAND_OFFSET_KHZ:.0f}kHz={sb_neg:.1f} Noise={noise_floor_sb:.1f} dBFS)"
    )
    cw.fig.canvas.draw_idle()

    return x_focus, peak_hz


def _apply_force_unlock(runtime: Any) -> None:
    if runtime.ncc_lock:
        if runtime.ncc_abs_ema <= config.NCC_FORCE_UNLOCK_THRESHOLD:
            runtime.ncc_force_unlock_streak += 1
        else:
            runtime.ncc_force_unlock_streak = max(
                0,
                runtime.ncc_force_unlock_streak - config.NCC_FORCE_UNLOCK_COUNTER_DECAY,
            )
        if runtime.ncc_force_unlock_streak >= config.NCC_FORCE_UNLOCK_FRAMES:
            print(
                f"[RX LOCK] force unlock: weak NCC EMA {runtime.ncc_abs_ema:.3f} for "
                f"{runtime.ncc_force_unlock_streak} frames",
                flush=True,
            )
            runtime.ncc_lock = False
            runtime.ncc_enter_count = 0
            runtime.ncc_exit_count = 0
            runtime.decode_lock_grace_chips = 0
            runtime.ncc_force_unlock_streak = 0
    else:
        runtime.ncc_force_unlock_streak = 0


def process_demod_frame(runtime: Any, x_dc: np.ndarray, peak_hz: float) -> None:
    """Run demodulation, lock policy, packet decode, and chip view updates."""
    nw = runtime.nw
    demod = bandpass_filter_around_carrier(
        x_dc, peak_hz, config.SAMPLE_RATE, config.DEMOD_FILTER_PASSBAND_HZ
    )
    demod = runtime.cfo.correct_chunk(demod, confidence=runtime.ncc_abs_ema)
    ncc_val = coherent_ncc(demod, peak_hz, config.SUBCARRIER_HZ, config.SAMPLE_RATE)
    nw.ncc_history = np.roll(nw.ncc_history, -1)
    nw.ncc_history[-1] = ncc_val
    nw.line_ncc.set_ydata(nw.ncc_history)

    lock_before_update = runtime.ncc_lock
    (
        runtime.ncc_abs_ema,
        runtime.ncc_lock,
        runtime.ncc_enter_count,
        runtime.ncc_exit_count,
    ) = update_lock_hysteresis(
        ncc_abs_ema=runtime.ncc_abs_ema,
        ncc_lock=runtime.ncc_lock,
        ncc_enter_count=runtime.ncc_enter_count,
        ncc_exit_count=runtime.ncc_exit_count,
        abs_ncc=abs(ncc_val),
    )

    _apply_force_unlock(runtime)

    runtime.phase_sample_buffer = np.concatenate((runtime.phase_sample_buffer, demod))
    chips_added = sum(
        slice_chips_for_phase(
            phase=p,
            st=st,
            peak_hz=peak_hz,
            phase_sample_buffer=runtime.phase_sample_buffer,
            bit_samples=runtime.bit_samples,
            ncc_lock=runtime.ncc_lock,
            env_chip_metric_history_by_phase=runtime.env_chip_metric_history_by_phase,
            env_chip_decision_history_by_phase=runtime.env_chip_decision_history_by_phase,
        )
        for p, st in runtime.phase_state.items()
    )

    ranked = (
        rank_phases(
            phase_offsets=runtime.phase_offsets,
            phase_state=runtime.phase_state,
            env_chip_metric_history_by_phase=runtime.env_chip_metric_history_by_phase,
            phase_score_window_chips=config.PHASE_SCORE_WINDOW_CHIPS,
            env_plot_len=runtime.nw.env_plot_len,
        )
        if chips_added > 0
        else []
    )

    (
        runtime.ncc_lock,
        runtime.ncc_enter_count,
        runtime.ncc_exit_count,
        runtime.decode_lock_grace_chips,
        best_phase_structured,
        runtime.lock_policy,
    ) = apply_lock_policy(
        ranked=ranked,
        phase_state=runtime.phase_state,
        ncc_lock=runtime.ncc_lock,
        ncc_abs_ema=runtime.ncc_abs_ema,
        lock_before_update=lock_before_update,
        ncc_enter_count=runtime.ncc_enter_count,
        ncc_exit_count=runtime.ncc_exit_count,
        decode_lock_grace_chips=runtime.decode_lock_grace_chips,
        chips_added=chips_added,
        state=runtime.lock_policy,
    )

    decode_allowed = (
        (not config.PACKET_DECODE_REQUIRE_LOCK)
        or runtime.ncc_lock
        or runtime.decode_lock_grace_chips > 0
        or runtime.ncc_abs_ema >= config.NCC_SOFT_DECODE_THRESHOLD
        or best_phase_structured
    )
    if config.PACKET_DECODE_ENABLED and decode_allowed:
        if not ranked:
            ranked = rank_phases(
                phase_offsets=runtime.phase_offsets,
                phase_state=runtime.phase_state,
                env_chip_metric_history_by_phase=runtime.env_chip_metric_history_by_phase,
                phase_score_window_chips=config.PHASE_SCORE_WINDOW_CHIPS,
                env_plot_len=runtime.nw.env_plot_len,
            )
        decode_phases = ranked[: max(1, config.PACKET_DECODE_TOP_PHASES)]
        packet_candidates: list[PacketCandidate] = []
        for p, score in decode_phases:
            packet_candidates.extend(
                try_decode_packets(
                    phase=p,
                    st=runtime.phase_state[p],
                    phase_score=score,
                    packet_header_bits=runtime.packet_header_bits,
                    preamble_bits=runtime.preamble_bits,
                    payload_patterns_bits=runtime.payload_patterns_bits,
                    payload_bits_len=runtime.payload_bits_len,
                )
            )
        if (
            not packet_candidates
            and config.PACKET_DECODE_FALLBACK_ALL_PHASES
            and len(decode_phases) < len(ranked)
        ):
            for p, score in ranked[len(decode_phases) :]:
                packet_candidates.extend(
                    try_decode_packets(
                        phase=p,
                        st=runtime.phase_state[p],
                        phase_score=score,
                        packet_header_bits=runtime.packet_header_bits,
                        preamble_bits=runtime.preamble_bits,
                        payload_patterns_bits=runtime.payload_patterns_bits,
                        payload_bits_len=runtime.payload_bits_len,
                    )
                )
        (
            runtime.pending_packet_candidate,
            runtime.last_packet_accept_bit,
            runtime.decoded_packets,
            packet_status_text,
            packet_status_hold,
        ) = accept_best_packet_candidate(
            candidates=packet_candidates,
            phase_state=runtime.phase_state,
            pending_packet_candidate=runtime.pending_packet_candidate,
            last_packet_accept_bit=runtime.last_packet_accept_bit,
            decoded_packets=runtime.decoded_packets,
            ncc_lock=runtime.ncc_lock,
            ncc_abs_ema=runtime.ncc_abs_ema,
        )
        if packet_status_text is not None:
            runtime.packet_status_text = packet_status_text
        if packet_status_hold is not None:
            runtime.packet_status_hold = packet_status_hold

    if chips_added > 0:
        if not ranked:
            ranked = rank_phases(
                phase_offsets=runtime.phase_offsets,
                phase_state=runtime.phase_state,
                env_chip_metric_history_by_phase=runtime.env_chip_metric_history_by_phase,
                phase_score_window_chips=config.PHASE_SCORE_WINDOW_CHIPS,
                env_plot_len=runtime.nw.env_plot_len,
            )
        runtime.display_phase, runtime.phase_score_summary = update_chip_view(
            nw=runtime.nw,
            ranked=ranked,
            phase_state=runtime.phase_state,
            env_chip_metric_history_by_phase=runtime.env_chip_metric_history_by_phase,
            env_chip_decision_history_by_phase=runtime.env_chip_decision_history_by_phase,
        )
        runtime.total_bits += chips_added
        while runtime.total_bits >= runtime.next_debug_bits:
            emit_chip_debug(
                best_phase=ranked[0][0],
                phase_state=runtime.phase_state,
                ncc_abs_ema=runtime.ncc_abs_ema,
                ncc_lock=runtime.ncc_lock,
                repetition_chips=config.REPETITION_CHIPS,
                debug_bit_tail=config.TERMINAL_DEBUG_BIT_TAIL,
            )
            runtime.next_debug_bits += config.TERMINAL_DEBUG_BITS_EVERY
        min_next = min(s.next_sample for s in runtime.phase_state.values())
        if min_next > runtime.bit_samples:
            trim = min_next - runtime.bit_samples
            runtime.phase_sample_buffer = runtime.phase_sample_buffer[trim:]
            for s in runtime.phase_state.values():
                s.next_sample -= trim

    if runtime.packet_status_hold > 0:
        runtime.packet_status_hold -= 1
    else:
        runtime.packet_status_text = config.packet_status_default_text()

    view_st = runtime.phase_state[runtime.display_phase]
    nw.ncc_status.set_text(
        f"NCC={ncc_val:+.3f} | EMA={runtime.ncc_abs_ema:.3f} | "
        f"{'LOCKED' if runtime.ncc_lock else 'searching'} | "
        f"rCFO={runtime.cfo.rcfo_hz:+.1f} Hz | "
        f"Carrier={peak_hz:,.0f} Hz | chip_view_phase={runtime.display_phase} "
        f"thr={view_st.bit_ncc_threshold:.3f} | {runtime.lock_policy.lock_quality_summary} | "
        f"{runtime.phase_score_summary} | {runtime.packet_status_text}"
    )
    nw.fig.canvas.draw_idle()
