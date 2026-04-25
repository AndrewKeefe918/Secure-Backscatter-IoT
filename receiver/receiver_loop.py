"""ReceiverLoop â€” per-frame state and FuncAnimation callback (FSK version).

Differences from OOK ReceiverLoop:
  - Chip decisions use coherent_fsk_metrics() comparing m_f1 vs m_f0,
    not absolute power thresholds. This eliminates threshold drift,
    which was the dominant failure mode in the OOK build.
  - The third window plots both FSK metrics per chip instead of a
    single |NCC| trace.
  - Repetition coding defaults to 1 (disabled) because FSK's self-
    normalising decisions don't benefit from it the way OOK did, and
    it actually HURT OOK by correlating threshold-drift errors.
"""

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

from . import config as config
from .dsp import (
    normalize_iq,
    dc_block_filter,
    compute_spectrum_dbfs,
    bandpass_filter_around_carrier,
    estimate_residual_cfo_hz,
    derotate_frequency,
    coherent_fsk_metrics,
    compute_sideband_snr,
    ema_spectrum_power_domain,
    find_exciter_peak,
    remap_and_compress_centered,
    smooth_1d,
)
from .packet_decoder import (
    bytes_to_bit_list,
    bits_to_bytes,
    find_header_match,
    majority_decode_triplets,
    bits_to_text,
    safe_ascii,
)
from .gui_setup import BasebandWindow, CarrierWindow, FskWindow


@dataclass
class PhaseState:
    next_sample: int
    chips: list[int] = field(default_factory=list)
    chip_metrics_f1: list[float] = field(default_factory=list)  # for plotting
    chip_metrics_f0: list[float] = field(default_factory=list)  # for plotting
    base_chip_index: int = 0
    search_start_by_offset: list[int] = field(
        default_factory=lambda: [0] * config.REPETITION_CHIPS
    )
    last_header_abs_by_offset: list[int] = field(
        default_factory=lambda: [-1] * config.REPETITION_CHIPS
    )
    chips_seen: int = 0


class ReceiverLoop:
    """Encapsulates all mutable receiver state and the per-frame update callback."""

    def __init__(
        self, sdr: object, bb: BasebandWindow, cw: CarrierWindow, fw: FskWindow
    ) -> None:
        self.sdr, self.bb, self.cw, self.fw = sdr, bb, cw, fw
        self.stop_requested = False

        # Spectrum / carrier state
        self.smoothed_raw = np.array([], dtype=np.float64)
        self.smoothed_dc = np.array([], dtype=np.float64)
        self.dc_prev_x = np.complex64(0.0 + 0.0j)
        self.dc_prev_y = np.complex64(0.0 + 0.0j)
        self.prev_peak_dbfs: float | None = None
        self.prev_peak_hz = 0.0

        # Packet decode state
        self.bit_samples = config.SAMPLES_PER_CHIP
        self.phase_offsets = list(range(0, self.bit_samples, config.BIT_PHASE_STEP_SAMPLES))
        self.phase_sample_buffer = np.array([], dtype=np.complex64)
        self.phase_state = {p: PhaseState(next_sample=p) for p in self.phase_offsets}
        self.packet_header_bits = bytes_to_bit_list(config.PREAMBLE_BYTES + config.SYNC_BYTES)
        self.expected_payload_bits = bytes_to_bit_list(config.PAYLOAD_BYTES)
        self.payload_bits_len = len(config.PAYLOAD_BYTES) * 8
        self.packet_status_text = "Waiting for preamble+sync"
        self.packet_status_hold = 0
        self.decoded_packets = 0
        self.total_bits = 0
        self.next_debug_bits = config.TERMINAL_DEBUG_BITS_EVERY

        # Lock-status hysteresis (drives display only, not decode gate)
        self.ncc_abs_ema = 0.0
        self.ncc_lock = False
        self.ncc_enter_count = 0
        self.ncc_exit_count = 0

        # Two-step CFO adaptation state (coarse average + fine residual tracking)
        self.cfo_coarse_hz = 0.0
        self.cfo_fine_hz = 0.0
        self.cfo_total_hz = 0.0
        self.cfo_phase_rad = 0.0

        # Per-frame header log throttling to keep terminal I/O from stalling decode.
        self.header_logs_this_frame = 0
        self.header_logs_suppressed_this_frame = 0

        # Runtime timing telemetry (host jitter / overrun risk visibility).
        self.frame_index = 0
        self.last_frame_start_s: float | None = None
        self.buffer_duration_s = float(config.RX_BUFFER_SIZE) / float(config.SAMPLE_RATE)
        self.update_interval_s = float(config.ANIMATION_INTERVAL_MS) / 1000.0
        self.realtime_budget_s = max(self.buffer_duration_s, self.update_interval_s)

    def close(self) -> None:
        # Keep a common interface with RX-only loop.
        return

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_lock_hysteresis(self, abs_ncc: float) -> None:
        self.ncc_abs_ema = (
            (1.0 - config.NCC_DISPLAY_ALPHA) * self.ncc_abs_ema
            + config.NCC_DISPLAY_ALPHA * abs_ncc
        )
        if not self.ncc_lock:
            self.ncc_enter_count = (
                self.ncc_enter_count + 1 if self.ncc_abs_ema >= config.NCC_ENTER_THRESHOLD else 0
            )
            if self.ncc_enter_count >= config.NCC_ENTER_FRAMES:
                self.ncc_lock, self.ncc_exit_count = True, 0
        else:
            self.ncc_exit_count = (
                self.ncc_exit_count + 1 if self.ncc_abs_ema <= config.NCC_EXIT_THRESHOLD else 0
            )
            if self.ncc_exit_count >= config.NCC_EXIT_FRAMES:
                self.ncc_lock, self.ncc_enter_count = False, 0

    def _slice_chips_for_phase(self, state: PhaseState, peak_hz: float) -> int:
        """Slice the phase sample buffer using FSK chip decisions.

        Each 50 ms chunk produces metrics m_f1 (power at FSK_F1_HZ) and
        m_f0 (power at FSK_F0_HZ). The chip is '1' if m_f1 dominates,
        '0' if m_f0 dominates. If neither metric exceeds the presence
        floor or the gap is below the dead zone, the chip is decoded as 0.
        """
        added = 0
        while state.next_sample + self.bit_samples <= len(self.phase_sample_buffer):
            chunk = self.phase_sample_buffer[state.next_sample : state.next_sample + self.bit_samples]
            m_f1, m_f0, decision = coherent_fsk_metrics(
                chunk,
                peak_hz,
                config.FSK_F1_HZ,
                config.FSK_F0_HZ,
                config.SAMPLE_RATE,
            )

            # Chip decision: differential, with a presence floor.
            both_below_floor = max(m_f1, m_f0) < config.FSK_PRESENCE_FLOOR
            in_dead_zone = abs(decision) < config.FSK_DECISION_DEAD_ZONE
            if both_below_floor or in_dead_zone:
                chip_value = 0
            else:
                chip_value = 1 if decision > 0 else 0

            state.chips.append(chip_value)
            state.chip_metrics_f1.append(m_f1)
            state.chip_metrics_f0.append(m_f0)
            state.next_sample += self.bit_samples
            state.chips_seen += 1
            added += 1

        # Cap history length
        max_chips = config.PHASE_HISTORY_BITS * config.REPETITION_CHIPS
        if len(state.chips) > max_chips:
            drop = len(state.chips) - max_chips
            del state.chips[:drop]
            del state.chip_metrics_f1[:drop]
            del state.chip_metrics_f0[:drop]
            state.base_chip_index += drop
        return added

    def _try_decode_packets(self, phase: int, state: PhaseState) -> None:
        """Scan all chip-offsets for header matches and emit packets on detection."""
        for decode_offset in range(config.REPETITION_CHIPS):
            decoded_bits = majority_decode_triplets(state.chips, decode_offset)
            base_bit_index = (state.base_chip_index + decode_offset) // config.REPETITION_CHIPS

            def _scan_with_header(
                header_bits: list[int],
                search_list: list[int],
                last_abs_list: list[int],
                tag: str,
            ) -> None:
                candidates_processed = 0
                while True:
                    if candidates_processed >= int(config.HEADER_CANDIDATES_PER_SCAN):
                        break
                    search_start = search_list[decode_offset]
                    header_idx, header_errors = find_header_match(
                        decoded_bits, header_bits, search_start, config.HEADER_MAX_BIT_ERRORS
                    )
                    if header_idx < 0:
                        search_list[decode_offset] = max(0, len(decoded_bits) - len(header_bits) + 1)
                        break

                    header_abs = base_bit_index + header_idx
                    if header_abs <= last_abs_list[decode_offset]:
                        search_list[decode_offset] = header_idx + 1
                        continue

                    payload_start = header_idx + len(header_bits)
                    payload_end = payload_start + self.payload_bits_len
                    if payload_end > len(decoded_bits):
                        search_list[decode_offset] = header_idx
                        break

                    payload_bits = decoded_bits[payload_start:payload_end]
                    payload = bits_to_bytes(payload_bits)
                    payload_hex = payload.hex().upper() if payload else ""
                    payload_errors_normal = sum(
                        1
                        for left, right in zip(payload_bits, self.expected_payload_bits)
                        if left != right
                    )
                    if config.ALLOW_INVERTED_PAYLOAD_MATCH:
                        payload_errors_inverted = sum(
                            1
                            for left, right in zip(payload_bits, self.expected_payload_bits)
                            if (1 - left) != right
                        )
                    else:
                        payload_errors_inverted = payload_errors_normal
                    use_inverted = payload_errors_inverted < payload_errors_normal
                    payload_errors = min(payload_errors_normal, payload_errors_inverted)
                    polarity = "inverted" if use_inverted else "normal"
                    if self.header_logs_this_frame < int(config.HEADER_LOGS_PER_FRAME):
                        print(
                            f"[RX HEADER] phase={phase} chip_offset={decode_offset} "
                            f"bit={header_abs} {tag} header_errors={header_errors} "
                            f"payload_errors={payload_errors} payload_polarity={polarity} "
                            f"payload_hex={payload_hex} payload_ascii={safe_ascii(payload)!r}",
                            flush=True,
                        )
                        self.header_logs_this_frame += 1
                    else:
                        self.header_logs_suppressed_this_frame += 1

                    if payload_errors <= config.PAYLOAD_MAX_BIT_ERRORS:
                        self.decoded_packets += 1
                        self.packet_status_text = (
                            f"Packet: OPEN detected at phase {phase}, offset {decode_offset}, "
                            f"{tag}, payload_errors={payload_errors}, payload_polarity={polarity}"
                        )
                        print(
                            f"[RX PACKET {self.decoded_packets}] phase={phase} chip_offset={decode_offset} "
                            f"bit={header_abs} {tag} header_errors={header_errors} "
                            f"payload_errors={payload_errors} payload_polarity={polarity} PREAMBLE+SYNC+OPEN",
                            flush=True,
                        )
                    else:
                        self.packet_status_text = (
                            f"Packet: header phase {phase}, offset {decode_offset}, "
                            f"{tag}, errors={header_errors}, payload_errors={payload_errors}, "
                            f"payload_polarity={polarity}, payload={payload!r}"
                        )

                    self.packet_status_hold = config.PACKET_STATUS_HOLD_FRAMES
                    last_abs_list[decode_offset] = header_abs
                    search_list[decode_offset] = header_idx + 1
                    candidates_processed += 1

            _scan_with_header(
                self.packet_header_bits,
                state.search_start_by_offset,
                state.last_header_abs_by_offset,
                "primary",
            )

    def _emit_debug(self) -> None:
        best_phase = self._select_best_phase_for_display()
        st = self.phase_state[best_phase]
        best_decoded = majority_decode_triplets(st.chips, 0)
        chip_tail = bits_to_text(st.chips[-config.TERMINAL_DEBUG_BIT_TAIL:])
        bit_tail = bits_to_text(best_decoded[-(config.TERMINAL_DEBUG_BIT_TAIL // config.REPETITION_CHIPS):])
        print(
            f"[RX DEBUG] phase={best_phase} chips={st.chips_seen} "
            f"lock={'1' if self.ncc_lock else '0'} ncc_ema={self.ncc_abs_ema:.3f} "
            f"cfo_hz={self.cfo_total_hz:+.1f} (coarse={self.cfo_coarse_hz:+.1f} fine={self.cfo_fine_hz:+.1f}) "
            f"chip_tail={chip_tail} bit_tail={bit_tail}",
            flush=True,
        )

    def _phase_quality_score(self, state: PhaseState) -> float:
        """Score a phase by recent symbol structure + FSK metric separation.

        This avoids a bias toward phase 0 that can happen when choosing by
        chips_seen only.
        """
        if not state.chips:
            return -1e9

        recent_len = min(96, len(state.chips))
        recent = state.chips[-recent_len:]
        transitions = sum(1 for left, right in zip(recent, recent[1:]) if left != right)
        transition_rate = transitions / max(recent_len - 1, 1)

        sep_len = min(96, len(state.chip_metrics_f1), len(state.chip_metrics_f0))
        if sep_len > 0:
            m1 = np.asarray(state.chip_metrics_f1[-sep_len:], dtype=np.float64)
            m0 = np.asarray(state.chip_metrics_f0[-sep_len:], dtype=np.float64)
            separation = float(np.mean(np.abs(m1 - m0)))
        else:
            separation = 0.0

        # Tiny chips_seen term only breaks ties for equally good phases.
        return (8.0 * transition_rate) + (2.0 * separation) + (1e-4 * state.chips_seen)

    def _select_best_phase_for_display(self) -> int:
        return max(self.phase_offsets, key=lambda p: self._phase_quality_score(self.phase_state[p]))

    def _update_fsk_metric_history(self, m_f1: float, m_f0: float, decision: float) -> None:
        self.fw.metric_history_f1 = np.roll(self.fw.metric_history_f1, -1)
        self.fw.metric_history_f1[-1] = m_f1
        self.fw.metric_history_f0 = np.roll(self.fw.metric_history_f0, -1)
        self.fw.metric_history_f0[-1] = m_f0
        self.fw.metric_history_decision = np.roll(self.fw.metric_history_decision, -1)
        self.fw.metric_history_decision[-1] = decision

        self.fw.line_m_f1.set_ydata(self.fw.metric_history_f1)
        self.fw.line_m_f0.set_ydata(self.fw.metric_history_f0)
        self.fw.line_decision.set_ydata(self.fw.metric_history_decision)

    def _update_chip_decision_plot(self) -> None:
        """Show the most recent CHIP_VIEW_HISTORY chip metrics + decisions."""
        # Pick the phase with best recent decode quality as display source.
        best_phase = self._select_best_phase_for_display()
        st = self.phase_state[best_phase]
        n = self.fw.chip_history_len

        if not st.chips:
            return

        recent_chips = st.chips[-n:]
        recent_m_f1 = st.chip_metrics_f1[-n:]
        recent_m_f0 = st.chip_metrics_f0[-n:]

        # Pad if we don't have enough yet
        pad = n - len(recent_chips)
        if pad > 0:
            recent_chips = [0] * pad + recent_chips
            recent_m_f1 = [0.0] * pad + recent_m_f1
            recent_m_f0 = [0.0] * pad + recent_m_f0

        self.fw.line_chip_m_f1.set_ydata(np.array(recent_m_f1, dtype=np.float64))
        self.fw.line_chip_m_f0.set_ydata(np.array(recent_m_f0, dtype=np.float64))
        # Scale the binary decision to ~0.85 so it sits visibly above the metrics
        self.fw.line_chip_decision.set_ydata(
            np.array([0.85 if c else 0.05 for c in recent_chips], dtype=np.float64)
        )

    # ------------------------------------------------------------------
    # Per-frame callback
    # ------------------------------------------------------------------

    def update(self, _frame: int) -> tuple:
        bb, cw, fw = self.bb, self.cw, self.fw
        if config.RENDER_PLOTS:
            artists = (
                bb.line_i, bb.line_q, bb.line_fft_raw, bb.line_fft_dc_blocked,
                bb.exciter_marker, cw.line_centered, cw.waterfall_img, bb.status, cw.status,
            )
        else:
            artists = tuple()
        if self.stop_requested:
            return artists

        frame_start_s = perf_counter()
        self.frame_index += 1
        self.header_logs_this_frame = 0
        self.header_logs_suppressed_this_frame = 0
        self.last_frame_start_s = frame_start_s

        # ---- Acquire + DC-block --------------------------------------------
        x_raw = normalize_iq(self.sdr.rx())
        x_dc, self.dc_prev_x, self.dc_prev_y = dc_block_filter(
            x_raw, self.dc_prev_x, self.dc_prev_y, config.DC_BLOCK_ALPHA
        )
        if config.RENDER_PLOTS:
            shown = x_raw[: config.TIME_SAMPLES]
            bb.line_i.set_ydata(np.real(shown))
            bb.line_q.set_ydata(np.imag(shown))

        # ---- Spectrum (EMA in power domain) --------------------------------
        freqs_hz, raw_dbfs = compute_spectrum_dbfs(x_raw, config.SAMPLE_RATE)
        _, dc_dbfs_full = compute_spectrum_dbfs(x_dc, config.SAMPLE_RATE)
        self.smoothed_raw = ema_spectrum_power_domain(raw_dbfs, self.smoothed_raw, config.FFT_AVG_ALPHA)
        self.smoothed_dc = ema_spectrum_power_domain(dc_dbfs_full, self.smoothed_dc, config.FFT_AVG_ALPHA)

        in_view = np.abs(freqs_hz) <= config.SPECTRUM_SPAN_HZ
        rms = float(np.sqrt(np.mean(np.abs(x_raw) ** 2)))

        # ---- Carrier lock ---------------------------------------------------
        peak_hz, peak_dbfs = find_exciter_peak(
            freqs_hz, self.smoothed_dc, in_view, self.prev_peak_hz,
            config.EXCITER_SEARCH_MIN_HZ, config.EXCITER_SEARCH_MAX_HZ,
        )
        self.prev_peak_hz = peak_hz
        if config.RENDER_PLOTS:
            bb.exciter_marker.set_xdata([peak_hz / 1000.0, peak_hz / 1000.0])

        dc_idx = int(np.argmin(np.abs(freqs_hz)))
        dc_level = float(self.smoothed_raw[dc_idx])
        dc_to_carrier_db = peak_dbfs - dc_level
        if config.RENDER_PLOTS:
            # ---- Baseband spectrum plot ----------------------------------------
            freqs_khz = freqs_hz[in_view] / 1000.0
            bb.line_fft_raw.set_data(freqs_khz, self.smoothed_raw[in_view])
            bb.line_fft_dc_blocked.set_data(freqs_khz, self.smoothed_dc[in_view])

            if self.prev_peak_dbfs is None or abs(peak_dbfs - self.prev_peak_dbfs) > 5.0:
                cw.ax_centered.set_ylim(peak_dbfs - 40.0, peak_dbfs + 10.0)
                self.prev_peak_dbfs = peak_dbfs

            # ---- Centered spectrum ---------------------------------------------
            centered_freqs_hz = freqs_hz - peak_hz
            centered_mask = np.abs(centered_freqs_hz) <= config.CENTERED_SPAN_HZ
            centered_freqs_khz = centered_freqs_hz[centered_mask] / 1000.0
            centered_spec = smooth_1d(self.smoothed_dc[centered_mask], config.CENTERED_FREQ_SMOOTH_BINS)
            cw.line_centered.set_data(centered_freqs_khz, centered_spec)

            off_carrier = np.abs(centered_freqs_khz) > 0.5
            if off_carrier.any():
                noise_y = float(np.percentile(centered_spec[off_carrier], 20))
                cw.ax_centered.set_ylim(noise_y - 6.0, peak_dbfs + 4.0)
                self.prev_peak_dbfs = peak_dbfs

            # ---- Sideband markers (uses '1' frequency) -------------------------
            snr_db, sb_pos, sb_neg, noise_floor_sb = compute_sideband_snr(
                centered_freqs_khz, centered_spec,
                config.SIDEBAND_OFFSET_KHZ, config.SIDEBAND_WINDOW_HZ,
            )
            cw.sideband_scatter.set_offsets(np.array([
                [-config.SIDEBAND_OFFSET_KHZ, sb_neg], [config.SIDEBAND_OFFSET_KHZ, sb_pos]
            ]))
            dot_color = "#44ff88" if snr_db >= config.SNR_LOCK_THRESHOLD_DB else "#ff4444"
            cw.sideband_scatter.set_facecolor([dot_color, dot_color])
            cw.snr_threshold_line.set_ydata([noise_floor_sb + config.SNR_LOCK_THRESHOLD_DB])

            # ---- Waterfall ------------------------------------------------------
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

                wf_axis_mask = np.abs(cw.centered_axis) > 0.5
                valid = cw.waterfall_data[:, wf_axis_mask]
                valid = valid[valid > -139.0]
                if valid.size:
                    nf = float(np.percentile(valid, 15))
                    cw.waterfall_img.set_clim(vmin=nf - 1.0, vmax=nf + config.WATERFALL_DYN_RANGE_DB)

            # ---- Status text ----------------------------------------------------
            bb.status.set_text(
                f"RX={config.RX_URI} | LO={config.FREQ_HZ/1e9:.3f} GHz | RMS={rms:.4f} FS | "
                f"Exciter={peak_hz:,.1f} Hz @ {peak_dbfs:.1f} dBFS | DC={dc_level:.1f} dBFS | "
                f"Carrier-DC={dc_to_carrier_db:.1f} dB"
            )
            cw.status.set_text(
                f"Carrier={peak_hz:,.1f} Hz | Span=Â±{config.CENTERED_SPAN_HZ/1000.0:.0f} kHz | "
                f"SB SNR={snr_db:+.1f} dB at Â±{config.SIDEBAND_OFFSET_KHZ:.1f} kHz "
                f"(+={sb_pos:.1f} -={sb_neg:.1f} Noise={noise_floor_sb:.1f} dBFS)"
            )

        # ---- FSK demod + packet decode -------------------------------------
        if peak_hz != 0.0:
            demod = bandpass_filter_around_carrier(
                x_dc, peak_hz, config.SAMPLE_RATE, config.DEMOD_FILTER_PASSBAND_HZ
            )

            if config.CFO_CORRECTION_ENABLED and demod.size:
                n = np.arange(demod.size, dtype=np.float64)
                mix = np.exp(-1j * 2.0 * np.pi * float(peak_hz) * n / float(config.SAMPLE_RATE))
                demod_bb = demod.astype(np.complex128) * mix

                residual_hat_hz = estimate_residual_cfo_hz(demod_bb.astype(np.complex64), config.SAMPLE_RATE)
                max_abs = float(config.CFO_MAX_ABS_HZ)
                residual_hat_hz = float(np.clip(residual_hat_hz, -max_abs, max_abs))

                self.cfo_coarse_hz = (
                    (1.0 - config.CFO_COARSE_ALPHA) * self.cfo_coarse_hz
                    + config.CFO_COARSE_ALPHA * residual_hat_hz
                )
                residual_part_hz = residual_hat_hz - self.cfo_coarse_hz
                self.cfo_fine_hz = (
                    (1.0 - config.CFO_FINE_ALPHA) * self.cfo_fine_hz
                    + config.CFO_FINE_ALPHA * residual_part_hz
                )
                self.cfo_total_hz = float(
                    np.clip(self.cfo_coarse_hz + self.cfo_fine_hz, -max_abs, max_abs)
                )

                demod, self.cfo_phase_rad = derotate_frequency(
                    demod,
                    self.cfo_total_hz,
                    config.SAMPLE_RATE,
                    self.cfo_phase_rad,
                )
            else:
                self.cfo_coarse_hz = 0.0
                self.cfo_fine_hz = 0.0
                self.cfo_total_hz = 0.0
                self.cfo_phase_rad = 0.0

            # Per-buffer FSK metrics for the top trace in the FSK window.
            m_f1_buf, m_f0_buf, decision_buf = coherent_fsk_metrics(
                demod, peak_hz, config.FSK_F1_HZ, config.FSK_F0_HZ, config.SAMPLE_RATE
            )
            if config.RENDER_PLOTS:
                self._update_fsk_metric_history(m_f1_buf, m_f0_buf, decision_buf)

            # Lock hysteresis driven by the larger of the two metrics
            self._update_lock_hysteresis(max(m_f1_buf, m_f0_buf))

            # Multi-phase chip slicing + packet decode
            self.phase_sample_buffer = np.concatenate((self.phase_sample_buffer, demod))
            chips_added = 0
            for phase, state in self.phase_state.items():
                phase_added = self._slice_chips_for_phase(state, peak_hz)
                chips_added += phase_added
                if config.PACKET_DECODE_ENABLED and phase_added > 0:
                    self._try_decode_packets(phase, state)

            if config.RENDER_PLOTS:
                self._update_chip_decision_plot()

            if chips_added > 0:
                self.total_bits += chips_added
                while self.total_bits >= self.next_debug_bits:
                    self._emit_debug()
                    self.next_debug_bits += config.TERMINAL_DEBUG_BITS_EVERY

                # Trim consumed samples
                min_next = min(s.next_sample for s in self.phase_state.values())
                if min_next > self.bit_samples:
                    trim = min_next - self.bit_samples
                    self.phase_sample_buffer = self.phase_sample_buffer[trim:]
                    for s in self.phase_state.values():
                        s.next_sample -= trim

            if self.packet_status_hold > 0:
                self.packet_status_hold -= 1
            else:
                self.packet_status_text = "Waiting for preamble+sync"

            if self.header_logs_suppressed_this_frame > 0:
                print(
                    f"[RX HEADER] suppressed={self.header_logs_suppressed_this_frame} "
                    f"(cap={int(config.HEADER_LOGS_PER_FRAME)}/frame)",
                    flush=True,
                )

            if config.RENDER_PLOTS:
                fw.status.set_text(
                    f"m_f1={m_f1_buf:.3f} | m_f0={m_f0_buf:.3f} | dec={decision_buf:+.3f} | "
                    f"{'LOCKED' if self.ncc_lock else 'searching'} | "
                    f"Carrier={peak_hz:,.0f} Hz | CFO={self.cfo_total_hz:+.1f} Hz | {self.packet_status_text}"
                )

        return artists

