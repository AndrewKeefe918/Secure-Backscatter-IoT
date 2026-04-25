"""ReceiverLoop — per-frame state and FuncAnimation callback."""

from dataclasses import dataclass, field

import numpy as np

from . import config
from .dsp import (
    normalize_iq, dc_block_filter, compute_spectrum_dbfs,
    bandpass_filter_around_carrier, coherent_ncc, coherent_chip_metric,
    compute_sideband_snr, ema_spectrum_power_domain, find_exciter_peak,
    remap_and_compress_centered, smooth_1d,
)
from .packet_decoder import (
    bytes_to_bit_list, bits_to_bytes, find_header_match,
    majority_decode_triplets, bits_to_text, safe_ascii,
)
from .gui_setup import BasebandWindow, CarrierWindow, NccWindow


@dataclass
class PhaseState:
    next_sample: int
    chips: list[int] = field(default_factory=list)
    base_chip_index: int = 0
    search_start_by_offset: list[int] = field(
        default_factory=lambda: [0] * config.REPETITION_CHIPS)
    last_header_abs_by_offset: list[int] = field(
        default_factory=lambda: [-1] * config.REPETITION_CHIPS)
    chips_seen: int = 0
    bit_ncc_noise_ema: float = 0.0
    bit_ncc_threshold: float = float(config.BIT_NCC_THRESHOLD)
    bit_ncc_low_ema: float = 0.0
    bit_ncc_high_ema: float = float(config.BIT_NCC_THRESHOLD)
    timing_error_ema: float = 0.0
    chip_stride_samples: float = 0.0
    chip_state: int = -1
    chip_rise_streak: int = 0
    chip_fall_streak: int = 0


@dataclass
class PacketCandidate:
    phase: int
    chip_offset: int
    header_abs: int
    header_errors: int
    preamble_transitions: int
    payload_errors: int
    payload: bytes
    phase_score: float
    observed_preamble: list[int]
    observed_sync: list[int]


def _packet_candidate_sort_key(candidate: PacketCandidate) -> tuple[float, ...]:
    return (
        candidate.payload_errors,
        candidate.header_errors,
        -candidate.phase_score,
        -candidate.preamble_transitions,
        candidate.header_abs,
    )


_UNLOCK_HIGH_FLOOR_MARGIN = 0.03
_UNLOCK_THRESHOLD_HEADROOM = 0.06
_UNLOCK_BLEND = 0.30
_LOCKED_THRESHOLD_HEADROOM = 0.12
_HIGH_SPIKE_REJECT_HEADROOM = 0.24
_HIGH_RELAX_ALPHA = 0.02


def _ema(prev: float, new: float, alpha: float) -> float:
    return (1.0 - alpha) * prev + alpha * new


class ReceiverLoop:
    def __init__(self, sdr, bb: BasebandWindow, cw: CarrierWindow, nw: NccWindow) -> None:
        self.sdr, self.bb, self.cw, self.nw = sdr, bb, cw, nw
        self.stop_requested = False

        self.smoothed_raw = np.array([], dtype=np.float64)
        self.smoothed_dc = np.array([], dtype=np.float64)
        self.dc_prev_x = np.complex64(0.0 + 0.0j)
        self.dc_prev_y = np.complex64(0.0 + 0.0j)
        self.prev_peak_dbfs: float | None = None
        self.prev_peak_hz = 0.0

        self.bit_samples = config.SAMPLES_PER_CHIP
        self.phase_offsets = list(range(0, self.bit_samples, config.BIT_PHASE_STEP_SAMPLES))
        self.phase_sample_buffer = np.array([], dtype=np.complex64)
        self.phase_state = {p: PhaseState(next_sample=p) for p in self.phase_offsets}
        for st in self.phase_state.values():
            st.chip_stride_samples = float(self.bit_samples)
        self.preamble_bits = bytes_to_bit_list(config.PREAMBLE_BYTES)
        self.packet_header_bits = bytes_to_bit_list(config.PREAMBLE_BYTES + config.SYNC_BYTES)
        self.payload_bits = bytes_to_bit_list(config.PAYLOAD_BYTES)
        self.payload_bits_len = len(config.PAYLOAD_BYTES) * 8
        self.packet_status_text = (
            "Continuous ON test: packet decode disabled"
            if not config.PACKET_DECODE_ENABLED
            else "Waiting for preamble+sync"
        )
        self.packet_status_hold = 0
        self.decoded_packets = 0
        self.last_packet_accept_bit = -10**9
        self.pending_packet_candidate: PacketCandidate | None = None
        self.total_bits = 0
        self.next_debug_bits = config.TERMINAL_DEBUG_BITS_EVERY

        self.ncc_abs_ema = 0.0
        self.ncc_lock = False
        self.ncc_enter_count = 0
        self.ncc_exit_count = 0
        self.decode_lock_grace_chips = 0

        plot_len = self.nw.env_plot_len
        self.display_phase = self.phase_offsets[0] if self.phase_offsets else 0
        self.env_chip_metric_history_by_phase = {
            p: np.zeros(plot_len, dtype=np.float64) for p in self.phase_offsets}
        self.env_chip_decision_history_by_phase = {
            p: np.zeros(plot_len, dtype=np.float64) for p in self.phase_offsets}
        self.phase_score_summary = ""

    # ---- lock / threshold helpers ------------------------------------

    def _update_lock_hysteresis(self, abs_ncc: float) -> None:
        self.ncc_abs_ema = _ema(self.ncc_abs_ema, abs_ncc, config.NCC_DISPLAY_ALPHA)
        if not self.ncc_lock:
            self.ncc_enter_count = self.ncc_enter_count + 1 if self.ncc_abs_ema >= config.NCC_ENTER_THRESHOLD else 0
            if self.ncc_enter_count >= config.NCC_ENTER_FRAMES:
                self.ncc_lock, self.ncc_exit_count = True, 0
        else:
            self.ncc_exit_count = self.ncc_exit_count + 1 if self.ncc_abs_ema <= config.NCC_EXIT_THRESHOLD else 0
            if self.ncc_exit_count >= config.NCC_EXIT_FRAMES:
                self.ncc_lock, self.ncc_enter_count = False, 0

    def _update_phase_threshold(self, st: PhaseState, abs_ncc: float, locked: bool) -> None:
        if config.CONTINUOUS_ON_TEST:
            # In continuous-ON mode there is no OFF-symbol cluster, so the standard
            # low/high threshold model is ill-posed. Track one signal cluster instead.
            if st.bit_ncc_high_ema <= 1e-9:
                st.bit_ncc_high_ema = abs_ncc
            else:
                st.bit_ncc_high_ema = _ema(
                    st.bit_ncc_high_ema, abs_ncc, config.CONTINUOUS_ON_THRESHOLD_ALPHA)
            st.bit_ncc_noise_ema = st.bit_ncc_high_ema
            if not config.ADAPTIVE_BIT_NCC_THRESHOLD:
                st.bit_ncc_threshold = float(config.BIT_NCC_THRESHOLD)
                return
            cand = config.CONTINUOUS_ON_THRESHOLD_SCALE * st.bit_ncc_high_ema
            st.bit_ncc_threshold = float(np.clip(
                cand, config.BIT_NCC_THRESHOLD_MIN, config.CONTINUOUS_ON_THRESHOLD_MAX))
            return

        # Two-cluster (low/high) EMA tracker; threshold blends between them.
        if st.bit_ncc_low_ema <= 1e-9:
            st.bit_ncc_low_ema = abs_ncc
            st.bit_ncc_high_ema = max(abs_ncc, config.BIT_NCC_THRESHOLD_MIN + 0.02)
        elif abs_ncc <= st.bit_ncc_threshold:
            st.bit_ncc_low_ema = _ema(st.bit_ncc_low_ema, abs_ncc, config.BIT_NCC_NOISE_ALPHA)
        elif locked:
            # Reject impulsive outliers that would pin threshold at max and erase real "1" chips.
            if abs_ncc <= st.bit_ncc_low_ema + _HIGH_SPIKE_REJECT_HEADROOM:
                capped = min(abs_ncc, st.bit_ncc_low_ema + _LOCKED_THRESHOLD_HEADROOM)
                st.bit_ncc_high_ema = _ema(
                    st.bit_ncc_high_ema, capped, max(0.5 * config.BIT_NCC_NOISE_ALPHA, 0.02))
            else:
                st.bit_ncc_high_ema = _ema(
                    st.bit_ncc_high_ema, st.bit_ncc_low_ema + _UNLOCK_HIGH_FLOOR_MARGIN, _HIGH_RELAX_ALPHA)
        else:
            st.bit_ncc_high_ema = _ema(
                st.bit_ncc_high_ema, st.bit_ncc_low_ema + _UNLOCK_HIGH_FLOOR_MARGIN, 0.03)

        if st.bit_ncc_high_ema < st.bit_ncc_low_ema + 0.02:
            st.bit_ncc_high_ema = st.bit_ncc_low_ema + 0.02
        st.bit_ncc_noise_ema = st.bit_ncc_low_ema

        if not config.ADAPTIVE_BIT_NCC_THRESHOLD:
            st.bit_ncc_threshold = float(config.BIT_NCC_THRESHOLD)
            return
        blend = 0.45 if locked else _UNLOCK_BLEND
        cand = st.bit_ncc_low_ema + blend * (st.bit_ncc_high_ema - st.bit_ncc_low_ema)
        if locked:
            cand = min(cand, st.bit_ncc_low_ema + _LOCKED_THRESHOLD_HEADROOM)
        else:
            cand = min(cand, st.bit_ncc_low_ema + _UNLOCK_THRESHOLD_HEADROOM)
        st.bit_ncc_threshold = float(np.clip(
            cand, config.BIT_NCC_THRESHOLD_MIN, config.BIT_NCC_THRESHOLD_MAX))

    # ---- chip slicing + decode --------------------------------------

    def _decide_chip_bistable(self, st: PhaseState, abs_ncc: float) -> int:
        thr = st.bit_ncc_threshold
        if not config.CHIP_BISTABLE_ENABLE:
            return 1 if abs_ncc >= thr else 0

        hi = min(config.BIT_NCC_THRESHOLD_MAX, thr + config.CHIP_DECISION_HYSTERESIS)
        lo = max(config.BIT_NCC_THRESHOLD_MIN, thr - config.CHIP_DECISION_HYSTERESIS)

        if st.chip_state < 0:
            st.chip_state = 1 if abs_ncc >= thr else 0
            st.chip_rise_streak = 0
            st.chip_fall_streak = 0
            return st.chip_state

        rise_confirm = config.CHIP_RISE_CONFIRM_CHIPS
        fall_confirm = config.CHIP_FALL_CONFIRM_CHIPS

        if st.chips:
            w = min(len(st.chips), config.CHIP_STICKY_WINDOW_CHIPS)
            if w > 0:
                one_ratio = float(sum(st.chips[-w:])) / float(w)
                if one_ratio >= config.CHIP_STICKY_ONE_RATIO:
                    fall_confirm = max(fall_confirm, config.CHIP_STICKY_FALL_CONFIRM_CHIPS)
        if config.CONTINUOUS_ON_TEST:
            fall_confirm = max(fall_confirm, config.CONTINUOUS_ON_FALL_CONFIRM_CHIPS)

        if st.chip_state == 1:
            if abs_ncc <= lo:
                st.chip_fall_streak += 1
            else:
                st.chip_fall_streak = 0
            if st.chip_fall_streak >= fall_confirm:
                st.chip_state = 0
                st.chip_fall_streak = 0
                st.chip_rise_streak = 0
        else:
            if abs_ncc >= hi:
                st.chip_rise_streak += 1
            else:
                st.chip_rise_streak = 0
            if st.chip_rise_streak >= rise_confirm:
                st.chip_state = 1
                st.chip_rise_streak = 0
                st.chip_fall_streak = 0
        return st.chip_state

    def _slice_chips_for_phase(self, phase: int, st: PhaseState, peak_hz: float) -> int:
        added = 0
        metric_hist = self.env_chip_metric_history_by_phase[phase]
        decision_hist = self.env_chip_decision_history_by_phase[phase]
        while True:
            chip_len = max(1, int(round(st.chip_stride_samples)))
            if st.next_sample + chip_len > len(self.phase_sample_buffer):
                break
            start = st.next_sample
            center_chunk = self.phase_sample_buffer[start : start + chip_len]
            center_metric = coherent_chip_metric(
                center_chunk, peak_hz, config.SUBCARRIER_HZ, config.SAMPLE_RATE)
            abs_ncc = center_metric
            step_advance = chip_len

            if config.CHIP_TIMING_TRACK_ENABLE:
                best_metric = center_metric
                best_shift = 0
                search = int(max(0, config.CHIP_TIMING_SEARCH_SAMPLES))
                step = int(max(1, config.CHIP_TIMING_STEP_SAMPLES))
                for shift in range(-search, search + 1, step):
                    if shift == 0:
                        continue
                    cand_start = start + shift
                    if cand_start < 0 or cand_start + chip_len > len(self.phase_sample_buffer):
                        continue
                    cand_chunk = self.phase_sample_buffer[cand_start : cand_start + chip_len]
                    cand_metric = coherent_chip_metric(
                        cand_chunk, peak_hz, config.SUBCARRIER_HZ, config.SAMPLE_RATE)
                    if cand_metric > best_metric:
                        best_metric = cand_metric
                        best_shift = shift

                # Apply timing correction only when this chip likely carries tone energy
                # and the early/late gain is meaningful; avoid random walk on OFF chips.
                if (best_shift != 0
                        and best_metric >= st.bit_ncc_threshold + config.CHIP_TIMING_TRACK_MARGIN
                        and (best_metric - center_metric) >= config.CHIP_TIMING_MIN_ADVANTAGE):
                    abs_ncc = best_metric
                    st.timing_error_ema = _ema(
                        st.timing_error_ema, float(best_shift), config.CHIP_TIMING_DRIFT_ALPHA)
                else:
                    st.timing_error_ema = _ema(st.timing_error_ema, 0.0, 0.05)

                min_stride = self.bit_samples * config.CHIP_TIMING_STRIDE_MIN_SCALE
                max_stride = self.bit_samples * config.CHIP_TIMING_STRIDE_MAX_SCALE
                st.chip_stride_samples = float(np.clip(
                    self.bit_samples + st.timing_error_ema, min_stride, max_stride))
                step_advance = max(1, int(round(st.chip_stride_samples)))

            self._update_phase_threshold(st, abs_ncc, self.ncc_lock)
            chip = self._decide_chip_bistable(st, abs_ncc)
            st.chips.append(chip)
            st.next_sample += step_advance
            st.chips_seen += 1
            added += 1
            metric_hist = np.roll(metric_hist, -1); metric_hist[-1] = float(np.clip(abs_ncc, 0.0, 1.0))
            decision_hist = np.roll(decision_hist, -1); decision_hist[-1] = float(chip)
        self.env_chip_metric_history_by_phase[phase] = metric_hist
        self.env_chip_decision_history_by_phase[phase] = decision_hist

        max_chips = config.PHASE_HISTORY_BITS * config.REPETITION_CHIPS
        if len(st.chips) > max_chips:
            drop = len(st.chips) - max_chips
            del st.chips[:drop]
            st.base_chip_index += drop
        return added

    def _try_decode_packets(self, phase: int, st: PhaseState, phase_score: float) -> list[PacketCandidate]:
        candidates: list[PacketCandidate] = []
        for off in range(config.REPETITION_CHIPS):
            decoded = majority_decode_triplets(st.chips, off)
            base_bit = (st.base_chip_index + off) // config.REPETITION_CHIPS
            while True:
                idx, errs = find_header_match(
                    decoded, self.packet_header_bits,
                    st.search_start_by_offset[off], config.HEADER_MAX_BIT_ERRORS)
                if idx < 0:
                    st.search_start_by_offset[off] = max(
                        0, len(decoded) - len(self.packet_header_bits) + 1)
                    break
                header_abs = base_bit + idx
                if header_abs <= st.last_header_abs_by_offset[off]:
                    st.search_start_by_offset[off] = idx + 1
                    continue

                preamble = decoded[idx : idx + len(self.preamble_bits)]
                transitions = sum(1 for a, b in zip(preamble, preamble[1:]) if a != b)
                if transitions < config.PREAMBLE_MIN_TRANSITIONS:
                    st.search_start_by_offset[off] = idx + 1
                    continue

                sync_start = idx + len(self.preamble_bits)
                sync_bits = decoded[sync_start : sync_start + len(self.packet_header_bits) - len(self.preamble_bits)]

                p_start = idx + len(self.packet_header_bits)
                p_end = p_start + self.payload_bits_len
                if p_end > len(decoded):
                    st.search_start_by_offset[off] = idx
                    break
                payload_bits = decoded[p_start:p_end]
                payload = bits_to_bytes(payload_bits)
                payload_errors = sum(
                    1 for left, right in zip(payload_bits, self.payload_bits) if left != right)
                candidates.append(PacketCandidate(
                    phase=phase,
                    chip_offset=off,
                    header_abs=header_abs,
                    header_errors=errs,
                    preamble_transitions=transitions,
                    payload_errors=payload_errors,
                    payload=payload,
                    phase_score=phase_score,
                    observed_preamble=preamble.copy(),
                    observed_sync=sync_bits.copy(),
                ))
                st.search_start_by_offset[off] = idx + 1
        return candidates

    def _accept_packet_candidate(self, candidate: PacketCandidate) -> None:
        self.decoded_packets += 1
        self.last_packet_accept_bit = candidate.header_abs
        self.pending_packet_candidate = None
        self.packet_status_text = (
            f"Packet: accepted at phase {candidate.phase}, offset {candidate.chip_offset}, "
            f"payload_errors={candidate.payload_errors}")
        print(f"[RX HEADER] phase={candidate.phase} chip_offset={candidate.chip_offset} "
              f"bit={candidate.header_abs} header_errors={candidate.header_errors} "
              f"preamble_transitions={candidate.preamble_transitions} "
              f"preamble_seen={bits_to_text(candidate.observed_preamble)} "
              f"preamble_expected={bits_to_text(self.preamble_bits)} "
              f"sync_seen={bits_to_text(candidate.observed_sync)} "
              f"sync_expected={bits_to_text(self.packet_header_bits[len(self.preamble_bits):])} "
              f"payload_errors={candidate.payload_errors} payload_hex={candidate.payload.hex().upper()} "
              f"payload_ascii={safe_ascii(candidate.payload)!r}", flush=True)
        print(f"[RX PACKET {self.decoded_packets}] phase={candidate.phase} chip_offset={candidate.chip_offset} "
              f"bit={candidate.header_abs} header_errors={candidate.header_errors} "
              f"payload_errors={candidate.payload_errors} payload_hex={candidate.payload.hex().upper()}",
              flush=True)
        self.packet_status_hold = config.PACKET_STATUS_HOLD_FRAMES
        st = self.phase_state[candidate.phase]
        st.last_header_abs_by_offset[candidate.chip_offset] = candidate.header_abs

    def _current_bit_horizon(self) -> int:
        return max(
            (st.base_chip_index + len(st.chips)) // config.REPETITION_CHIPS
            for st in self.phase_state.values()
        )

    def _accept_best_packet_candidate(self, candidates: list[PacketCandidate]) -> None:
        current_bit_horizon = self._current_bit_horizon()
        viable = [
            cand for cand in candidates
            if cand.payload_errors <= config.PAYLOAD_MAX_BIT_ERRORS
            and cand.header_abs > self.last_packet_accept_bit + config.PACKET_MERGE_GAP_BITS
        ]
        if viable:
            viable.sort(key=_packet_candidate_sort_key)
            best_new = viable[0]
            if self.pending_packet_candidate is None:
                self.pending_packet_candidate = best_new
            elif best_new.header_abs <= (
                self.pending_packet_candidate.header_abs + config.PACKET_CANDIDATE_SETTLE_BITS
            ):
                if _packet_candidate_sort_key(best_new) < _packet_candidate_sort_key(self.pending_packet_candidate):
                    self.pending_packet_candidate = best_new
            else:
                self._accept_packet_candidate(self.pending_packet_candidate)
                self.pending_packet_candidate = best_new

        if self.pending_packet_candidate is None:
            return
        if current_bit_horizon >= (
            self.pending_packet_candidate.header_abs + config.PACKET_CANDIDATE_SETTLE_BITS
        ):
            self._accept_packet_candidate(self.pending_packet_candidate)

    # ---- chip view + debug ------------------------------------------

    def _ranked_phases(self) -> list[tuple[int, float]]:
        n = min(16, self.nw.env_plot_len)
        scores = [(p, float(np.mean(self.env_chip_metric_history_by_phase[p][-n:])))
                  for p in self.phase_offsets]
        scores.sort(key=lambda it: it[1], reverse=True)
        return scores

    def _emit_debug(self, best_phase: int) -> None:
        st = self.phase_state[best_phase]
        # Try all chip offsets; pick the one with the most 0↔1 transitions for display.
        best_decoded = majority_decode_triplets(st.chips, 0)
        for _off in range(1, config.REPETITION_CHIPS):
            cand = majority_decode_triplets(st.chips, _off)
            if sum(a != b for a, b in zip(cand, cand[1:])) > sum(a != b for a, b in zip(best_decoded, best_decoded[1:])):
                best_decoded = cand
        decoded = best_decoded
        chip_tail = bits_to_text(st.chips[-config.TERMINAL_DEBUG_BIT_TAIL:])
        bit_tail = bits_to_text(decoded[-(config.TERMINAL_DEBUG_BIT_TAIL // config.REPETITION_CHIPS):])
        print(f"[RX DEBUG] phase={best_phase} chips={st.chips_seen} "
              f"ncc_ema={self.ncc_abs_ema:.3f} lock={int(self.ncc_lock)} "
              f"bit_thr={st.bit_ncc_threshold:.3f} noise_ema={st.bit_ncc_noise_ema:.3f} "
              f"chip_tail={chip_tail} bit_tail={bit_tail}", flush=True)

    def _update_chip_view(self, ranked: list[tuple[int, float]]) -> None:
        self.display_phase = ranked[0][0]
        self.phase_score_summary = " ".join(f"p{p}:{s:.2f}" for p, s in ranked[:3])
        st = self.phase_state[self.display_phase]
        self.nw.line_env.set_ydata(self.env_chip_metric_history_by_phase[self.display_phase])
        self.nw.line_env_bits.set_ydata(0.9 * self.env_chip_decision_history_by_phase[self.display_phase])
        self.nw.env_threshold_line.set_ydata([st.bit_ncc_threshold])

    # ---- per-frame --------------------------------------------------

    def update(self, _frame: int) -> tuple:
        bb, cw, nw = self.bb, self.cw, self.nw
        artists = (bb.line_i, bb.line_q, bb.line_fft_raw, bb.line_fft_dc_blocked,
                   bb.exciter_marker, cw.line_centered, cw.waterfall_img, bb.status, cw.status)
        if self.stop_requested:
            return artists

        x_raw = normalize_iq(self.sdr.rx())
        x_dc, self.dc_prev_x, self.dc_prev_y = dc_block_filter(
            x_raw, self.dc_prev_x, self.dc_prev_y, config.DC_BLOCK_ALPHA)

        shown = x_raw[: config.TIME_SAMPLES]
        bb.line_i.set_ydata(np.real(shown))
        bb.line_q.set_ydata(np.imag(shown))

        freqs_hz, raw_dbfs = compute_spectrum_dbfs(x_raw, config.SAMPLE_RATE)
        _, dc_dbfs_full = compute_spectrum_dbfs(x_dc, config.SAMPLE_RATE)
        self.smoothed_raw = ema_spectrum_power_domain(raw_dbfs, self.smoothed_raw, config.FFT_AVG_ALPHA)
        self.smoothed_dc = ema_spectrum_power_domain(dc_dbfs_full, self.smoothed_dc, config.FFT_AVG_ALPHA)

        in_view = np.abs(freqs_hz) <= config.SPECTRUM_SPAN_HZ
        rms = float(np.sqrt(np.mean(np.abs(x_raw) ** 2)))

        peak_hz, peak_dbfs = find_exciter_peak(
            freqs_hz, self.smoothed_dc, in_view, self.prev_peak_hz,
            config.EXCITER_SEARCH_MIN_HZ, config.EXCITER_SEARCH_MAX_HZ)
        self.prev_peak_hz = peak_hz
        bb.exciter_marker.set_xdata([peak_hz / 1000.0, peak_hz / 1000.0])

        freqs_khz = freqs_hz[in_view] / 1000.0
        bb.line_fft_raw.set_data(freqs_khz, self.smoothed_raw[in_view])
        bb.line_fft_dc_blocked.set_data(freqs_khz, self.smoothed_dc[in_view])

        dc_idx = int(np.argmin(np.abs(freqs_hz)))
        dc_level = float(self.smoothed_raw[dc_idx])

        centered_freqs_hz = freqs_hz - peak_hz
        cm = np.abs(centered_freqs_hz) <= config.CENTERED_SPAN_HZ
        centered_freqs_khz = centered_freqs_hz[cm] / 1000.0
        centered_spec = smooth_1d(self.smoothed_dc[cm], config.CENTERED_FREQ_SMOOTH_BINS)
        cw.line_centered.set_data(centered_freqs_khz, centered_spec)

        off_carrier = np.abs(centered_freqs_khz) > 0.5
        if off_carrier.any():
            noise_y = float(np.percentile(centered_spec[off_carrier], 20))
            cw.ax_centered.set_ylim(noise_y - 6.0, peak_dbfs + 4.0)
            self.prev_peak_dbfs = peak_dbfs

        snr_db, sb_pos, sb_neg, noise_floor_sb = compute_sideband_snr(
            centered_freqs_khz, centered_spec,
            config.SIDEBAND_OFFSET_KHZ, config.SIDEBAND_WINDOW_HZ)
        cw.sideband_scatter.set_offsets(np.array(
            [[-config.SIDEBAND_OFFSET_KHZ, sb_neg], [config.SIDEBAND_OFFSET_KHZ, sb_pos]]))
        dot_color = "#44ff88" if snr_db >= config.SNR_LOCK_THRESHOLD_DB else "#ff4444"
        cw.sideband_scatter.set_facecolor([dot_color, dot_color])
        cw.snr_threshold_line.set_ydata([noise_floor_sb + config.SNR_LOCK_THRESHOLD_DB])

        if centered_spec.size:
            remapped = remap_and_compress_centered(cw.centered_axis, centered_freqs_khz, centered_spec)
            cw.waterfall_data[:-1] = cw.waterfall_data[1:]
            if config.WATERFALL_ROWS > 1:
                remapped = (config.WATERFALL_ROW_BLEND * remapped
                            + (1.0 - config.WATERFALL_ROW_BLEND) * cw.waterfall_data[-2])
            cw.waterfall_data[-1] = remapped
            cw.waterfall_img.set_data(cw.waterfall_data)
            wf_mask = np.abs(cw.centered_axis) > 0.5
            valid = cw.waterfall_data[:, wf_mask]
            valid = valid[valid > -139.0]
            if valid.size:
                nf = float(np.percentile(valid, 15))
                cw.waterfall_img.set_clim(vmin=nf - 1.0, vmax=nf + config.WATERFALL_DYN_RANGE_DB)

        bb.status.set_text(
            f"RX={config.RX_URI} | LO={config.FREQ_HZ/1e9:.3f} GHz | RMS={rms:.4f} FS | "
            f"Exciter={peak_hz:,.1f} Hz @ {peak_dbfs:.1f} dBFS | DC={dc_level:.1f} dBFS | "
            f"Carrier-DC={peak_dbfs - dc_level:.1f} dB")
        cw.status.set_text(
            f"Carrier={peak_hz:,.1f} Hz | Span=±{config.CENTERED_SPAN_HZ/1000.0:.0f} kHz | "
            f"SB SNR={snr_db:+.1f} dB (+{config.SIDEBAND_OFFSET_KHZ:.0f}kHz={sb_pos:.1f} "
            f"-{config.SIDEBAND_OFFSET_KHZ:.0f}kHz={sb_neg:.1f} Noise={noise_floor_sb:.1f} dBFS)")
        cw.fig.canvas.draw_idle()

        if peak_hz != 0.0:
            self._process_demod(x_dc, peak_hz)
        return artists

    def _process_demod(self, x_dc: np.ndarray, peak_hz: float) -> None:
        nw = self.nw
        demod = bandpass_filter_around_carrier(
            x_dc, peak_hz, config.SAMPLE_RATE, config.DEMOD_FILTER_PASSBAND_HZ)
        ncc_val, _ = coherent_ncc(demod, peak_hz, config.SUBCARRIER_HZ, config.SAMPLE_RATE)
        nw.ncc_history = np.roll(nw.ncc_history, -1); nw.ncc_history[-1] = ncc_val
        nw.line_ncc.set_ydata(nw.ncc_history)
        self._update_lock_hysteresis(abs(ncc_val))

        self.phase_sample_buffer = np.concatenate((self.phase_sample_buffer, demod))
        chips_added = sum(self._slice_chips_for_phase(p, st, peak_hz)
                          for p, st in self.phase_state.items())

        if self.ncc_lock:
            self.decode_lock_grace_chips = config.DECODE_UNLOCK_GRACE_CHIPS
        elif chips_added > 0:
            self.decode_lock_grace_chips = max(0, self.decode_lock_grace_chips - chips_added)

        if (config.PACKET_DECODE_ENABLED
            and (self.ncc_lock or self.decode_lock_grace_chips > 0
                 or self.ncc_abs_ema >= config.NCC_SOFT_DECODE_THRESHOLD)):
            ranked = self._ranked_phases()
            decode_phases = ranked[:max(1, config.PACKET_DECODE_TOP_PHASES)]
            packet_candidates: list[PacketCandidate] = []
            for p, score in decode_phases:
                packet_candidates.extend(self._try_decode_packets(p, self.phase_state[p], score))
            self._accept_best_packet_candidate(packet_candidates)

        if chips_added > 0:
            ranked = self._ranked_phases()
            self._update_chip_view(ranked)
            self.total_bits += chips_added
            while self.total_bits >= self.next_debug_bits:
                self._emit_debug(ranked[0][0])
                self.next_debug_bits += config.TERMINAL_DEBUG_BITS_EVERY
            min_next = min(s.next_sample for s in self.phase_state.values())
            if min_next > self.bit_samples:
                trim = min_next - self.bit_samples
                self.phase_sample_buffer = self.phase_sample_buffer[trim:]
                for s in self.phase_state.values():
                    s.next_sample -= trim

        if self.packet_status_hold > 0:
            self.packet_status_hold -= 1
        else:
            self.packet_status_text = (
                "Continuous ON test: packet decode disabled"
                if not config.PACKET_DECODE_ENABLED
                else "Waiting for preamble+sync"
            )

        view_st = self.phase_state[self.display_phase]
        nw.ncc_status.set_text(
            f"NCC={ncc_val:+.3f} | EMA={self.ncc_abs_ema:.3f} | "
            f"{'LOCKED' if self.ncc_lock else 'searching'} | "
            f"Carrier={peak_hz:,.0f} Hz | chip_view_phase={self.display_phase} "
            f"thr={view_st.bit_ncc_threshold:.3f} | {self.phase_score_summary} | "
            f"{self.packet_status_text}")
        nw.fig.canvas.draw_idle()
