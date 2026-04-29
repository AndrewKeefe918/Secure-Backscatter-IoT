"""RX-only lightweight loop for ingest/slicing/capture.

This module intentionally avoids packet decoding and plotting so the realtime
path stays minimal. Heavy packet search runs offline on the captured chip stream.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

import numpy as np

from . import config as config
from .dsp import (
    coherent_fsk_metrics_cached,
    compute_spectrum_dbfs,
    compute_sideband_snr,
    dc_block_filter,
    derotate_frequency,
    ema_spectrum_power_domain,
    estimate_residual_cfo_hz,
    find_exciter_peak,
    normalize_iq,
    remap_and_compress_centered,
    smooth_1d,
)
from .live_decode import analyze_live_decode
from .packet_decoder import bits_to_text
from .secure_packet import DecodedPacket
from .secure_packet import SecureReceiver


@dataclass
class PhaseState:
    next_sample: int
    chips: list[int] = field(default_factory=list)
    base_chip_index: int = 0
    chips_seen: int = 0


class ReceiverLoopRxOnly:
    """Minimal realtime loop: capture samples, slice chips, emit telemetry, persist chips."""

    def __init__(self, sdr: object) -> None:
        self.sdr = sdr
        self.stop_requested = False

        self.bit_samples = config.SAMPLES_PER_CHIP
        phase_step = self.bit_samples // max(1, int(config.PHASE_COUNT))
        self.phase_offsets = [i * phase_step for i in range(int(config.PHASE_COUNT))]
        self.phase_sample_buffer = np.array([], dtype=np.complex64)
        self.phase_state = {p: PhaseState(next_sample=p) for p in self.phase_offsets}

        # Cached complex64 reference vectors for the chip metric. The two
        # subcarrier references depend only on bit_samples and SAMPLE_RATE,
        # so they are built once. The mix-to-DC vector depends on the
        # tracked peak; rebuild only when peak_hz crosses a coarse grid.
        n_chip = np.arange(self.bit_samples, dtype=np.float64)
        two_pi_over_fs = 2.0 * np.pi / float(config.SAMPLE_RATE)
        self._ref_f1_c64 = np.exp(
            -1j * two_pi_over_fs * float(config.FSK_F1_HZ) * n_chip
        ).astype(np.complex64)
        self._ref_f0_c64 = np.exp(
            -1j * two_pi_over_fs * float(config.FSK_F0_HZ) * n_chip
        ).astype(np.complex64)
        self._chip_mix_c64: np.ndarray | None = None
        self._chip_mix_peak_hz: float | None = None

        self.smoothed_raw = np.array([], dtype=np.float64)
        self.smoothed_dc = np.array([], dtype=np.float64)
        self.dc_prev_x = np.complex64(0.0 + 0.0j)
        self.dc_prev_y = np.complex64(0.0 + 0.0j)
        self.prev_peak_hz: float | None = None
        self.monitor_axis_khz = np.linspace(
            -config.CENTERED_SPAN_HZ / 1000.0,
            config.CENTERED_SPAN_HZ / 1000.0,
            int(config.RX_MONITOR_SPECTRUM_BINS),
            dtype=np.float64,
        )
        self.monitor_raw_row = np.full_like(self.monitor_axis_khz, -140.0)
        self.monitor_centered_row = np.full_like(self.monitor_axis_khz, -140.0)
        self.monitor_noise_floor_dbfs = -140.0
        self.monitor_sideband_snr_db = 0.0
        self.monitor_sideband_pos_dbfs = -140.0
        self.monitor_sideband_neg_dbfs = -140.0

        self.cfo_coarse_hz = 0.0
        self.cfo_fine_hz = 0.0
        self.cfo_total_hz = 0.0
        self.cfo_phase_rad = 0.0

        self.ncc_lock = False
        self._lock_enter_count = 0
        self._lock_exit_count = 0
        self.last_decode_summary = "No live decode yet"
        self.last_logic_tail = ""

        self.frame_index = 0
        self.last_frame_start_s: float | None = None
        self.buffer_duration_s = float(config.RX_BUFFER_SIZE) / float(config.SAMPLE_RATE)
        self.update_interval_s = float(config.ANIMATION_INTERVAL_MS) / 1000.0
        self.realtime_budget_s = max(self.buffer_duration_s, self.update_interval_s)
        self.rx_time_ema_s = 0.0
        self.process_time_ema_s = 0.0
        self.frame_gap_ema_s = 0.0
        self.late_frame_count = 0
        self.gap_slip_count = 0

        capture_path = Path(config.RX_CAPTURE_NDJSON)
        capture_path.parent.mkdir(parents=True, exist_ok=True)
        self.capture_file = capture_path.open("a", encoding="ascii")
        self.status_path = Path(config.RX_STATUS_JSON)
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        self.secure_rx = SecureReceiver(
            bytes.fromhex(config.SHARED_KEY_HEX),
            Path(config.SECURE_RX_STATE_PATH),
        )
        self._packet_cache: dict[bytes, DecodedPacket] = {}
        self.decoded_packets = 0
        self.decode_status_hold = 0
        self.last_reject_signature = ""
        self.last_displayed_auth_counter: int | None = None

    @property
    def _dc_bin_guard_hz(self) -> float:
        return float(config.SAMPLE_RATE) / float(config.RX_BUFFER_SIZE)

    def close(self) -> None:
        try:
            self.capture_file.close()
        except Exception:
            pass

    def _update_lock_hysteresis(self) -> None:
        """Pure SNR-driven lock hysteresis.

        Enter requires SNR_LOCK_THRESHOLD_DB sustained for LOCK_ENTER_FRAMES;
        exit requires SNR below threshold - SNR_LOCK_EXIT_MARGIN_DB sustained
        for LOCK_EXIT_FRAMES.
        """
        snr = float(self.monitor_sideband_snr_db)
        snr_enter_ok = snr >= float(config.SNR_LOCK_THRESHOLD_DB)
        snr_exit_ok = snr < (
            float(config.SNR_LOCK_THRESHOLD_DB) - float(config.SNR_LOCK_EXIT_MARGIN_DB)
        )
        if not self.ncc_lock:
            self._lock_enter_count = self._lock_enter_count + 1 if snr_enter_ok else 0
            if self._lock_enter_count >= int(config.LOCK_ENTER_FRAMES):
                self.ncc_lock = True
                self._lock_exit_count = 0
        else:
            self._lock_exit_count = self._lock_exit_count + 1 if snr_exit_ok else 0
            if self._lock_exit_count >= int(config.LOCK_EXIT_FRAMES):
                self.ncc_lock = False
                self._lock_enter_count = 0

    def _slice_chips_for_phase(self, state: PhaseState, peak_hz: float) -> int:
        added = 0
        # Rebuild the carrier-to-DC mix only when the tracked peak moves
        # by more than one FFT bin; otherwise reuse the cached complex64
        # vector across all chips and phases in this frame.
        peak_bin_hz = float(config.SAMPLE_RATE) / float(config.RX_BUFFER_SIZE)
        if (
            self._chip_mix_c64 is None
            or self._chip_mix_peak_hz is None
            or abs(float(peak_hz) - float(self._chip_mix_peak_hz)) > peak_bin_hz
        ):
            n_chip = np.arange(self.bit_samples, dtype=np.float64)
            self._chip_mix_c64 = np.exp(
                -1j * 2.0 * np.pi * float(peak_hz) * n_chip / float(config.SAMPLE_RATE)
            ).astype(np.complex64)
            self._chip_mix_peak_hz = float(peak_hz)
        mix_c64 = self._chip_mix_c64
        ref_f1 = self._ref_f1_c64
        ref_f0 = self._ref_f0_c64
        while state.next_sample + self.bit_samples <= len(self.phase_sample_buffer):
            chunk = self.phase_sample_buffer[state.next_sample : state.next_sample + self.bit_samples]
            m_f1, m_f0, decision = coherent_fsk_metrics_cached(chunk, mix_c64, ref_f1, ref_f0)
            both_below_floor = max(m_f1, m_f0) < config.FSK_PRESENCE_FLOOR
            in_dead_zone = abs(decision) < config.FSK_DECISION_DEAD_ZONE
            if both_below_floor or in_dead_zone:
                chip_value = 0
            else:
                chip_value = 1 if decision > 0 else 0

            state.chips.append(chip_value)
            state.next_sample += self.bit_samples
            state.chips_seen += 1
            added += 1

        max_chips = config.PHASE_HISTORY_BITS * config.REPETITION_CHIPS
        if len(state.chips) > max_chips:
            drop = len(state.chips) - max_chips
            del state.chips[:drop]
            state.base_chip_index += drop
        return added

    def _capture_phase_update(self, phase: int, state: PhaseState, added: int) -> None:
        if added <= 0:
            return
        start_abs = state.base_chip_index + (len(state.chips) - added)
        rec = {
            "frame": self.frame_index,
            "phase": phase,
            "start_chip": start_abs,
            "chips": state.chips[-added:],
        }
        self.capture_file.write(json.dumps(rec, separators=(",", ":")) + "\n")

    def _update_live_decode_summary(self) -> None:
        chips_by_phase = {phase: state.chips for phase, state in self.phase_state.items()}
        analysis = analyze_live_decode(
            chips_by_phase,
            tail_bits=48,
            include_matches=False,
            secure_rx=self.secure_rx,
            verified_cache=self._packet_cache,
        )

        is_auth = "AUTHENTICATED" in analysis.best_text
        is_reject = "REJECTED" in analysis.best_text
        is_cached = "[cached]" in analysis.best_text
        last_is_auth = "AUTHENTICATED" in self.last_decode_summary

        reject_signature = ""
        if is_reject and "payload=" in analysis.best_text:
            payload_token = analysis.best_text.split("payload=", 1)[1].split(" ", 1)[0]
            reason_token = analysis.best_text.split(": ", 1)[1] if ": " in analysis.best_text else ""
            reject_signature = f"{payload_token}|{reason_token}"
        same_cached_reject = bool(
            is_reject
            and is_cached
            and reject_signature
            and reject_signature == self.last_reject_signature
        )

        # Keep authenticated summaries stable for a short hold window.
        # Reject reasons are still shown, but they don't immediately displace a
        # fresh authenticated packet in the very next status frame.
        auth_hold_frames = max(
            int(config.PACKET_STATUS_HOLD_FRAMES),
            int(config.RX_TERMINAL_STATUS_EVERY_FRAMES) + 1,
        )
        reject_hold_frames = int(config.RX_TERMINAL_STATUS_EVERY_FRAMES) + 1

        if is_auth:
            self.last_decode_summary = analysis.best_text
            self.decode_status_hold = auth_hold_frames
            self.last_reject_signature = ""
        elif is_reject and self.decode_status_hold > 0 and last_is_auth:
            self.decode_status_hold -= 1
        elif same_cached_reject:
            # Suppress noisy repeats of the exact same cached reject candidate.
            self.last_decode_summary = "No live decode yet"
            self.decode_status_hold = 0
        elif analysis.best_text != "No live decode yet":
            self.last_decode_summary = analysis.best_text
            self.decode_status_hold = reject_hold_frames if is_reject else auth_hold_frames
            if is_reject:
                self.last_reject_signature = reject_signature
            else:
                self.last_reject_signature = ""
        elif self.decode_status_hold > 0:
            self.decode_status_hold -= 1
        else:
            self.last_decode_summary = analysis.best_text

        self.last_logic_tail = bits_to_text(analysis.best_tail_bits) if analysis.best_tail_bits else ""

        # Keep cache bounded for long-running captures.
        max_cache = 2048
        if len(self._packet_cache) > max_cache:
            drop = len(self._packet_cache) - max_cache
            for _ in range(drop):
                self._packet_cache.pop(next(iter(self._packet_cache)))

    def _emit_status(self, peak_hz: float | None) -> None:
        if self.frame_index % max(1, int(config.RX_TERMINAL_STATUS_EVERY_FRAMES)) != 0:
            return
        best_phase = max(self.phase_offsets, key=lambda p: self.phase_state[p].chips_seen)
        peak_label = 0.0 if peak_hz is None else float(peak_hz)

        # Suppress repeat status lines for an AUTHENTICATED packet whose
        # counter has already been displayed since acceptance. The first time
        # each new counter actually prints, we let it through and remember it;
        # subsequent prints with the same counter are quieted to "No live
        # decode yet". Tracking happens at print time (not at live-decode
        # update time) so the first display is never lost when live decode
        # fires multiple times between status frames.
        decode_text = self.last_decode_summary
        if (
            bool(config.AUTHENTICATED_STATUS_SUPPRESS_REPEATS)
            and "AUTHENTICATED" in decode_text
            and "counter=" in decode_text
        ):
            shown_counter: int | None
            try:
                shown_counter = int(
                    decode_text.split("counter=", 1)[1].split(" ", 1)[0]
                )
            except (ValueError, IndexError):
                shown_counter = None
            if shown_counter is not None:
                if shown_counter == self.last_displayed_auth_counter:
                    decode_text = "No live decode yet"
                else:
                    self.last_displayed_auth_counter = shown_counter

        print(
            f"[RX STATUS] frame={self.frame_index} lock={'1' if self.ncc_lock else '0'} "
            f"phase={best_phase} peak={peak_label:+.1f}Hz cfo={self.cfo_total_hz:+.1f}Hz "
            f"snr={self.monitor_sideband_snr_db:+.1f}dB "
            f"proc={1000.0*self.process_time_ema_s:.1f}ms "
            f"late={self.late_frame_count} slip={self.gap_slip_count} "
            f"bits={self.last_logic_tail or '-'} "
            f"decode={decode_text}",
            flush=True,
        )

    def _update_monitor_spectrum(self, freqs_hz: np.ndarray, peak_hz: float | None) -> None:
        if self.smoothed_raw.size:
            raw_mask = np.abs(freqs_hz) <= config.CENTERED_SPAN_HZ
            if np.any(raw_mask):
                raw_freqs_khz = freqs_hz[raw_mask] / 1000.0
                raw_spec = smooth_1d(self.smoothed_raw[raw_mask], config.CENTERED_FREQ_SMOOTH_BINS)
                self.monitor_raw_row = np.interp(
                    self.monitor_axis_khz,
                    raw_freqs_khz,
                    raw_spec,
                    left=-140.0,
                    right=-140.0,
                ).astype(np.float64, copy=False)

        if peak_hz is None or self.smoothed_dc.size == 0:
            return

        centered_freqs_hz = freqs_hz - peak_hz
        centered_mask = np.abs(centered_freqs_hz) <= config.CENTERED_SPAN_HZ
        if not np.any(centered_mask):
            return

        centered_freqs_khz = centered_freqs_hz[centered_mask] / 1000.0
        centered_spec = smooth_1d(self.smoothed_dc[centered_mask], config.CENTERED_FREQ_SMOOTH_BINS)
        if centered_spec.size == 0:
            return

        remapped = remap_and_compress_centered(self.monitor_axis_khz, centered_freqs_khz, centered_spec)
        self.monitor_centered_row = remapped.astype(np.float64, copy=False)

        off_carrier = np.abs(self.monitor_axis_khz) > 0.5
        if np.any(off_carrier):
            self.monitor_noise_floor_dbfs = float(np.percentile(self.monitor_centered_row[off_carrier], 20))

        snr_db, sb_pos, sb_neg, _ = compute_sideband_snr(
            centered_freqs_khz,
            centered_spec,
            config.SIDEBAND_OFFSET_KHZ,
            config.SIDEBAND_WINDOW_HZ,
        )
        self.monitor_sideband_snr_db = float(snr_db)
        self.monitor_sideband_pos_dbfs = float(sb_pos)
        self.monitor_sideband_neg_dbfs = float(sb_neg)

    def _write_status_snapshot(self, peak_hz: float | None) -> None:
        if self.frame_index % max(1, int(config.RX_STATUS_EVERY_FRAMES)) != 0:
            return
        best_phase = max(self.phase_offsets, key=lambda p: self.phase_state[p].chips_seen)
        st = self.phase_state[best_phase]
        peak_value = 0.0 if peak_hz is None else float(peak_hz)
        payload = {
            "frame": self.frame_index,
            "lock": 1 if self.ncc_lock else 0,
            "cfo_hz": round(self.cfo_total_hz, 3),
            "peak_hz": round(peak_value, 3),
            "best_phase": int(best_phase),
            "chips_seen": int(st.chips_seen),
            "rx_ms": round(1000.0 * self.rx_time_ema_s, 3),
            "proc_ms": round(1000.0 * self.process_time_ema_s, 3),
            "gap_ms": round(1000.0 * self.frame_gap_ema_s, 3),
            "late_frames": int(self.late_frame_count),
            "gap_slips": int(self.gap_slip_count),
            "monitor_raw_row_dbfs": [round(float(v), 2) for v in self.monitor_raw_row],
            "monitor_row_dbfs": [round(float(v), 2) for v in self.monitor_centered_row],
            "monitor_noise_floor_dbfs": round(float(self.monitor_noise_floor_dbfs), 2),
            "monitor_sideband_snr_db": round(float(self.monitor_sideband_snr_db), 2),
            "monitor_sideband_pos_dbfs": round(float(self.monitor_sideband_pos_dbfs), 2),
            "monitor_sideband_neg_dbfs": round(float(self.monitor_sideband_neg_dbfs), 2),
        }
        payload_text = json.dumps(payload, separators=(",", ":"))
        tmp_path = self.status_path.with_suffix(self.status_path.suffix + ".tmp")
        try:
            tmp_path.write_text(payload_text, encoding="ascii")
            tmp_path.replace(self.status_path)
        except PermissionError:
            # Windows can transiently lock files while another process is reading.
            # Drop this snapshot instead of crashing the realtime loop.
            try:
                self.status_path.write_text(payload_text, encoding="ascii")
            except PermissionError:
                pass

    def update(self, _frame: int) -> tuple:
        if self.stop_requested:
            return tuple()

        frame_start_s = perf_counter()
        self.frame_index += 1
        frame_gap_s = 0.0
        if self.last_frame_start_s is not None:
            frame_gap_s = frame_start_s - self.last_frame_start_s
        self.last_frame_start_s = frame_start_s

        rx_start_s = perf_counter()
        x_raw = normalize_iq(self.sdr.rx())
        rx_elapsed_s = perf_counter() - rx_start_s
        x_dc, self.dc_prev_x, self.dc_prev_y = dc_block_filter(
            x_raw, self.dc_prev_x, self.dc_prev_y, config.DC_BLOCK_ALPHA
        )

        peak_hz = self.prev_peak_hz
        need_peak_update = (
            peak_hz is None
            or self.frame_index % max(1, int(config.RX_ONLY_PEAK_TRACK_EVERY_FRAMES)) == 0
        )
        if need_peak_update:
            _, raw_dbfs_full = compute_spectrum_dbfs(x_raw, config.SAMPLE_RATE)
            self.smoothed_raw = ema_spectrum_power_domain(raw_dbfs_full, self.smoothed_raw, config.FFT_AVG_ALPHA)

            search_samples = x_dc
            if peak_hz is None and config.EXCITER_SEARCH_MIN_HZ <= 0.0:
                search_samples = x_raw

            freqs_hz, dc_dbfs_full = compute_spectrum_dbfs(search_samples, config.SAMPLE_RATE)
            self.smoothed_dc = ema_spectrum_power_domain(dc_dbfs_full, self.smoothed_dc, config.FFT_AVG_ALPHA)
            in_view = np.abs(freqs_hz) <= config.SPECTRUM_SPAN_HZ
            peak_hz, _ = find_exciter_peak(
                freqs_hz,
                self.smoothed_dc,
                in_view,
                self.prev_peak_hz,
                config.EXCITER_SEARCH_MIN_HZ,
                config.EXCITER_SEARCH_MAX_HZ,
                expected_hz=float(config.EXCITER_EXPECTED_HZ),
                expected_tol_hz=float(config.EXCITER_EXPECTED_TOL_HZ),
                strict_expected_band=bool(config.EXCITER_STRICT_EXPECTED_BAND),
                max_step_hz=float(config.EXCITER_MAX_STEP_HZ),
                switch_margin_db=float(config.EXCITER_SWITCH_MARGIN_DB),
            )
            self.prev_peak_hz = peak_hz
            self._update_monitor_spectrum(freqs_hz, peak_hz)

        if peak_hz is not None:
            demod_input = x_dc
            if abs(float(peak_hz)) <= self._dc_bin_guard_hz:
                demod_input = x_raw

            demod = demod_input

            if config.CFO_CORRECTION_ENABLED and demod.size:
                # CFO estimation is expensive at full buffer size. Decimate by
                # CFO_DECIMATE for the residual estimate (Kay's estimator stays
                # accurate for any oversampled tone), then derotate the full
                # buffer with the smoothed CFO.
                cfo_decim = max(1, int(getattr(config, "CFO_DECIMATE", 8)))
                demod_for_cfo = demod[::cfo_decim]
                n = np.arange(demod_for_cfo.size, dtype=np.float64)
                mix = np.exp(
                    -1j * 2.0 * np.pi * float(peak_hz) * (n * cfo_decim)
                    / float(config.SAMPLE_RATE)
                )
                demod_bb = demod_for_cfo.astype(np.complex128) * mix

                # Gate CFO estimate on carrier SNR: the carrier at DC must dominate
                # before the phase-advance estimator is trustworthy.  At low SNR the
                # estimator returns noise-dominated values and corrupts the EMA.
                dc_power = abs(complex(np.mean(demod_bb))) ** 2
                total_power = float(np.mean(np.abs(demod_bb) ** 2))
                residual_power = max(total_power - dc_power, 1e-30)
                snr_db = 10.0 * np.log10(dc_power / residual_power) if dc_power > 0 else -999.0

                max_abs = float(config.CFO_MAX_ABS_HZ)
                if float(snr_db) >= float(config.CFO_SNR_GATE_DB):
                    residual_hat_hz = estimate_residual_cfo_hz(
                        demod_bb.astype(np.complex64),
                        float(config.SAMPLE_RATE) / float(cfo_decim),
                    )
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
                    self.cfo_total_hz = float(np.clip(self.cfo_coarse_hz + self.cfo_fine_hz, -max_abs, max_abs))
                # else: SNR too low — freeze EMA, do not corrupt with noise

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

            # Lock hysteresis is driven by the sideband SNR computed during
            # peak-update frames; running coherent_fsk_metrics over the full
            # buffer here was a duplicate ~30 ms/frame of work.
            self._update_lock_hysteresis()

            self.phase_sample_buffer = np.concatenate((self.phase_sample_buffer, demod))
            chips_added = 0
            for phase, state in self.phase_state.items():
                phase_added = self._slice_chips_for_phase(state, peak_hz)
                chips_added += phase_added
                if phase_added > 0:
                    self._capture_phase_update(phase, state, phase_added)

            if chips_added > 0:
                # Throttle the heavy packet search; chips remain in history.
                decode_cadence = max(1, int(getattr(config, "LIVE_DECODE_EVERY_FRAMES", 1)))
                if self.frame_index % decode_cadence == 0:
                    self._update_live_decode_summary()

                min_next = min(s.next_sample for s in self.phase_state.values())
                if min_next > self.bit_samples:
                    trim = min_next - self.bit_samples
                    self.phase_sample_buffer = self.phase_sample_buffer[trim:]
                    for s in self.phase_state.values():
                        s.next_sample -= trim

        process_elapsed_s = perf_counter() - frame_start_s
        alpha = 0.10
        self.rx_time_ema_s = ((1.0 - alpha) * self.rx_time_ema_s) + (alpha * rx_elapsed_s)
        self.process_time_ema_s = (
            ((1.0 - alpha) * self.process_time_ema_s) + (alpha * process_elapsed_s)
        )
        if frame_gap_s > 0.0:
            self.frame_gap_ema_s = (
                ((1.0 - alpha) * self.frame_gap_ema_s) + (alpha * frame_gap_s)
            )

        late_budget_s = self.realtime_budget_s * float(config.JITTER_LATE_FACTOR)
        gap_budget_s = self.realtime_budget_s * float(config.JITTER_GAP_FACTOR)
        if process_elapsed_s > late_budget_s:
            self.late_frame_count += 1
        if frame_gap_s > gap_budget_s:
            self.gap_slip_count += 1

        self._write_status_snapshot(peak_hz)
        self._emit_status(peak_hz)

        return tuple()

