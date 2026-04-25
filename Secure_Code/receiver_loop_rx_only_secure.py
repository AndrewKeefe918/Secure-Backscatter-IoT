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

from . import config_secure as config
from .dsp_secure import (
    bandpass_filter_around_carrier,
    coherent_fsk_metrics,
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
from .packet_decoder_secure import bits_to_text
from .packet_decoder_secure import (
    bits_to_bytes,
    bytes_to_bit_list,
    majority_decode_triplets,
)
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
        self.phase_offsets = [0]
        self.phase_sample_buffer = np.array([], dtype=np.complex64)
        self.phase_state = {p: PhaseState(next_sample=p) for p in self.phase_offsets}

        self.smoothed_dc = np.array([], dtype=np.float64)
        self.dc_prev_x = np.complex64(0.0 + 0.0j)
        self.dc_prev_y = np.complex64(0.0 + 0.0j)
        self.prev_peak_hz = 0.0
        self.monitor_axis_khz = np.linspace(
            -config.CENTERED_SPAN_HZ / 1000.0,
            config.CENTERED_SPAN_HZ / 1000.0,
            int(config.RX_MONITOR_SPECTRUM_BINS),
            dtype=np.float64,
        )
        self.monitor_centered_row = np.full_like(self.monitor_axis_khz, -140.0)
        self.monitor_noise_floor_dbfs = -140.0
        self.monitor_sideband_snr_db = 0.0
        self.monitor_sideband_pos_dbfs = -140.0
        self.monitor_sideband_neg_dbfs = -140.0

        self.cfo_coarse_hz = 0.0
        self.cfo_fine_hz = 0.0
        self.cfo_total_hz = 0.0
        self.cfo_phase_rad = 0.0

        self.ncc_abs_ema = 0.0
        self.ncc_lock = False
        self.ncc_enter_count = 0
        self.ncc_exit_count = 0
        self.last_decode_summary = "No live decode yet"
        self.last_logic_tail = ""

        self.total_bits = 0

        self.frame_index = 0
        self.last_frame_start_s: float | None = None
        self.buffer_duration_s = float(config.RX_BUFFER_SIZE) / float(config.SAMPLE_RATE)
        self.update_interval_s = float(config.ANIMATION_INTERVAL_MS) / 1000.0
        self.realtime_budget_s = max(self.buffer_duration_s, self.update_interval_s)
        self.rx_time_ema_s = 0.0
        self.process_time_ema_s = 0.0
        self.frame_gap_ema_s = 0.0
        self.max_process_s = 0.0
        self.late_frame_count = 0
        self.gap_slip_count = 0

        self.secure_rx = SecureReceiver(
            bytes.fromhex(config.SHARED_KEY_HEX),
            Path(config.SECURE_RX_STATE_PATH),
        )
        self.decoded_packets = 0
        self._packet_cache: dict[tuple, object] = {}

        capture_path = Path(config.RX_CAPTURE_NDJSON)
        capture_path.parent.mkdir(parents=True, exist_ok=True)
        self.capture_file = capture_path.open("a", encoding="ascii")
        self.status_path = Path(config.RX_STATUS_JSON)
        self.status_path.parent.mkdir(parents=True, exist_ok=True)

    def close(self) -> None:
        try:
            self.capture_file.close()
        except Exception:
            pass

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

    def _weak_triplet_count(
        self,
        chips: list[int],
        decode_offset: int,
        bit_start: int,
        bit_count: int,
    ) -> int:
        """Count logical bits decided by a weak 2-of-3 majority in a window."""
        weak = 0
        step = int(config.REPETITION_CHIPS)
        for bit_idx in range(bit_start, bit_start + bit_count):
            chip_start = decode_offset + bit_idx * step
            chip_end = chip_start + step
            if chip_end > len(chips):
                break
            ones = sum(chips[chip_start:chip_end])
            if ones not in (0, step):
                weak += 1
        return weak

    def _update_live_decode_summary(self) -> None:
        header_bits = bytes_to_bit_list(config.PREAMBLE_BYTES + config.SYNC_BYTES)
        payload_len = int(config.LIVE_DECODE_PAYLOAD_BYTES) * 8
        packet_bits = len(header_bits) + payload_len

        best_text = "No live decode yet"
        best_score: tuple[int, ...] | None = None
        best_logic_tail = ""

        for phase, state in sorted(self.phase_state.items()):
            chips = state.chips
            if len(chips) < config.REPETITION_CHIPS * packet_bits:
                continue

            for decode_offset in range(config.REPETITION_CHIPS):
                decoded_bits = majority_decode_triplets(chips, decode_offset)
                if not decoded_bits:
                    continue

                logic_tail = bits_to_text(decoded_bits[-48:])
                if best_logic_tail == "":
                    best_logic_tail = logic_tail

                if len(decoded_bits) < packet_bits:
                    continue

                recent_start = max(0, len(decoded_bits) - config.LIVE_DECODE_RECENT_BITS)
                last_start = len(decoded_bits) - packet_bits
                scan_start = max(0, min(recent_start, last_start))
                for header_idx in range(scan_start, last_start + 1):
                    header_errors = sum(
                        1
                        for left, right in zip(
                            decoded_bits[header_idx : header_idx + len(header_bits)],
                            header_bits,
                        )
                        if left != right
                    )
                    if header_errors > config.LIVE_DECODE_MAX_HEADER_ERRORS:
                        continue

                    payload_start = header_idx + len(header_bits)
                    payload_end = payload_start + payload_len
                    payload_bits = decoded_bits[payload_start:payload_end]
                    payload = bits_to_bytes(payload_bits)
                    recency = len(decoded_bits) - payload_end
                    weak_bits = self._weak_triplet_count(
                        chips,
                        decode_offset,
                        header_idx,
                        len(header_bits) + payload_len,
                    )
                    if weak_bits > int(config.LIVE_DECODE_MAX_WEAK_BITS):
                        continue

                    # Compute absolute bit position for deduplication across frames.
                    header_abs = state.base_chip_index + header_idx

                    # Verify once per unique absolute packet position; cache the result
                    # so repeated scans of the same chip window don't consume the
                    # replay counter or produce duplicate AUTHENTICATED log lines.
                    cache_key = (phase, decode_offset, header_abs)
                    if cache_key not in self._packet_cache:
                        result = self.secure_rx.verify_and_decrypt(payload)
                        self._packet_cache[cache_key] = result
                        if result.valid:
                            self.decoded_packets += 1
                            print(
                                f"[RX PACKET {self.decoded_packets}] phase={phase} "
                                f"chip_offset={decode_offset} bit={header_abs} "
                                f"header_errors={header_errors} "
                                f"counter={result.counter} plaintext={result.plaintext!r}  AUTHENTICATED",
                                flush=True,
                            )
                    else:
                        result = self._packet_cache[cache_key]

                    if result.valid:
                        score = (0, header_errors, weak_bits, recency)
                        text = (
                            f"phase={phase} off={decode_offset} bit={header_idx} herr={header_errors} "
                            f"weak={weak_bits} counter={result.counter} "
                            f"plaintext={result.plaintext!r}  AUTHENTICATED"
                        )
                    else:
                        score = (1, header_errors, weak_bits, recency)
                        text = (
                            f"phase={phase} off={decode_offset} bit={header_idx} herr={header_errors} "
                            f"weak={weak_bits} payload={payload.hex().upper()} "
                            f"REJECTED: {result.reason}"
                        )

                    if best_score is None or score < best_score:
                        best_score = score
                        best_text = text
                        best_logic_tail = logic_tail

        self.last_decode_summary = best_text
        self.last_logic_tail = best_logic_tail

    def _emit_status(self, peak_hz: float) -> None:
        if self.frame_index % max(1, int(config.RX_TERMINAL_STATUS_EVERY_FRAMES)) != 0:
            return
        best_phase = max(self.phase_offsets, key=lambda p: self.phase_state[p].chips_seen)
        print(
            f"[RX STATUS] frame={self.frame_index} lock={'1' if self.ncc_lock else '0'} "
            f"phase={best_phase} peak={peak_hz:+.1f}Hz cfo={self.cfo_total_hz:+.1f}Hz "
            f"snr={self.monitor_sideband_snr_db:+.1f}dB bits={self.last_logic_tail or '-'} "
            f"decode={self.last_decode_summary}",
            flush=True,
        )

    def _update_monitor_spectrum(self, freqs_hz: np.ndarray, peak_hz: float) -> None:
        if peak_hz == 0.0 or self.smoothed_dc.size == 0:
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

    def _write_status_snapshot(self, peak_hz: float) -> None:
        if self.frame_index % max(1, int(config.RX_STATUS_EVERY_FRAMES)) != 0:
            return
        best_phase = max(self.phase_offsets, key=lambda p: self.phase_state[p].chips_seen)
        st = self.phase_state[best_phase]
        payload = {
            "frame": self.frame_index,
            "lock": 1 if self.ncc_lock else 0,
            "ncc_ema": round(self.ncc_abs_ema, 4),
            "cfo_hz": round(self.cfo_total_hz, 3),
            "coarse_hz": round(self.cfo_coarse_hz, 3),
            "fine_hz": round(self.cfo_fine_hz, 3),
            "peak_hz": round(float(peak_hz), 3),
            "best_phase": int(best_phase),
            "chips_seen": int(st.chips_seen),
            "chip_tail": bits_to_text(st.chips[-48:]),
            "rx_ms": round(1000.0 * self.rx_time_ema_s, 3),
            "proc_ms": round(1000.0 * self.process_time_ema_s, 3),
            "gap_ms": round(1000.0 * self.frame_gap_ema_s, 3),
            "budget_ms": round(1000.0 * self.realtime_budget_s, 3),
            "late_frames": int(self.late_frame_count),
            "gap_slips": int(self.gap_slip_count),
            "monitor_axis_khz": [round(float(v), 3) for v in self.monitor_axis_khz],
            "monitor_row_dbfs": [round(float(v), 2) for v in self.monitor_centered_row],
            "monitor_noise_floor_dbfs": round(float(self.monitor_noise_floor_dbfs), 2),
            "monitor_sideband_snr_db": round(float(self.monitor_sideband_snr_db), 2),
            "monitor_sideband_pos_dbfs": round(float(self.monitor_sideband_pos_dbfs), 2),
            "monitor_sideband_neg_dbfs": round(float(self.monitor_sideband_neg_dbfs), 2),
            "logic_tail": self.last_logic_tail,
            "decode_summary": self.last_decode_summary,
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
            peak_hz == 0.0
            or self.frame_index % max(1, int(config.RX_ONLY_PEAK_TRACK_EVERY_FRAMES)) == 0
        )
        if need_peak_update:
            freqs_hz, dc_dbfs_full = compute_spectrum_dbfs(x_dc, config.SAMPLE_RATE)
            self.smoothed_dc = ema_spectrum_power_domain(dc_dbfs_full, self.smoothed_dc, config.FFT_AVG_ALPHA)
            in_view = np.abs(freqs_hz) <= config.SPECTRUM_SPAN_HZ
            peak_hz, _ = find_exciter_peak(
                freqs_hz,
                self.smoothed_dc,
                in_view,
                self.prev_peak_hz,
                config.EXCITER_SEARCH_MIN_HZ,
                config.EXCITER_SEARCH_MAX_HZ,
            )
            self.prev_peak_hz = peak_hz
            self._update_monitor_spectrum(freqs_hz, peak_hz)

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
                self.cfo_total_hz = float(np.clip(self.cfo_coarse_hz + self.cfo_fine_hz, -max_abs, max_abs))

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

            m_f1_buf, m_f0_buf, _ = coherent_fsk_metrics(
                demod, peak_hz, config.FSK_F1_HZ, config.FSK_F0_HZ, config.SAMPLE_RATE
            )
            self._update_lock_hysteresis(max(m_f1_buf, m_f0_buf))

            self.phase_sample_buffer = np.concatenate((self.phase_sample_buffer, demod))
            chips_added = 0
            for phase, state in self.phase_state.items():
                phase_added = self._slice_chips_for_phase(state, peak_hz)
                chips_added += phase_added
                if phase_added > 0:
                    self._capture_phase_update(phase, state, phase_added)

            if chips_added > 0:
                self.total_bits += chips_added
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

        self.max_process_s = max(self.max_process_s, process_elapsed_s)
        late_budget_s = self.realtime_budget_s * float(config.JITTER_LATE_FACTOR)
        gap_budget_s = self.realtime_budget_s * float(config.JITTER_GAP_FACTOR)
        if process_elapsed_s > late_budget_s:
            self.late_frame_count += 1
        if frame_gap_s > gap_budget_s:
            self.gap_slip_count += 1

        self._write_status_snapshot(peak_hz)
        self._emit_status(peak_hz)

        return tuple()
