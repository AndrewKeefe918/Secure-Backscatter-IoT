"""ReceiverLoop — holds all per-frame mutable state and the FuncAnimation callback.

The original monolithic update() closure is refactored into this class so that
state previously held as nonlocal variables becomes explicit instance attributes.
"""

import numpy as np

from . import config
from .dsp import (
    normalize_iq,
    dc_block_filter,
    compute_spectrum_dbfs,
    bandpass_filter_around_carrier,
    coherent_ncc,
    compute_sideband_snr,
)
from .packet_decoder import (
    bytes_to_bit_list,
    bits_to_bytes,
    find_header_match,
    majority_decode_triplets,
    bits_to_text,
    safe_ascii,
)
from .gui_setup import BasebandWindow, CarrierWindow, NccWindow


class ReceiverLoop:
    """Encapsulates all mutable receiver state and the per-frame update callback."""

    def __init__(
        self,
        sdr: object,
        bb_win: BasebandWindow,
        carrier_win: CarrierWindow,
        ncc_win: NccWindow,
    ) -> None:
        self.sdr = sdr
        self.bb = bb_win
        self.cw = carrier_win
        self.nw = ncc_win
        self.stop_requested: bool = False

        # Spectrum EMA state (power domain)
        self.smoothed_raw_spec_dbfs: np.ndarray = np.array([], dtype=np.float64)
        self.smoothed_dc_blocked_spec_dbfs: np.ndarray = np.array([], dtype=np.float64)

        # DC-block filter continuity state
        self.dc_prev_x = np.complex64(0.0 + 0.0j)
        self.dc_prev_y = np.complex64(0.0 + 0.0j)

        # Carrier tracking
        self.prev_peak_dbfs: float | None = None
        self.prev_peak_hz: float = 0.0

        # Envelope autoscale
        self.env_ylim_low: float = -0.02
        self.env_ylim_high: float = 0.02

        # Packet / bit decoding
        self.bit_samples = int(config.SAMPLE_RATE * (config.BIT_DURATION_MS / 1000.0))
        self.phase_offsets = list(range(0, self.bit_samples, config.BIT_PHASE_STEP_SAMPLES))
        self.phase_sample_buffer: np.ndarray = np.array([], dtype=np.complex64)
        self.packet_header_bits = bytes_to_bit_list(config.PREAMBLE_BYTES + config.SYNC_BYTES)
        self.payload_bits_len = len(config.PAYLOAD_BYTES) * 8
        self.packet_status_text = "Waiting for preamble+sync"
        self.packet_status_hold = 0
        self.decoded_packets = 0
        self.total_bits = 0
        self.next_debug_bits = config.TERMINAL_DEBUG_BITS_EVERY

        # NCC lock state
        self.ncc_abs_ema: float = 0.0
        self.ncc_lock: bool = False
        self.ncc_enter_count: int = 0
        self.ncc_exit_count: int = 0

        # Per-phase decode state
        self.phase_state: dict[int, dict[str, object]] = {
            phase: {
                "next_sample": phase,
                "chips": [],
                "base_chip_index": 0,
                "search_start_by_offset": [0, 0, 0],
                "last_header_abs_by_offset": [-1, -1, -1],
                "chips_seen": 0,
            }
            for phase in self.phase_offsets
        }

    # ------------------------------------------------------------------
    # Per-frame callback — called by FuncAnimation every interval ms
    # ------------------------------------------------------------------

    def update(self, _frame: int) -> tuple:
        bb = self.bb
        cw = self.cw
        nw = self.nw

        if self.stop_requested:
            return (
                bb.line_i, bb.line_q, bb.line_fft_raw, bb.line_fft_dc_blocked,
                bb.exciter_marker, cw.line_centered, cw.waterfall_img,
                bb.status, cw.status,
            )

        # ---- Acquire samples ------------------------------------------------
        x_raw = normalize_iq(self.sdr.rx())
        x_dc_blocked, self.dc_prev_x, self.dc_prev_y = dc_block_filter(
            x_raw, self.dc_prev_x, self.dc_prev_y, config.DC_BLOCK_ALPHA
        )

        # ---- Time-domain plot -----------------------------------------------
        shown = x_raw[: config.TIME_SAMPLES]
        bb.line_i.set_ydata(np.real(shown))
        bb.line_q.set_ydata(np.imag(shown))

        # ---- Spectrum computation (power-domain EMA) ------------------------
        freqs_hz, raw_spectrum_dbfs = compute_spectrum_dbfs(x_raw, config.SAMPLE_RATE)
        _, dc_unfiltered_spectrum_dbfs = compute_spectrum_dbfs(x_dc_blocked, config.SAMPLE_RATE)

        in_view = np.abs(freqs_hz) <= config.SPECTRUM_SPAN_HZ
        rms = float(np.sqrt(np.mean(np.abs(x_raw) ** 2)))

        # EMA in linear power domain — avoids log-domain Jensen's bias.
        raw_pow = np.power(10.0, raw_spectrum_dbfs * 0.1)
        dc_pow = np.power(10.0, dc_unfiltered_spectrum_dbfs * 0.1)
        if self.smoothed_raw_spec_dbfs.size == 0:
            smoothed_raw_pow = raw_pow
            smoothed_dc_pow = dc_pow
        else:
            smoothed_raw_pow = np.power(10.0, self.smoothed_raw_spec_dbfs * 0.1)
            smoothed_dc_pow = np.power(10.0, self.smoothed_dc_blocked_spec_dbfs * 0.1)
            smoothed_raw_pow = (
                config.FFT_AVG_ALPHA * raw_pow + (1.0 - config.FFT_AVG_ALPHA) * smoothed_raw_pow
            )
            smoothed_dc_pow = (
                config.FFT_AVG_ALPHA * dc_pow + (1.0 - config.FFT_AVG_ALPHA) * smoothed_dc_pow
            )
        self.smoothed_raw_spec_dbfs = 10.0 * np.log10(np.maximum(smoothed_raw_pow, 1e-20))
        self.smoothed_dc_blocked_spec_dbfs = 10.0 * np.log10(np.maximum(smoothed_dc_pow, 1e-20))

        # ---- Carrier search -------------------------------------------------
        search_mask = (
            (np.abs(freqs_hz) >= config.EXCITER_SEARCH_MIN_HZ)
            & (np.abs(freqs_hz) <= config.EXCITER_SEARCH_MAX_HZ)
        )
        search_view = in_view & search_mask
        exciter_freqs_hz = freqs_hz[search_view]
        exciter_spec_view = self.smoothed_dc_blocked_spec_dbfs[search_view]
        if len(exciter_spec_view):
            exciter_idx = int(np.argmax(exciter_spec_view))
            cand_peak_hz = float(exciter_freqs_hz[exciter_idx])
            cand_peak_dbfs = float(exciter_spec_view[exciter_idx])
            noise_ref_dbfs = float(
                np.percentile(self.smoothed_dc_blocked_spec_dbfs[in_view], 50)
            )
            if (cand_peak_dbfs - noise_ref_dbfs) >= 10.0 or self.prev_peak_hz == 0.0:
                peak_hz = cand_peak_hz
                peak_dbfs = cand_peak_dbfs
            else:
                peak_hz = self.prev_peak_hz
                prev_idx = int(np.argmin(np.abs(freqs_hz - peak_hz)))
                peak_dbfs = float(self.smoothed_dc_blocked_spec_dbfs[prev_idx])
        else:
            peak_hz = self.prev_peak_hz
            prev_idx = int(np.argmin(np.abs(freqs_hz - peak_hz)))
            peak_dbfs = float(self.smoothed_dc_blocked_spec_dbfs[prev_idx])

        self.prev_peak_hz = peak_hz
        bb.exciter_marker.set_xdata([peak_hz / 1000.0, peak_hz / 1000.0])

        if config.ENABLE_FOCUSED_FILTER:
            x_processed = bandpass_filter_around_carrier(
                x_dc_blocked, peak_hz, config.SAMPLE_RATE, config.FOCUSED_PASSBAND_HZ,
            )
            _, processed_spectrum_dbfs = compute_spectrum_dbfs(x_processed, config.SAMPLE_RATE)
            self.smoothed_dc_blocked_spec_dbfs = (
                config.FFT_AVG_ALPHA * processed_spectrum_dbfs
                + (1.0 - config.FFT_AVG_ALPHA) * self.smoothed_dc_blocked_spec_dbfs
            )

        # ---- Baseband spectrum plot -----------------------------------------
        freqs_khz = freqs_hz[in_view] / 1000.0
        bb.line_fft_raw.set_data(freqs_khz, self.smoothed_raw_spec_dbfs[in_view])
        bb.line_fft_dc_blocked.set_data(freqs_khz, self.smoothed_dc_blocked_spec_dbfs[in_view])

        dc_idx = int(np.argmin(np.abs(freqs_hz)))
        dc_dbfs = float(self.smoothed_raw_spec_dbfs[dc_idx])
        dc_to_carrier_db = peak_dbfs - dc_dbfs

        if self.prev_peak_dbfs is None or abs(peak_dbfs - self.prev_peak_dbfs) > 5.0:
            cw.ax_centered.set_ylim(peak_dbfs - 40.0, peak_dbfs + 10.0)
            self.prev_peak_dbfs = peak_dbfs

        # ---- Carrier-centered spectrum ---------------------------------------
        centered_freqs_hz = freqs_hz - peak_hz
        centered_view = np.abs(centered_freqs_hz) <= config.CENTERED_SPAN_HZ
        centered_freqs_khz = centered_freqs_hz[centered_view] / 1000.0
        centered_spec = self.smoothed_dc_blocked_spec_dbfs[centered_view]
        centered_spec_display = centered_spec
        if centered_spec.size >= config.CENTERED_FREQ_SMOOTH_BINS:
            kernel = (
                np.ones(config.CENTERED_FREQ_SMOOTH_BINS, dtype=np.float64)
                / float(config.CENTERED_FREQ_SMOOTH_BINS)
            )
            centered_spec_display = np.convolve(centered_spec, kernel, mode="same")
        cw.line_centered.set_data(centered_freqs_khz, centered_spec_display)

        carrier_mask = np.abs(centered_freqs_khz) > 0.5
        if carrier_mask.any():
            noise_floor_est_y = float(np.percentile(centered_spec_display[carrier_mask], 20))
            cw.ax_centered.set_ylim(noise_floor_est_y - 6.0, peak_dbfs + 4.0)
            self.prev_peak_dbfs = peak_dbfs

        # ---- Sideband SNR markers -------------------------------------------
        snr_db, sb_pos, sb_neg, noise_floor_sb = compute_sideband_snr(
            centered_freqs_khz, centered_spec_display,
            config.SIDEBAND_OFFSET_KHZ, config.SIDEBAND_WINDOW_HZ,
        )
        cw.sideband_scatter.set_offsets(
            np.array([[-config.SIDEBAND_OFFSET_KHZ, sb_neg], [config.SIDEBAND_OFFSET_KHZ, sb_pos]])
        )
        dot_color = "#44ff88" if snr_db >= config.SNR_LOCK_THRESHOLD_DB else "#ff4444"
        cw.sideband_scatter.set_facecolor([dot_color, dot_color])
        cw.snr_threshold_line.set_ydata([noise_floor_sb + config.SNR_LOCK_THRESHOLD_DB])

        # ---- Waterfall ------------------------------------------------------
        if centered_spec.size:
            remapped_centered = np.interp(
                cw.centered_axis,
                centered_freqs_khz,
                centered_spec_display,
                left=-140.0,
                right=-140.0,
            )
            remapped_centered = np.maximum(remapped_centered, -140.0)

            # Compress carrier core so sidebands stay visible.
            frame_noise_mask = np.abs(cw.centered_axis) > 0.5
            if frame_noise_mask.any():
                frame_noise = float(np.percentile(remapped_centered[frame_noise_mask], 20))
                carrier_core = np.abs(cw.centered_axis) <= 0.20
                remapped_centered[carrier_core] = np.minimum(
                    remapped_centered[carrier_core],
                    frame_noise + 18.0,
                )

            cw.waterfall_data[:-1] = cw.waterfall_data[1:]
            if config.WATERFALL_ROWS > 1:
                remapped_centered = (
                    config.WATERFALL_ROW_BLEND * remapped_centered
                    + (1.0 - config.WATERFALL_ROW_BLEND) * cw.waterfall_data[-2]
                )
            cw.waterfall_data[-1] = remapped_centered
            cw.waterfall_img.set_data(cw.waterfall_data)

            wf_axis_mask = np.abs(cw.centered_axis) > 0.5
            valid_wf = cw.waterfall_data[:, wf_axis_mask]
            valid_wf = valid_wf[valid_wf > -139.0]
            if valid_wf.size:
                noise_floor_est = float(np.percentile(valid_wf, 15))
                cw.waterfall_img.set_clim(
                    vmin=noise_floor_est - 1.0,
                    vmax=noise_floor_est + config.WATERFALL_DYN_RANGE_DB,
                )

        # ---- Status text ----------------------------------------------------
        bb.status.set_text(
            f"RX={config.RX_URI} | LO={config.FREQ_HZ/1e9:.3f} GHz | RMS={rms:.4f} FS | "
            f"Exciter={peak_hz:,.1f} Hz @ {peak_dbfs:.1f} dBFS | DC={dc_dbfs:.1f} dBFS | "
            f"Carrier-DC={dc_to_carrier_db:.1f} dB"
        )
        cw.status.set_text(
            f"Carrier={peak_hz:,.1f} Hz | Span=±{config.CENTERED_SPAN_HZ/1000.0:.0f} kHz | "
            f"SB SNR={snr_db:+.1f} dB (+{config.SIDEBAND_OFFSET_KHZ:.0f}kHz={sb_pos:.1f} "
            f"-{config.SIDEBAND_OFFSET_KHZ:.0f}kHz={sb_neg:.1f} Noise={noise_floor_sb:.1f} dBFS)"
        )
        cw.fig.canvas.draw_idle()

        # ---- Coherent AM demodulation ---------------------------------------
        if peak_hz != 0.0:
            ncc_val, envelope = coherent_ncc(x_raw, peak_hz, 1000.0, config.SAMPLE_RATE)
            nw.ncc_history = np.roll(nw.ncc_history, -1)
            nw.ncc_history[-1] = ncc_val
            nw.line_ncc.set_ydata(nw.ncc_history)

            env_show = envelope[: nw.env_plot_len]
            nw.line_env.set_ydata(env_show)

            lo_p = float(np.percentile(env_show, 2.0))
            hi_p = float(np.percentile(env_show, 98.0))
            mid = 0.5 * (lo_p + hi_p)
            span = max(hi_p - lo_p, config.ENV_Y_MIN_SPAN)
            target_low = mid - 0.6 * span
            target_high = mid + 0.6 * span
            self.env_ylim_low = (
                (1.0 - config.ENV_Y_SMOOTH_ALPHA) * self.env_ylim_low
                + config.ENV_Y_SMOOTH_ALPHA * target_low
            )
            self.env_ylim_high = (
                (1.0 - config.ENV_Y_SMOOTH_ALPHA) * self.env_ylim_high
                + config.ENV_Y_SMOOTH_ALPHA * target_high
            )
            nw.ax_env.set_ylim(self.env_ylim_low, self.env_ylim_high)

            # NCC lock hysteresis
            self.ncc_abs_ema = (
                (1.0 - config.NCC_DISPLAY_ALPHA) * self.ncc_abs_ema
                + config.NCC_DISPLAY_ALPHA * abs(ncc_val)
            )
            if not self.ncc_lock:
                if self.ncc_abs_ema >= config.NCC_ENTER_THRESHOLD:
                    self.ncc_enter_count += 1
                else:
                    self.ncc_enter_count = 0
                if self.ncc_enter_count >= config.NCC_ENTER_FRAMES:
                    self.ncc_lock = True
                    self.ncc_exit_count = 0
            else:
                if self.ncc_abs_ema <= config.NCC_EXIT_THRESHOLD:
                    self.ncc_exit_count += 1
                else:
                    self.ncc_exit_count = 0
                if self.ncc_exit_count >= config.NCC_EXIT_FRAMES:
                    self.ncc_lock = False
                    self.ncc_enter_count = 0

            # ---- Multi-phase 50 ms slicer -----------------------------------
            self.phase_sample_buffer = np.concatenate((self.phase_sample_buffer, x_raw))
            phase_chips_added = 0

            for phase, state in self.phase_state.items():
                next_sample = int(state["next_sample"])
                phase_chips: list[int] = state["chips"]  # type: ignore[assignment]

                while next_sample + self.bit_samples <= len(self.phase_sample_buffer):
                    bit_chunk = self.phase_sample_buffer[next_sample : next_sample + self.bit_samples]
                    bit_ncc, _ = coherent_ncc(bit_chunk, peak_hz, 1000.0, config.SAMPLE_RATE)
                    phase_chips.append(1 if abs(bit_ncc) >= config.BIT_NCC_THRESHOLD else 0)
                    next_sample += self.bit_samples
                    state["chips_seen"] = int(state["chips_seen"]) + 1
                    phase_chips_added += 1
                state["next_sample"] = next_sample

                if len(phase_chips) > config.PHASE_HISTORY_BITS * config.REPETITION_CHIPS:
                    drop = len(phase_chips) - config.PHASE_HISTORY_BITS * config.REPETITION_CHIPS
                    del phase_chips[:drop]
                    state["base_chip_index"] = int(state["base_chip_index"]) + drop

                search_start_by_offset: list[int] = state["search_start_by_offset"]  # type: ignore[assignment]
                last_header_abs_by_offset: list[int] = state["last_header_abs_by_offset"]  # type: ignore[assignment]

                for decode_offset in range(config.REPETITION_CHIPS):
                    decoded_bits = majority_decode_triplets(phase_chips, decode_offset)
                    base_bit_index = (
                        (int(state["base_chip_index"]) + decode_offset) // config.REPETITION_CHIPS
                    )
                    while True:
                        search_start = search_start_by_offset[decode_offset]
                        header_idx, header_errors = find_header_match(
                            decoded_bits,
                            self.packet_header_bits,
                            search_start,
                            config.HEADER_MAX_BIT_ERRORS,
                        )
                        if header_idx < 0:
                            search_start_by_offset[decode_offset] = max(
                                0,
                                len(decoded_bits) - len(self.packet_header_bits) + 1,
                            )
                            break
                        header_abs = base_bit_index + header_idx
                        if header_abs <= last_header_abs_by_offset[decode_offset]:
                            search_start_by_offset[decode_offset] = header_idx + 1
                            continue
                        payload_start = header_idx + len(self.packet_header_bits)
                        payload_end = payload_start + self.payload_bits_len
                        if payload_end > len(decoded_bits):
                            search_start_by_offset[decode_offset] = header_idx
                            break
                        payload = bits_to_bytes(decoded_bits[payload_start:payload_end])
                        payload_hex = payload.hex().upper() if payload else ""
                        payload_ascii = safe_ascii(payload)
                        print(
                            f"[RX HEADER] phase={phase} chip_offset={decode_offset} "
                            f"bit={header_abs} header_errors={header_errors} "
                            f"payload_hex={payload_hex} payload_ascii={payload_ascii!r}",
                            flush=True,
                        )
                        if payload == config.PAYLOAD_BYTES:
                            self.decoded_packets += 1
                            self.packet_status_text = (
                                f"Packet: OPEN detected at phase {phase}, offset {decode_offset}"
                            )
                            self.packet_status_hold = config.PACKET_STATUS_HOLD_FRAMES
                            print(
                                f"[RX PACKET {self.decoded_packets}] phase={phase} "
                                f"chip_offset={decode_offset} bit={header_abs} "
                                f"header_errors={header_errors} PREAMBLE+SYNC+OPEN",
                                flush=True,
                            )
                        else:
                            self.packet_status_text = (
                                f"Packet: header phase {phase}, offset {decode_offset}, "
                                f"errors={header_errors}, payload={payload!r}"
                            )
                            self.packet_status_hold = config.PACKET_STATUS_HOLD_FRAMES
                        last_header_abs_by_offset[decode_offset] = header_abs
                        search_start_by_offset[decode_offset] = header_idx + 1

            if phase_chips_added > 0:
                self.total_bits += phase_chips_added
                while self.total_bits >= self.next_debug_bits:
                    best_phase = max(
                        self.phase_offsets,
                        key=lambda p: int(self.phase_state[p]["chips_seen"]),
                    )
                    best_chips: list[int] = self.phase_state[best_phase]["chips"]  # type: ignore[assignment]
                    best_decoded = majority_decode_triplets(best_chips, 0)
                    chip_tail = bits_to_text(best_chips[-config.TERMINAL_DEBUG_BIT_TAIL :])
                    bit_tail = bits_to_text(
                        best_decoded[-(config.TERMINAL_DEBUG_BIT_TAIL // config.REPETITION_CHIPS) :]
                    )
                    print(
                        f"[RX DEBUG] phase={best_phase} "
                        f"chips={int(self.phase_state[best_phase]['chips_seen'])} "
                        f"ncc_ema={self.ncc_abs_ema:.3f} lock={'1' if self.ncc_lock else '0'} "
                        f"chip_tail={chip_tail} bit_tail={bit_tail}",
                        flush=True,
                    )
                    self.next_debug_bits += config.TERMINAL_DEBUG_BITS_EVERY

                min_next_sample = min(
                    int(state["next_sample"]) for state in self.phase_state.values()
                )
                if min_next_sample > self.bit_samples:
                    trim = min_next_sample - self.bit_samples
                    self.phase_sample_buffer = self.phase_sample_buffer[trim:]
                    for state in self.phase_state.values():
                        state["next_sample"] = int(state["next_sample"]) - trim

            if self.packet_status_hold > 0:
                self.packet_status_hold -= 1
            else:
                self.packet_status_text = "Waiting for preamble+sync"

            nw.ncc_status.set_text(
                f"NCC={ncc_val:+.3f} | EMA={self.ncc_abs_ema:.3f} | "
                f"{'LOCKED' if self.ncc_lock else 'searching'} | "
                f"Carrier={peak_hz:,.0f} Hz | {self.packet_status_text}"
            )
            nw.fig.canvas.draw_idle()

        return (
            bb.line_i, bb.line_q, bb.line_fft_raw, bb.line_fft_dc_blocked,
            bb.exciter_marker, cw.line_centered, cw.waterfall_img,
            bb.status, cw.status,
        )
