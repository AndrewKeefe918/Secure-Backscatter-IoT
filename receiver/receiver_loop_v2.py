"""ReceiverLoop — per-frame state and FuncAnimation callback."""

from dataclasses import dataclass, field

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
from .packet_decoder import (
    bytes_to_bit_list,
    bits_to_bytes,
    find_header_match,
    majority_decode_triplets,
    bits_to_text,
    safe_ascii,
)
from .gui_setup import BasebandWindow, CarrierWindow, NccWindow


@dataclass
class PhaseState:
    next_sample: int
    chips: list[int] = field(default_factory=list)
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
        self, sdr: object, bb: BasebandWindow, cw: CarrierWindow, nw: NccWindow
    ) -> None:
        self.sdr, self.bb, self.cw, self.nw = sdr, bb, cw, nw
        self.stop_requested = False

        # Spectrum / carrier state
        self.smoothed_raw = np.array([], dtype=np.float64)
        self.smoothed_dc = np.array([], dtype=np.float64)
        self.dc_prev_x = np.complex64(0.0 + 0.0j)
        self.dc_prev_y = np.complex64(0.0 + 0.0j)
        self.prev_peak_dbfs: float | None = None
        self.prev_peak_hz = 0.0

        # Envelope plot autoscale
        self.env_ylim_low, self.env_ylim_high = -0.02, 0.02

        # Packet/bit decode state
        self.bit_samples = config.SAMPLES_PER_CHIP
        self.phase_offsets = list(range(0, self.bit_samples, config.BIT_PHASE_STEP_SAMPLES))
        self.phase_sample_buffer = np.array([], dtype=np.complex64)
        self.phase_state = {p: PhaseState(next_sample=p) for p in self.phase_offsets}
        self.packet_header_bits = bytes_to_bit_list(config.PREAMBLE_BYTES + config.SYNC_BYTES)
        self.payload_bits_len = len(config.PAYLOAD_BYTES) * 8
        self.packet_status_text = "Waiting for preamble+sync"
        self.packet_status_hold = 0
        self.decoded_packets = 0
        self.total_bits = 0
        self.next_debug_bits = config.TERMINAL_DEBUG_BITS_EVERY

        # NCC lock / adaptive threshold state
        self.ncc_abs_ema = 0.0
        self.ncc_lock = False
        self.ncc_enter_count = 0
        self.ncc_exit_count = 0
        self.bit_ncc_noise_ema = 0.0
        self.bit_ncc_threshold = float(config.BIT_NCC_THRESHOLD)

    # ------------------------------------------------------------------
    # Helpers — keep update() readable
    # ------------------------------------------------------------------

    def _update_lock_hysteresis(self, abs_ncc: float) -> None:
        self.ncc_abs_ema = (
            (1.0 - config.NCC_DISPLAY_ALPHA) * self.ncc_abs_ema
            + config.NCC_DISPLAY_ALPHA * abs_ncc
        )
        if not self.ncc_lock:
            self.ncc_enter_count = self.ncc_enter_count + 1 if self.ncc_abs_ema >= config.NCC_ENTER_THRESHOLD else 0
            if self.ncc_enter_count >= config.NCC_ENTER_FRAMES:
                self.ncc_lock, self.ncc_exit_count = True, 0
        else:
            self.ncc_exit_count = self.ncc_exit_count + 1 if self.ncc_abs_ema <= config.NCC_EXIT_THRESHOLD else 0
            if self.ncc_exit_count >= config.NCC_EXIT_FRAMES:
                self.ncc_lock, self.ncc_enter_count = False, 0

    def _update_bit_threshold(self, abs_ncc: float) -> None:
        if self.bit_ncc_noise_ema <= 1e-9:
            self.bit_ncc_noise_ema = abs_ncc
        else:
            self.bit_ncc_noise_ema = (
                (1.0 - config.BIT_NCC_NOISE_ALPHA) * self.bit_ncc_noise_ema
                + config.BIT_NCC_NOISE_ALPHA * abs_ncc
            )
        if config.ADAPTIVE_BIT_NCC_THRESHOLD:
            self.bit_ncc_threshold = float(np.clip(
                self.bit_ncc_noise_ema + config.BIT_NCC_MARGIN,
                config.BIT_NCC_THRESHOLD_MIN, config.BIT_NCC_THRESHOLD_MAX,
            ))
        else:
            self.bit_ncc_threshold = float(config.BIT_NCC_THRESHOLD)

    def _slice_chips_for_phase(self, state: PhaseState, peak_hz: float) -> int:
        """Consume the phase sample buffer for this phase, returning how many chips were added."""
        added = 0
        while state.next_sample + self.bit_samples <= len(self.phase_sample_buffer):
            chunk = self.phase_sample_buffer[state.next_sample : state.next_sample + self.bit_samples]
            bit_ncc, _ = coherent_ncc(chunk, peak_hz, config.SUBCARRIER_HZ, config.SAMPLE_RATE)
            abs_ncc = abs(bit_ncc)
            self._update_bit_threshold(abs_ncc)
            state.chips.append(1 if abs_ncc >= self.bit_ncc_threshold else 0)
            state.next_sample += self.bit_samples
            state.chips_seen += 1
            added += 1

        # Cap history length
        max_chips = config.PHASE_HISTORY_BITS * config.REPETITION_CHIPS
        if len(state.chips) > max_chips:
            drop = len(state.chips) - max_chips
            del state.chips[:drop]
            state.base_chip_index += drop
        return added

    def _try_decode_packets(self, phase: int, state: PhaseState) -> None:
        """Scan all chip-offsets for header matches and emit packets on detection."""
        for decode_offset in range(config.REPETITION_CHIPS):
            decoded_bits = majority_decode_triplets(state.chips, decode_offset)
            base_bit_index = (state.base_chip_index + decode_offset) // config.REPETITION_CHIPS

            while True:
                search_start = state.search_start_by_offset[decode_offset]
                header_idx, header_errors = find_header_match(
                    decoded_bits, self.packet_header_bits, search_start, config.HEADER_MAX_BIT_ERRORS
                )
                if header_idx < 0:
                    state.search_start_by_offset[decode_offset] = max(
                        0, len(decoded_bits) - len(self.packet_header_bits) + 1
                    )
                    break

                header_abs = base_bit_index + header_idx
                if header_abs <= state.last_header_abs_by_offset[decode_offset]:
                    state.search_start_by_offset[decode_offset] = header_idx + 1
                    continue

                payload_start = header_idx + len(self.packet_header_bits)
                payload_end = payload_start + self.payload_bits_len
                if payload_end > len(decoded_bits):
                    state.search_start_by_offset[decode_offset] = header_idx
                    break

                payload = bits_to_bytes(decoded_bits[payload_start:payload_end])
                payload_hex = payload.hex().upper() if payload else ""
                print(
                    f"[RX HEADER] phase={phase} chip_offset={decode_offset} "
                    f"bit={header_abs} header_errors={header_errors} "
                    f"payload_hex={payload_hex} payload_ascii={safe_ascii(payload)!r}",
                    flush=True,
                )

                if payload == config.PAYLOAD_BYTES:
                    self.decoded_packets += 1
                    self.packet_status_text = (
                        f"Packet: OPEN detected at phase {phase}, offset {decode_offset}"
                    )
                    print(
                        f"[RX PACKET {self.decoded_packets}] phase={phase} chip_offset={decode_offset} "
                        f"bit={header_abs} header_errors={header_errors} PREAMBLE+SYNC+OPEN",
                        flush=True,
                    )
                else:
                    self.packet_status_text = (
                        f"Packet: header phase {phase}, offset {decode_offset}, "
                        f"errors={header_errors}, payload={payload!r}"
                    )

                self.packet_status_hold = config.PACKET_STATUS_HOLD_FRAMES
                state.last_header_abs_by_offset[decode_offset] = header_abs
                state.search_start_by_offset[decode_offset] = header_idx + 1

    def _emit_debug(self) -> None:
        best_phase = max(self.phase_offsets, key=lambda p: self.phase_state[p].chips_seen)
        st = self.phase_state[best_phase]
        best_decoded = majority_decode_triplets(st.chips, 0)
        chip_tail = bits_to_text(st.chips[-config.TERMINAL_DEBUG_BIT_TAIL:])
        bit_tail = bits_to_text(best_decoded[-(config.TERMINAL_DEBUG_BIT_TAIL // config.REPETITION_CHIPS):])
        print(
            f"[RX DEBUG] phase={best_phase} chips={st.chips_seen} "
            f"ncc_ema={self.ncc_abs_ema:.3f} lock={'1' if self.ncc_lock else '0'} "
            f"bit_thr={self.bit_ncc_threshold:.3f} noise_ema={self.bit_ncc_noise_ema:.3f} "
            f"chip_tail={chip_tail} bit_tail={bit_tail}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Per-frame callback
    # ------------------------------------------------------------------

    def update(self, _frame: int) -> tuple:
        bb, cw, nw = self.bb, self.cw, self.nw
        artists = (
            bb.line_i, bb.line_q, bb.line_fft_raw, bb.line_fft_dc_blocked,
            bb.exciter_marker, cw.line_centered, cw.waterfall_img, bb.status, cw.status,
        )
        if self.stop_requested:
            return artists

        # ---- Acquire + DC-block --------------------------------------------
        x_raw = normalize_iq(self.sdr.rx())
        x_dc, self.dc_prev_x, self.dc_prev_y = dc_block_filter(
            x_raw, self.dc_prev_x, self.dc_prev_y, config.DC_BLOCK_ALPHA
        )
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
        bb.exciter_marker.set_xdata([peak_hz / 1000.0, peak_hz / 1000.0])

        # ---- Baseband spectrum plot ----------------------------------------
        freqs_khz = freqs_hz[in_view] / 1000.0
        bb.line_fft_raw.set_data(freqs_khz, self.smoothed_raw[in_view])
        bb.line_fft_dc_blocked.set_data(freqs_khz, self.smoothed_dc[in_view])

        dc_idx = int(np.argmin(np.abs(freqs_hz)))
        dc_level = float(self.smoothed_raw[dc_idx])
        dc_to_carrier_db = peak_dbfs - dc_level

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

        # ---- Sideband markers ----------------------------------------------
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
            f"Carrier={peak_hz:,.1f} Hz | Span=±{config.CENTERED_SPAN_HZ/1000.0:.0f} kHz | "
            f"SB SNR={snr_db:+.1f} dB (+{config.SIDEBAND_OFFSET_KHZ:.0f}kHz={sb_pos:.1f} "
            f"-{config.SIDEBAND_OFFSET_KHZ:.0f}kHz={sb_neg:.1f} Noise={noise_floor_sb:.1f} dBFS)"
        )
        cw.fig.canvas.draw_idle()

        # ---- Coherent demod + packet decode --------------------------------
        if peak_hz != 0.0:
            demod = bandpass_filter_around_carrier(
                x_dc, peak_hz, config.SAMPLE_RATE, config.DEMOD_FILTER_PASSBAND_HZ
            )
            ncc_val, envelope = coherent_ncc(demod, peak_hz, config.SUBCARRIER_HZ, config.SAMPLE_RATE)
            nw.ncc_history = np.roll(nw.ncc_history, -1)
            nw.ncc_history[-1] = ncc_val
            nw.line_ncc.set_ydata(nw.ncc_history)

            env_show = smooth_1d(envelope[: nw.env_plot_len], config.DEMOD_ENV_SMOOTH_SAMPLES)
            nw.line_env.set_ydata(env_show)

            # Envelope autoscale (smoothed)
            lo_p, hi_p = np.percentile(env_show, [2.0, 98.0])
            mid, span = 0.5 * (lo_p + hi_p), max(hi_p - lo_p, config.ENV_Y_MIN_SPAN)
            self.env_ylim_low = (1.0 - config.ENV_Y_SMOOTH_ALPHA) * self.env_ylim_low + config.ENV_Y_SMOOTH_ALPHA * (mid - 0.6 * span)
            self.env_ylim_high = (1.0 - config.ENV_Y_SMOOTH_ALPHA) * self.env_ylim_high + config.ENV_Y_SMOOTH_ALPHA * (mid + 0.6 * span)
            nw.ax_env.set_ylim(self.env_ylim_low, self.env_ylim_high)

            self._update_lock_hysteresis(abs(ncc_val))

            # Multi-phase chip slicing + packet decode
            self.phase_sample_buffer = np.concatenate((self.phase_sample_buffer, demod))
            chips_added = 0
            for phase, state in self.phase_state.items():
                chips_added += self._slice_chips_for_phase(state, peak_hz)
                self._try_decode_packets(phase, state)

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

            nw.ncc_status.set_text(
                f"NCC={ncc_val:+.3f} | EMA={self.ncc_abs_ema:.3f} | "
                f"{'LOCKED' if self.ncc_lock else 'searching'} | "
                f"Carrier={peak_hz:,.0f} Hz | {self.packet_status_text}"
            )
            nw.fig.canvas.draw_idle()

        return artists
