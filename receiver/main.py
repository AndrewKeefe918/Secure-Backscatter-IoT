#!/usr/bin/env python3
"""Orchestrator — initialises the SDR, builds the GUI windows, and runs the animation.

Run as a module from the repo root:
    python -m receiver.main
"""

import signal
import sys
from dataclasses import dataclass, field
from typing import Any, Protocol

import adi
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from . import config
from .cfo import SatoriCfoCorrector
from .gui_setup import setup_baseband_window, setup_carrier_window, setup_ncc_window
from .lock import LockPolicyState
from .packet import PacketCandidate, bytes_to_bit_list
from .flow import process_frontend_frame, process_demod_frame


class SdrLike(Protocol):
    """Minimal SDR interface required by the receiver runtime."""

    def rx(self) -> object:
        ...


@dataclass
class PhaseState:
    next_sample: int
    chips: list[int] = field(default_factory=lambda: list[int]())
    base_chip_index: int = 0
    search_start_by_offset: list[int] = field(
        default_factory=lambda: [0] * config.REPETITION_CHIPS)
    last_header_abs_by_offset: list[int] = field(
        default_factory=lambda: [-1] * config.REPETITION_CHIPS)
    last_logged_header_abs_by_offset: list[int] = field(
        default_factory=lambda: [-1] * config.REPETITION_CHIPS)
    chips_seen: int = 0
    bit_ncc_noise_ema: float = 0.0
    bit_ncc_threshold: float = float(config.BIT_NCC_THRESHOLD)
    bit_ncc_low_ema: float = 0.0
    bit_ncc_high_ema: float = float(config.BIT_NCC_THRESHOLD)
    timing_error_ema: float = 0.0
    chip_stride_samples: float = float(config.SAMPLES_PER_CHIP)
    chip_state: int = -1
    chip_rise_streak: int = 0
    chip_fall_streak: int = 0


class ReceiverRuntime:
    """Mutable per-frame receiver state and animation callback."""

    def __init__(self, sdr: SdrLike, bb: Any, cw: Any, nw: Any) -> None:
        self.sdr, self.bb, self.cw, self.nw = sdr, bb, cw, nw
        self.stop_requested = False

        self.smoothed_raw = np.array([], dtype=np.float64)
        self.smoothed_dc = np.array([], dtype=np.float64)
        self.dc_prev_x = np.complex64(0.0 + 0.0j)
        self.dc_prev_y = np.complex64(0.0 + 0.0j)
        self.prev_peak_hz = 0.0

        self.bit_samples = config.SAMPLES_PER_CHIP
        self.phase_offsets = list(range(0, self.bit_samples, config.BIT_PHASE_STEP_SAMPLES))
        self.phase_sample_buffer = np.array([], dtype=np.complex64)
        self.phase_state = {p: PhaseState(next_sample=p) for p in self.phase_offsets}
        self.preamble_bits = bytes_to_bit_list(config.PREAMBLE_BYTES)
        self.packet_header_bits = bytes_to_bit_list(config.PREAMBLE_BYTES + config.SYNC_BYTES)
        self.payload_bits_len = int(config.PAYLOAD_LENGTH_BYTES) * 8
        mode = config.PAYLOAD_MATCH_MODE.strip().lower()
        if mode not in ("off", "expected"):
            raise ValueError("PAYLOAD_MATCH_MODE must be 'off' or 'expected'")
        if mode == "expected":
            if not config.EXPECTED_PAYLOADS:
                raise ValueError("EXPECTED_PAYLOADS must be non-empty when PAYLOAD_MATCH_MODE='expected'")
            if any(len(payload) != config.PAYLOAD_LENGTH_BYTES for payload in config.EXPECTED_PAYLOADS):
                raise ValueError("Each EXPECTED_PAYLOADS entry must match PAYLOAD_LENGTH_BYTES")
            self.payload_patterns_bits = [bytes_to_bit_list(payload) for payload in config.EXPECTED_PAYLOADS]
        else:
            self.payload_patterns_bits = []
        self.packet_status_text = config.packet_status_default_text()
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
        self.ncc_force_unlock_streak = 0
        self.decode_lock_grace_chips = 0
        self.lock_policy = LockPolicyState()
        self.cfo = SatoriCfoCorrector(sample_rate=config.SAMPLE_RATE)

        plot_len = self.nw.env_plot_len
        self.display_phase = self.phase_offsets[0] if self.phase_offsets else 0
        self.env_chip_metric_history_by_phase = {
            p: np.zeros(plot_len, dtype=np.float64) for p in self.phase_offsets}
        self.env_chip_decision_history_by_phase = {
            p: np.zeros(plot_len, dtype=np.float64) for p in self.phase_offsets}
        self.phase_score_summary = ""

    def update(self, _frame: int) -> tuple[Any, ...]:
        bb, cw = self.bb, self.cw
        artists = (bb.line_i, bb.line_q, bb.line_fft_raw, bb.line_fft_dc_blocked,
                   bb.exciter_marker, cw.line_centered, cw.waterfall_img, bb.status, cw.status)
        if self.stop_requested:
            return artists

        x_focus, peak_hz = process_frontend_frame(self)
        if peak_hz != 0.0:
            process_demod_frame(self, x_focus, peak_hz)
        return artists


def main() -> int:
    # ---- SDR initialisation -------------------------------------------------
    sdr = adi.Pluto(config.RX_URI)
    sdr.rx_enabled_channels = [0]
    sdr.sample_rate = int(config.SAMPLE_RATE)
    sdr.rx_lo = int(config.FREQ_HZ)
    sdr.rx_rf_bandwidth = int(config.RX_RF_BANDWIDTH)
    sdr.rx_buffer_size = int(config.RX_BUFFER_SIZE)
    sdr.gain_control_mode_chan0 = config.RX_GAIN_MODE
    if config.RX_GAIN_MODE == "manual":
        sdr.rx_hardwaregain_chan0 = float(config.RX_GAIN_DB)

    # ---- GUI windows --------------------------------------------------------
    bb_win = setup_baseband_window()
    carrier_win = setup_carrier_window()
    ncc_win = setup_ncc_window()

    # ---- Receiver runtime ---------------------------------------------------
    loop = ReceiverRuntime(sdr, bb_win, carrier_win, ncc_win)

    # ---- Stop handlers (signals + window close) -----------------------------
    def _stop(*_args: object) -> None:
        loop.stop_requested = True

    def _stop_and_close(*_args: object) -> None:
        _stop()
        plt.close("all")

    signal.signal(signal.SIGINT, _stop_and_close)
    signal.signal(signal.SIGTERM, _stop_and_close)
    for win in (bb_win, carrier_win, ncc_win):
        win.fig.canvas.mpl_connect("close_event", _stop)

    # ---- Start animation ----------------------------------------------------
    animation = FuncAnimation(
        bb_win.fig,
        loop.update,
        interval=config.ANIMATION_INTERVAL_MS,
        blit=False,
        cache_frame_data=False,
    )

    print("Starting Pluto GUI receiver...")
    print(f"  RX URI    : {config.RX_URI}")
    print(f"  Freq (Hz) : {int(config.FREQ_HZ)}")
    print(f"  SR (SPS)  : {int(config.SAMPLE_RATE)}")
    print(f"  Gain mode : {config.RX_GAIN_MODE}")
    if config.RX_GAIN_MODE == "manual":
        print(f"  RX gain   : {config.RX_GAIN_DB} dB")
    if config.CONTINUOUS_ON_TEST:
        print("  Mode      : Continuous ON test (packet decode disabled)")

    bb_win.fig.tight_layout()
    carrier_win.fig.tight_layout()
    _ = animation  # keep reference alive for the event loop
    try:
        plt.show()
    finally:
        try:
            if hasattr(sdr, "rx_destroy_buffer"):
                sdr.rx_destroy_buffer()
        except Exception:
            pass
        try:
            del sdr
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
