#!/usr/bin/env python3
"""Orchestrator — initialises the SDR, builds the GUI windows, and runs the animation.

Run as a module from the repo root:
    python -m Receiver_FSK_secure.main_secure

(or rename the files to drop the `_fsk` suffix and run the OOK entrypoint.)
"""

import signal
import sys
from types import FrameType

import adi
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from . import config_secure as config
from .gui_setup_secure import setup_baseband_window, setup_carrier_window, setup_fsk_window
from .receiver_loop_secure import ReceiverLoop
from .receiver_loop_rx_only_secure import ReceiverLoopRxOnly


def main() -> int:
    # ---- SDR initialisation -------------------------------------------------
    sdr = adi.Pluto(config.RX_URI)
    sdr.rx_enabled_channels = [0]
    sdr.sample_rate = int(config.SAMPLE_RATE)
    sdr.rx_lo = int(config.FREQ_HZ)
    sdr.rx_rf_bandwidth = int(config.SAMPLE_RATE)
    sdr.rx_buffer_size = int(config.RX_BUFFER_SIZE)
    sdr.gain_control_mode_chan0 = config.RX_GAIN_MODE
    if config.RX_GAIN_MODE == "manual":
        sdr.rx_hardwaregain_chan0 = float(config.RX_GAIN_DB)

    # ---- GUI windows --------------------------------------------------------
    enable_plots = config.RENDER_PLOTS and not config.RX_ONLY_MODE
    if enable_plots:
        bb_win = setup_baseband_window()
        carrier_win = setup_carrier_window()
        fsk_win = setup_fsk_window()
    else:
        bb_win = None
        carrier_win = None
        fsk_win = None

    # ---- Receiver loop ------------------------------------------------------
    if config.RX_ONLY_MODE:
        loop = ReceiverLoopRxOnly(sdr)
    else:
        loop = ReceiverLoop(sdr, bb_win, carrier_win, fsk_win)

    # ---- Signal / close handlers --------------------------------------------
    def _request_stop(_sig: int, _frame: FrameType | None) -> None:
        loop.stop_requested = True
        plt.close("all")

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    def _close_event(_event: object) -> None:
        loop.stop_requested = True

    if enable_plots:
        bb_win.fig.canvas.mpl_connect("close_event", _close_event)
        carrier_win.fig.canvas.mpl_connect("close_event", _close_event)
        fsk_win.fig.canvas.mpl_connect("close_event", _close_event)

    animation = None
    if enable_plots:
        # ---- Start animation ------------------------------------------------
        animation = FuncAnimation(
            bb_win.fig,
            loop.update,
            interval=int(config.ANIMATION_INTERVAL_MS),
            blit=False,
            cache_frame_data=False,
        )

    print("Starting Pluto FSK receiver (SECURE: AES-128 + CMAC + counter)...")
    print(f"  RX URI         : {config.RX_URI}")
    print(f"  Freq (Hz)      : {int(config.FREQ_HZ)}")
    print(f"  SR (SPS)       : {int(config.SAMPLE_RATE)}")
    print(f"  Gain mode      : {config.RX_GAIN_MODE}")
    if config.RX_GAIN_MODE == "manual":
        print(f"  RX gain        : {config.RX_GAIN_DB} dB")
    print(f"  '1' subcarrier : {config.FSK_F1_HZ:.1f} Hz")
    print(f"  '0' subcarrier : {config.FSK_F0_HZ:.1f} Hz")
    print(f"  Bit duration   : {config.BIT_DURATION_MS:.0f} ms")
    print(f"  Repetition     : {config.REPETITION_CHIPS} chip(s) per bit")
    if config.LIVE_DECODE_REQUIRE_KNOWN_PAYLOAD:
        print(f"  Expected packet: {config.PREAMBLE_BYTES.hex().upper()} "
              f"{config.SYNC_BYTES.hex().upper()} '{config.PAYLOAD_BYTES.decode()}'")
    else:
        print(f"  Packet detect   : header {config.PREAMBLE_BYTES.hex().upper()} {config.SYNC_BYTES.hex().upper()} + "
              f"{int(config.LIVE_DECODE_PAYLOAD_BYTES)} unknown payload byte(s)")

    if enable_plots:
        bb_win.fig.tight_layout()
        carrier_win.fig.tight_layout()
    _ = animation  # keep reference alive for the event loop
    try:
        if enable_plots:
            plt.show()
        else:
            print("  Mode           : low-overhead decode (no live plots)")
            if config.RX_ONLY_MODE:
                print(f"  Capture output : {config.RX_CAPTURE_NDJSON}")
            while not loop.stop_requested:
                loop.update(0)
    finally:
        loop.close()
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
