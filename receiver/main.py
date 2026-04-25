#!/usr/bin/env python3
"""Orchestrator — initialises the SDR, builds the GUI windows, and runs the animation.

Run as a module from the repo root:
    python -m receiver.main
"""

import signal
import sys
from types import FrameType

import adi
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from . import config
from .gui_setup import setup_baseband_window, setup_carrier_window, setup_ncc_window
from .receiver_loop import ReceiverLoop


def main() -> int:
    # ---- SDR initialisation -------------------------------------------------
    sdr = adi.Pluto(config.RX_URI)
    sdr.rx_enabled_channels = [0]
    sdr.sample_rate = int(config.SAMPLE_RATE)
    sdr.rx_lo = int(config.FREQ_HZ)
    sdr.rx_rf_bandwidth = int(config.SAMPLE_RATE)  # full 1 MHz — let sideband spread through
    sdr.rx_buffer_size = int(config.RX_BUFFER_SIZE)
    sdr.gain_control_mode_chan0 = config.RX_GAIN_MODE
    if config.RX_GAIN_MODE == "manual":
        sdr.rx_hardwaregain_chan0 = float(config.RX_GAIN_DB)

    # ---- GUI windows --------------------------------------------------------
    bb_win = setup_baseband_window()
    carrier_win = setup_carrier_window()
    ncc_win = setup_ncc_window()

    # ---- Receiver loop ------------------------------------------------------
    loop = ReceiverLoop(sdr, bb_win, carrier_win, ncc_win)

    # ---- Signal / close handlers --------------------------------------------
    def _request_stop(_sig: int, _frame: FrameType | None) -> None:
        loop.stop_requested = True
        plt.close("all")

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    def _close_event(_event: object) -> None:
        loop.stop_requested = True

    bb_win.fig.canvas.mpl_connect("close_event", _close_event)
    carrier_win.fig.canvas.mpl_connect("close_event", _close_event)
    ncc_win.fig.canvas.mpl_connect("close_event", _close_event)

    # ---- Start animation ----------------------------------------------------
    animation = FuncAnimation(
        bb_win.fig, loop.update, interval=100, blit=False, cache_frame_data=False
    )

    print("Starting Pluto GUI receiver...")
    print(f"  RX URI    : {config.RX_URI}")
    print(f"  Freq (Hz) : {int(config.FREQ_HZ)}")
    print(f"  SR (SPS)  : {int(config.SAMPLE_RATE)}")
    print(f"  Gain mode : {config.RX_GAIN_MODE}")
    if config.RX_GAIN_MODE == "manual":
        print(f"  RX gain   : {config.RX_GAIN_DB} dB")

    bb_win.fig.tight_layout()
    carrier_win.fig.tight_layout()
    _ = animation  # keep reference alive for the event loop
    plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
