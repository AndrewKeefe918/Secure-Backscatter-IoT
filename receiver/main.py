#!/usr/bin/env python3
"""Orchestrator -- initialises the SDR and runs the RX-only ingest loop.

The receiver hot path is intentionally kept free of any GUI work.
Run the decoupled monitor in a separate terminal for live visualisation:

    python -m receiver.rx_monitor

Run as a module from the repo root:
    python -m receiver.main
"""

import signal
import sys
from types import FrameType

import adi

from . import config as config


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

    # ---- Receiver loop (RX-only, no GUI on hot path) ------------------------
    from .receiver_loop_rx_only import ReceiverLoopRxOnly

    loop = ReceiverLoopRxOnly(sdr)

    # ---- Signal handlers ----------------------------------------------------
    def _request_stop(_sig: int, _frame: FrameType | None) -> None:
        loop.stop_requested = True

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    print("Starting Pluto FSK receiver (RX-only mode, SECURE: AES-128 + CMAC + counter)...")
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
    print(f"  Secure mode    : {config.SECURE_MODE}  (key={config.SHARED_KEY_HEX[:8]}...)")
    print(f"  Payload format : COUNTER(4) || AES-CTR ciphertext(4) || CMAC tag(8)  [{config.LIVE_DECODE_PAYLOAD_BYTES} bytes]")
    print(f"  Replay state   : {config.SECURE_RX_STATE_PATH}")
    print(f"  Capture output : {config.RX_CAPTURE_NDJSON}")
    print(f"  Status output  : {config.RX_STATUS_JSON}")
    print("  Monitor        : python -m receiver.rx_monitor  (separate terminal)")

    try:
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
