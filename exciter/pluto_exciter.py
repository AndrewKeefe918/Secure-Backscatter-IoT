#!/usr/bin/env python3
"""PlutoSDR backscatter exciter (CW transmitter) — runs on Raspberry Pi.

Hardware topology
-----------------
  Exciter  : PlutoSDR on Raspberry Pi  (this script)
  Receiver : PlutoSDR on PC            (pluto_receiver_gui.py)
  Tag      : MSP430 EXP-G2ET eval board (backscatter_1khz.c)

The exciter transmits a continuous-wave tone at FREQ_HZ + TONE_HZ.
The tag load-modulates the reflected carrier at 1 kHz, producing
sidebands at ±1 kHz around the exciter carrier as seen by the receiver.

TX gain guidance (separate hardware, no self-interference)
----------------------------------------------------------
  TX_GAIN_DB = -10  → ~−3 dBm output. Good starting point at 0.5–2 m.
  TX_GAIN_DB =   0  → ~+7 dBm output. Max Pluto output; use with
                       attenuator or at > 1 m to avoid tag overdrive.
  Step up in 5 dB increments and watch the receiver SNR readout.

Frequency relationship
----------------------
  Both Plutos must be tuned to the same FREQ_HZ.
  TONE_HZ offsets the exciter from its own LO; the receiver will see
  the reflected carrier at that same offset from its LO.
  Keep TONE_HZ > 3000 Hz so the receiver's exciter search can find it
  (EXCITER_SEARCH_MIN_HZ = 3000 in pluto_receiver_gui.py).

Use Ctrl+C to stop transmission.
"""

import signal
import sys
import time

import adi
import numpy as np

# --- Hardware addresses ---
# Exciter Pluto is accessed from the Raspberry Pi; edit the IP if yours differs.
URI = "ip:192.168.2.1"

# --- Frequency ---
# Must match FREQ_HZ in pluto_receiver_gui.py so both radios share the same LO.
FREQ_HZ = 2.48e9

SAMPLE_RATE = int(2e6)   # 2 MHz — matches original working exciter

# --- TX power ---
TX_GAIN_DB = -5.0        # matches original working exciter

# Raw IQ amplitude in DAC counts (int16 scale: 2^14 = 16384).
# The adi library passes these directly to libiio without normalisation.
IQ_AMPLITUDE = 2 ** 14

BUFFER_LEN = 1024        # matches original working exciter


def make_waveform(buffer_len: int, iq_amplitude: int) -> np.ndarray:
    """Constant IQ at iq_amplitude — pure CW at TX LO (DC in baseband)."""
    return np.ones(buffer_len, dtype=np.complex64) * np.complex64(iq_amplitude)


def main() -> int:
    sdr = adi.Pluto(URI)
    sdr.tx_enabled_channels = [0]
    sdr.sample_rate = int(SAMPLE_RATE)
    sdr.tx_lo = int(FREQ_HZ)
    sdr.tx_rf_bandwidth = int(SAMPLE_RATE)
    sdr.tx_hardwaregain_chan0 = float(TX_GAIN_DB)
    sdr.tx_cyclic_buffer = True

    waveform = make_waveform(BUFFER_LEN, IQ_AMPLITUDE)

    stop_requested = False

    def _request_stop(_sig: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    print("Starting Pluto exciter with:")
    print(f"  URI         : {URI}")
    print(f"  Freq (Hz)   : {int(FREQ_HZ)}")
    print(f"  SR (SPS)    : {int(SAMPLE_RATE)}")
    print(f"  TX Gain (dB): {TX_GAIN_DB}")
    print(f"  IQ Amplitude: {IQ_AMPLITUDE}")

    try:
        sdr.tx(waveform)
        print("Transmitting... Press Ctrl+C to stop.")
        while not stop_requested:
            time.sleep(0.2)
    finally:
        try:
            sdr.tx_destroy_buffer()
        except Exception:
            pass
        print("Transmission stopped.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
