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
    Keep TONE_HZ well above the receiver's non-DC search floor.

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

SAMPLE_RATE = int(1e6)   # 1 MHz — must match receiver SAMPLE_RATE in receiver/config.py

# --- TX power ---
TX_GAIN_DB = -15.0         # step up from -15 for stronger backscatter

# Raw IQ amplitude in DAC counts (int16 scale: 2^14 = 16384).
# The adi library passes these directly to libiio without normalisation.
IQ_AMPLITUDE = 2 ** 14

BUFFER_LEN = 4096
# TONE_BIN chosen so TONE_HZ = SAMPLE_RATE * TONE_BIN / BUFFER_LEN = 15625 Hz
# at fs = 1 MS/s. Must equal EXCITER_EXPECTED_HZ in receiver/config.py.
TONE_BIN = 64
TONE_HZ = SAMPLE_RATE * TONE_BIN / BUFFER_LEN  # 15625 Hz, coherent with BUFFER_LEN and clear of DC leakage
MIN_NON_DC_OFFSET_HZ = 12000.0


def make_waveform(buffer_len: int, iq_amplitude: int, tone_hz: float, sample_rate: float) -> np.ndarray:
    """Generate a cyclic complex tone offset from LO to avoid the DC bin."""
    n = np.arange(buffer_len, dtype=np.float32)
    phase = (2.0 * np.pi * float(tone_hz) / float(sample_rate)) * n
    waveform = np.exp(1j * phase).astype(np.complex64)
    return waveform * np.complex64(iq_amplitude)


def main() -> int:
    if TONE_HZ < MIN_NON_DC_OFFSET_HZ:
        raise ValueError(
            f"TONE_HZ={TONE_HZ:.1f} must be >= {MIN_NON_DC_OFFSET_HZ:.1f} Hz to stay out of the DC bin"
        )

    sdr = adi.Pluto(URI)
    sdr.tx_enabled_channels = [0]
    sdr.sample_rate = int(SAMPLE_RATE)
    sdr.tx_lo = int(FREQ_HZ)
    sdr.tx_rf_bandwidth = int(SAMPLE_RATE)
    sdr.tx_hardwaregain_chan0 = float(TX_GAIN_DB)
    sdr.tx_cyclic_buffer = True

    waveform = make_waveform(BUFFER_LEN, IQ_AMPLITUDE, TONE_HZ, SAMPLE_RATE)

    stop_requested = False

    def _request_stop(_sig: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    print("Starting Pluto exciter with:")
    print(f"  URI         : {URI}")
    print(f"  Freq (Hz)   : {int(FREQ_HZ)}")
    print(f"  Tone (Hz)   : {TONE_HZ:.1f}")
    print(f"  RF Out (Hz) : {int(FREQ_HZ + TONE_HZ)}")
    print(f"  SR (SPS)    : {int(SAMPLE_RATE)}")
    print(f"  Buffer Len  : {BUFFER_LEN}")
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
