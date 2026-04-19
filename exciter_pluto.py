"""
Exciter: park a CW tone at 2.48 GHz for the backscatter tag to reflect.
The receiver auto-locates the carrier, so exact TX LO only has to be close.
"""

import time

import adi
import numpy as np

TX_LO_HZ         = 2_480_000_000
TX_SAMPLE_RATE   = int(2e6)
TX_HARDWAREGAIN  = -5     # keep below 0 dB to avoid front-end overload
IQ_LEN           = 1024
IQ_AMPLITUDE     = 2 ** 14

try:
    sdr = adi.Pluto("ip:192.168.2.1")

    sdr.tx_lo                  = TX_LO_HZ
    sdr.tx_hardwaregain_chan0  = TX_HARDWAREGAIN
    sdr.sample_rate            = TX_SAMPLE_RATE
    sdr.tx_cyclic_buffer       = True

    # Constant IQ -> pure tone at TX_LO (DC in baseband avoids the
    # AD9361 DC-null servo by providing a real non-zero baseband).
    iq = np.ones(IQ_LEN, dtype=np.complex64) * IQ_AMPLITUDE
    sdr.tx(iq)

    print(f"[exciter] CW live at {TX_LO_HZ/1e9:.4f} GHz, gain {TX_HARDWAREGAIN} dB.")
    print("Press Ctrl+C to stop.")
    while True:
        time.sleep(1)

except KeyboardInterrupt:
    print("\n[exciter] stopping...")
    try:
        sdr.tx_destroy_buffer()
    except Exception:
        pass
    print("[exciter] done.")
except Exception as e:
    print(f"[exciter] error: {e}")