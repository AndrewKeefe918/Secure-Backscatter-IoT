import adi
import numpy as np
import time

try:
    sdr = adi.Pluto("ip:192.168.2.1")
    
    # Exciter configuration
    sdr.tx_lo = int(2480000000)
    sdr.tx_hardwaregain_chan0 = -10   # NOT zero - start lower to avoid overload
    sdr.sample_rate = int(2e6)
    sdr.tx_cyclic_buffer = True
    
    # True CW: zero baseband signal → pure LO leakage (the carrier)
    # Actually, we want a non-zero DC offset so the AD9361 doesn't null it
    # Use a modest-amplitude constant
    N = 1024
    iq = np.ones(N, dtype=np.complex64) * (2**14)   # constant I, constant Q
    
    sdr.tx(iq)
    
    print("\n[SUCCESS] Exciter is LIVE at 2.48 GHz (CW)")
    print("Press Ctrl+C to stop.\n")
    
    while True:
        time.sleep(1)

except KeyboardInterrupt:
    print("\nShutting down exciter...")
    sdr.tx_destroy_buffer()   # Critical - stops transmission cleanly
    print("Done.")
except Exception as e:
    print(f"\n[ERROR] Could not start Exciter: {e}")