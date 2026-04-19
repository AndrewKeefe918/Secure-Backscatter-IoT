import adi
import numpy as np

print("Step 1: Connecting to Pluto...")
try:
    sdr = adi.Pluto("ip:192.168.2.1")
    print("  Connected via IP.")
except Exception as e:
    print(f"  IP connection failed: {e}")
    print("  Trying USB instead...")
    try:
        sdr = adi.Pluto("usb:")
        print("  Connected via USB.")
    except Exception as e2:
        print(f"  USB also failed: {e2}")
        exit(1)

print("\nStep 2: Configuring...")
sdr.sample_rate = 1_000_000
sdr.rx_lo = 2_480_000_000
sdr.rx_rf_bandwidth = 1_000_000
sdr.rx_buffer_size = 4096
sdr.gain_control_mode_chan0 = "manual"
sdr.rx_hardwaregain_chan0 = 40
print("  Config set.")

print("\nStep 3: Reading one buffer...")
try:
    samples = sdr.rx()
    print(f"  SUCCESS! Got {len(samples)} samples")
    print(f"  Sample dtype: {samples.dtype}")
    print(f"  Mean |sample|: {np.mean(np.abs(samples)):.2f}")
    print(f"  Max |sample|: {np.max(np.abs(samples)):.2f}")
except Exception as e:
    print(f"  READ FAILED: {e}")
    exit(1)

print("\nStep 4: Reading 10 more buffers to check stability...")
for i in range(10):
    try:
        samples = sdr.rx()
        print(f"  Buffer {i+1}: OK, {len(samples)} samples, mean={np.mean(np.abs(samples)):.1f}")
    except Exception as e:
        print(f"  Buffer {i+1}: FAILED: {e}")
        exit(1)

print("\nAll tests passed. Pluto is working correctly.")