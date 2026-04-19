# Secure-Backscatter-IoT

A minimal backscatter-style RF link used as a testbed for "secure IoT" experiments.
An MSP430G2553 LaunchPad modulates a 1 kHz subcarrier onto an RF carrier produced by
an ADALM-Pluto SDR acting as a CW exciter. A second Pluto (or the same one in RX mode)
demodulates the sidebands and decodes ASCII packets.

```
 [ Pluto TX ]  2.480 GHz CW  ─►  [ RF gate driven by MSP430 P1.1 @ 1 kHz OOK ]  ─►  [ Pluto RX ]
    exciter_pluto.py                    MSP_OOK_Transmit.c                            Receiver.py
```

## Current status (April 2026)

- **Exciter (Pluto TX):** stable CW at 2.480 GHz via `exciter_pluto.py`.
- **Transmitter (MSP430):** emits clean `0xAA 0x7E 'O' 'P' 'E' 'N'` packets.
  Timer_A0 runs free at 1 kHz; bits gate the subcarrier only by (dis)connecting the
  pin via `P1SEL`. This fixed the earlier broadband splatter caused by restarting the
  timer phase every bit.
- **Receiver:** `Receiver.py` decodes packets via a matched filter on the sideband
  envelope, locking bit timing to the known preamble+sync. `Receiver_Plot.py` is a
  spectrum/waterfall-only view with no decoder, used for alignment.
- **Known physics (not bugs):** the 1 kHz *square* subcarrier produces odd-harmonic
  spurs at carrier ± 3 / 5 / 7 kHz. An RC LPF on P1.1 (or a sine subcarrier) would
  remove them.

## Signal chain

| Layer | Value | Produced by |
| --- | --- | --- |
| RF carrier | 2.480 GHz | Pluto TX (`exciter_pluto.py`) |
| Subcarrier | 1 kHz square | MSP430 Timer_A0 on P1.1 |
| Modulation | OOK, 50 ms/bit (20 bps) | MSP430 gating P1.1 via `P1SEL` |
| Packet | `0xAA 0x7E` + 4 ASCII bytes (48 bits, MSB first) | `MSP_OOK_Transmit.c` |
| Decode | matched filter on sideband-power envelope | `Receiver.py` |

## Files

### Python (host / SDR)

- **`exciter_pluto.py`** – Drives a Pluto as a CW exciter at 2.480 GHz
  (constant I/Q so the AD9361 doesn't null the LO).
- **`Receiver.py`** – Full receiver. Captures IQ at 1 MS/s, computes a 2 ms
  sideband-power envelope at carrier ± 1 kHz, cross-correlates against the known
  preamble+sync template to lock bit timing, then samples the 4 payload bytes.
  Plots wide/zoom spectrum, waterfall, and the envelope with decode status.
- **`Receiver_Plot.py`** – Spectrum and waterfall only (wide ±50 kHz and zoomed ±5 kHz).
  No decoder. Useful for tuning and sanity-checking sideband presence.
- **`test_pluto.py`** – Connectivity smoke test: connects, configures, reads a few
  buffers, prints stats. Run this first if anything seems wrong.

### MSP430 firmware (`Blink LED/`)

CCS project; only one `main`/source file should be built into the target at a time.

- **`MSP_OOK_Transmit.c`** – **Production transmitter.** Sends
  `0xAA 0x7E 'O' 'P' 'E' 'N'` every 2 s at 50 ms/bit. Timer_A0 is started once and
  never stopped; bits are keyed only by `P1SEL |= / &= ~BIT1` to keep subcarrier phase
  continuous (the fix for the "strange sidebands").
- **`carrier_verify.c`** – Hardware sanity check: continuous 1 kHz square on P1.1 with
  the red LED solid. Use to verify wiring and SDR alignment.
- **`Sidebands_Test.c`** – Like `carrier_verify.c` but with the full DCO calibration
  and `P1SEL2` setup used by the transmitter. Produces a clean, always-on ±1 kHz
  sideband pair for receiver calibration.
- **`Message_verify.c`** – Older message transmitter (sends `0x4F70656E` / "Open" at
  100 ms/bit, no preamble or sync). Kept for reference; not compatible with
  `Receiver.py`'s decoder.
- **`main.c`** – Earliest prototype: reed switch on P1.3, toggles P1.1 in a delay
  loop while a magnet is present. Not a real modulator; kept for bring-up.
- **`lnk_msp430g2553.cmd`** – Linker command file for the MSP430G2553.
- **`Debug/`** – CCS build outputs (auto-generated).
- **`targetConfigs/MSP430G2553.ccxml`** – CCS target configuration for the LaunchPad.

## Packet format

```
bit: 0        8        16       24       32       40       47
     [0xAA  ][0x7E  ][  'O'  ][  'P'  ][  'E'  ][  'N'  ]
     preamble  sync    payload (4 ASCII bytes, MSB first)
```

- Bit `1` = subcarrier on, bit `0` = subcarrier off.
- 50 ms per bit ⇒ one packet is 48 × 50 ms = 2.4 s, followed by a ~2 s idle gap.

## How to run

1. Flash `Blink LED/MSP_OOK_Transmit.c` onto the MSP430G2553 LaunchPad with CCS.
2. Wire P1.1 into the RF switch / backscatter gate.
3. `python exciter_pluto.py` (first Pluto, TX-only at 2.480 GHz).
4. `python Receiver.py` (second Pluto, RX). Watch the console for
   `PACKET #N DECODED: "OPEN" (corr=...)`.

Use `Receiver_Plot.py` if you just want to see whether the ±1 kHz sidebands are
present before worrying about decoding.