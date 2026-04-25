# Receiver (FSK)

This receiver uses binary FSK backscatter modulation:
- bit `1` at 1.0 kHz subcarrier
- bit `0` at 1.7 kHz subcarrier

Run from repo root:

```bash
python -m receiver.main
```

Optional utilities:

```bash
python -m receiver.rx_monitor
```

## What differs from OOK

- **Chip decision**: differential metric (`m_f1` vs `m_f0`) instead of absolute thresholding.
- **Tone separation**: 1.0 kHz and 1.7 kHz are non-harmonic, reducing overlap.
- **Metrics view**: GUI panel shows both per-chip tone metrics and decision overlay.

## MSP firmware

The matching MSP430 firmware is in the chat — it drives Timer_A0 with two
different CCR0 values per bit:
- `CCR0 = 499` for the '1' bit (1 kHz subcarrier)
- `CCR0 = 293` for the '0' bit (1.7 kHz subcarrier)
- Different tick counts per bit (100 for '1', 170 for '0') keep
  wall-clock bit duration constant at 50 ms.

## Tuning

If chips are mostly being decided as 0 (FSK_F0_HZ wins), increase
`FSK_DECISION_DEAD_ZONE` to require a stronger '1' margin, OR check that
your tag is producing distinguishable signal at both 1.0 kHz and 1.7 kHz
on the spectrum view. Both subcarrier markers (lime for 1 kHz, orange for
1.7 kHz) are shown in the carrier-detail and waterfall plots so you can
verify visually before debugging the decoder.
