# FSK Receiver — Drop-in Files

These six Python files are the FSK-modulation version of your backscatter
receiver. They are organised as a parallel set to your OOK files, all
suffixed with `_fsk`. Pick ONE of the two installation styles below.

## What's different from OOK

- **Modulation**: '1' bits transmit at 1.0 kHz subcarrier, '0' bits at 1.7 kHz.
  Both frequencies are non-harmonic (1 kHz harmonics: 3, 5, 7, 9 kHz; 1.7
  kHz harmonics: 5.1, 8.5 kHz — no overlap), so the receiver can cleanly
  measure power at each frequency.
- **Chip decision**: differential, `m_f1 vs m_f0`, instead of absolute power
  threshold. No threshold drift, no per-frame threshold tuning.
- **Repetition coding**: defaults to 1 (disabled). FSK doesn't suffer from
  the threshold-drift correlated-error problem that hurt OOK-with-repetition.
  You can re-enable by setting `REPETITION_CHIPS = 3` in `config_fsk.py`.
- **GUI third panel**: replaced single |NCC| trace with two metric traces
  (`m_f1` for '1' tone, `m_f0` for '0' tone) plus chip decision overlay.

## Option A: Side-by-side (recommended)

Drop these files into your `receiver/` package directory alongside the OOK
files. Run with:

```bash
python -m receiver.main_fsk
```

Your existing `python -m receiver.main` (OOK) keeps working unchanged.

To make this work you need to give the FSK files an `__init__` entry — but
since `__init__.py` is shared at the package level, just keep your existing
`__init__.py` and ignore `__init__fsk.py` (it's only here so the package
description is preserved).

## Option B: Drop-in replacement

If you want FSK to BE the receiver, rename these files:

```
__init__fsk.py        -> __init__.py    (or merge with existing)
config_fsk.py         -> config.py
dsp_fsk.py            -> dsp.py
packet_decoder_fsk.py -> packet_decoder.py
gui_setup_fsk.py      -> gui_setup.py
receiver_loop_fsk.py  -> receiver_loop.py
main_fsk.py           -> main.py
```

Then go through each file and change every `from . import config_fsk as config`
back to `from . import config`, and similarly for the other module imports.
This removes all the `_fsk` suffixes from the imports.

After that, run with `python -m receiver.main` like before.

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
