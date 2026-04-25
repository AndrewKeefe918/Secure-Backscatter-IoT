# `receiver/` — PlutoSDR backscatter receiver

Real-time receiver for the MSP430 OOK backscatter tag (1 kHz subcarrier, 20 bps,
no repetition coding). Pulls IQ from a PlutoSDR over USB/Ethernet, runs a
coherent demod + multi-phase chip slicer + Satori-inspired packet stage, and
renders three live Matplotlib windows.

Run from the repo root:

```powershell
python -m receiver.main
```

## Module layout

| Module | Role |
| --- | --- |
| [`main.py`](main.py) | Entry point plus `ReceiverRuntime` state/coordinator. Initialises SDR/GUI, wires signal handlers, and starts `FuncAnimation`. |
| [`flow.py`](flow.py) | Per-frame orchestration (`process_frontend_frame`, `process_demod_frame`): front-end DSP wiring, lock/decode gating, and immediate force-unlock guard. |
| [`config.py`](config.py) | All tunable constants (RF, FFT, chip timing, lock thresholds, packet framing, Satori CFO params). No magic numbers in the rest of the code. |
| [`dsp.py`](dsp.py) | Stateless DSP primitives: IQ normalisation, IIR DC block, spectrum + EMA in power domain, exciter peak finder, sideband SNR, coherent NCC / chip metric, bandpass filter, smoothing. |
| [`gui_setup.py`](gui_setup.py) | Builds the three Matplotlib windows and returns dataclass containers (`BasebandWindow`, `CarrierWindow`, `NccWindow`) of artist handles. |
| [`ui.py`](ui.py) | Per-frame artist updates: IQ panel (tuning circle + Satori triangle), chip-view, terminal debug emission. |
| [`chips.py`](chips.py) | Adaptive chip threshold, multi-phase 50 ms slicer, NCC lock hysteresis, and phase ranking helpers. |
| [`cfo.py`](cfo.py) | `SatoriCfoCorrector` — two-step residual CFO tracker (slow average + fast even-symbol track). |
| [`fading.py`](fading.py) | Triangle-geometry and weighting helpers (`fading_weight`, `weighted_voltage_recovery`) for Satori-style fading adaptation; currently utility-only (not wired into the live frame path). |
| [`packet.py`](packet.py) | Packet-stage codec/model/pipeline (`PacketCandidate`, preamble→sync→payload candidate scoring/settling), plus shared logical-bit downsampling helper (`majority_decode_triplets`) used by chips/lock/UI diagnostics. |
| [`lock.py`](lock.py) | Lock/watchdog policy module (structure-aware lock gating, quality veto/unlock, decode grace). |

## Data flow per frame

```
sdr.rx()
  → normalize_iq → dc_block_filter
  → compute_spectrum_dbfs (raw + DC-blocked, EMA in power domain)
  → find_exciter_peak  ──► update baseband + IQ panel + tuning circle/triangle
  → carrier-centered spectrum + waterfall + sideband SNR scatter
  → coherent_ncc(full frame)  ──► NCC history line
       ↓
  slice_chips_for_phase  (per phase offset)
       ↓
  update_lock_hysteresis (single shared NCC lock state)
       ↓
  try_decode_packets → accept_best_packet_candidate
       ↓
  update_chip_view + emit_chip_debug
```

## Three GUI windows

1. **Pluto Baseband Receiver** — time domain + IQ plane (tuning circle + Satori
   tuning triangle: ref carrier / +1 kHz / −1 kHz phasors) + wide spectrum.
2. **Pluto Carrier Detail** — carrier-centered spectrum with ±1 kHz sideband
   SNR markers, plus carrier-centered waterfall.
3. **Coherent 1 kHz Demodulator** — NCC history and per-chip |NCC| view with
   live chip-decision threshold.

## Why this split

- **`main.py`** is the app orchestrator and owns mutable runtime state.
- **`flow.py`** keeps per-frame orchestration in focused step functions and centralises demod/lock/decode control flow.
- **`dsp.py`** holds stateless math.
- **`chips.py`, `lock.py`, `packet.py`, `cfo.py`** hold stateful demod/lock/decode policy.
- **`gui_setup.py` + `ui.py`** isolate Matplotlib I/O from signal logic.
- **`config.py`** is the single source of tunables.
- **`fading.py`** is an isolated utility module for geometry-based fading experiments.

## Conventions

- One source of truth for constants → `config.py`.
- Stateful algorithms live in focused domain modules; stateless math lives in
  `dsp.py`.
- GUI artists are created in `gui_setup.py` and only mutated in
     `ui.py` / `flow.py`.
- The tag and exciter sides are intentionally dumb — adaptive logic
  (CFO tracking, fading weighting, packet candidate selection) lives here in
  Python, per the SyncScatter/Satori split documented in
  [`/memories/repo/backscatter_sync_concepts.md`](../../memories/repo/backscatter_sync_concepts.md).
