#!/usr/bin/env python3
"""Decoupled live monitor for the RX-only FSK loop.

Run in a separate terminal:
    python -m receiver.rx_monitor
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from . import config as config
from .live_decode import analyze_live_decode
from .packet_decoder import (
    bits_to_text,
    bytes_to_bit_list,
)


class MonitorState:
    def __init__(self, history_len: int = 180) -> None:
        self.history_len = history_len
        self.frames = deque(maxlen=history_len)
        self.rx_ms = deque(maxlen=history_len)
        self.proc_ms = deque(maxlen=history_len)
        self.gap_ms = deque(maxlen=history_len)
        self.cfo_hz = deque(maxlen=history_len)
        self.ncc = deque(maxlen=history_len)
        self.snr_db: deque[float] = deque(maxlen=history_len)
        self.locked: bool = False
        self.monitor_axis_khz = np.array([], dtype=np.float64)
        self.monitor_row_dbfs = np.array([], dtype=np.float64)
        self.monitor_waterfall = np.full((config.WATERFALL_ROWS, 1), -140.0, dtype=np.float64)
        self.capture_offset = 0
        self.chips_by_phase: dict[int, list[int]] = defaultdict(list)
        self.last_announced_bit: dict[tuple[int, int], int] = {}
        self.last_packet_text = "No live decode yet"
        self.packet_events = deque(maxlen=6)
        self.packet_bits: list[int] = []
        self.packet_bits_label = "phase=0 off=0"


def _read_status(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="ascii"))
    except Exception:
        return None


def _read_new_capture(path: Path, state: MonitorState) -> None:
    if not path.exists():
        return
    with path.open("r", encoding="ascii") as handle:
        handle.seek(state.capture_offset)
        while True:
            line = handle.readline()
            if not line:
                break
            text = line.strip()
            if not text:
                continue
            try:
                rec = json.loads(text)
            except json.JSONDecodeError:
                continue
            phase = int(rec.get("phase", 0))
            chips = [1 if int(v) else 0 for v in rec.get("chips", [])]
            state.chips_by_phase[phase].extend(chips)
        state.capture_offset = handle.tell()


def _update_live_decode(state: MonitorState) -> None:
    analysis = analyze_live_decode(state.chips_by_phase, tail_bits=96, include_matches=True)
    state.last_packet_text = analysis.best_text
    state.packet_bits = analysis.best_packet_bits
    state.packet_bits_label = analysis.best_packet_label

    for phase, decode_offset, header_idx in analysis.matches:
        key = (phase, decode_offset)
        if header_idx > state.last_announced_bit.get(key, -1):
            state.last_announced_bit[key] = header_idx
            state.packet_events.appendleft(f"PKT phase={phase} off={decode_offset} bit={header_idx}")


def main() -> int:
    status_path = Path(config.RX_STATUS_JSON)
    capture_path = Path(config.RX_CAPTURE_NDJSON)
    state = MonitorState()

    default_axis_khz = np.linspace(
        -config.CENTERED_SPAN_HZ / 1000.0,
        config.CENTERED_SPAN_HZ / 1000.0,
        int(config.RX_MONITOR_SPECTRUM_BINS),
        dtype=np.float64,
    )
    state.monitor_axis_khz = default_axis_khz
    state.monitor_row_dbfs = np.full_like(default_axis_khz, -140.0)
    state.monitor_waterfall = np.full((config.WATERFALL_ROWS, default_axis_khz.size), -140.0, dtype=np.float64)

    fig, (ax_centered, ax_waterfall, ax_quality, ax_bits) = plt.subplots(
        4, 1, figsize=(11, 11), gridspec_kw={"height_ratios": [2, 2.5, 1.5, 1.5]}
    )
    fig.subplots_adjust(top=0.93, bottom=0.10, hspace=0.82)
    fig.canvas.manager.set_window_title("FSK RX Monitor")
    fig.suptitle("FSK RX Monitor \u2014 spectrum \u00b7 waterfall \u00b7 signal quality \u00b7 decode")

    (line_centered,) = ax_centered.plot(default_axis_khz, state.monitor_row_dbfs, lw=1.2, color="C0")
    ax_centered.axvline(0.0, color="C3", lw=1.0, alpha=0.8)
    ax_centered.axvline(config.SIDEBAND_OFFSET_KHZ, color="lime", lw=0.9, alpha=0.7, linestyle="--")
    ax_centered.axvline(-config.SIDEBAND_OFFSET_KHZ, color="lime", lw=0.9, alpha=0.7, linestyle="--")
    ax_centered.axvline(config.SIDEBAND_OFFSET_F0_KHZ, color="orange", lw=0.9, alpha=0.7, linestyle="--")
    ax_centered.axvline(-config.SIDEBAND_OFFSET_F0_KHZ, color="orange", lw=0.9, alpha=0.7, linestyle="--")
    ax_centered.set_title("Carrier-Centered Spectrum", pad=10)
    ax_centered.set_ylabel("Magnitude (dBFS)")
    ax_centered.set_xlim(default_axis_khz[0], default_axis_khz[-1])
    ax_centered.set_ylim(-120.0, -60.0)
    ax_centered.grid(True, alpha=0.3)

    waterfall_img = ax_waterfall.imshow(
        state.monitor_waterfall,
        aspect="auto",
        origin="lower",
        extent=[default_axis_khz[0], default_axis_khz[-1], 0, config.WATERFALL_ROWS],
        cmap="plasma",
        vmin=-120.0,
        vmax=-80.0,
        interpolation="nearest",
    )
    ax_waterfall.set_title("Carrier-Centered Waterfall", pad=10)
    ax_waterfall.set_ylabel("Snapshot")
    ax_waterfall.axvline(config.SIDEBAND_OFFSET_KHZ, color="lime", lw=0.8, alpha=0.6, linestyle="--")
    ax_waterfall.axvline(-config.SIDEBAND_OFFSET_KHZ, color="lime", lw=0.8, alpha=0.6, linestyle="--")
    ax_waterfall.axvline(config.SIDEBAND_OFFSET_F0_KHZ, color="orange", lw=0.8, alpha=0.6, linestyle="--")
    ax_waterfall.axvline(-config.SIDEBAND_OFFSET_F0_KHZ, color="orange", lw=0.8, alpha=0.6, linestyle="--")

    # ---- Signal quality panel (SNR + CFO rolling history) ------------------
    (line_snr,) = ax_quality.plot([], [], lw=1.4, color="C0", label="SNR dB")
    ax_quality.axhline(
        config.SNR_LOCK_THRESHOLD_DB, color="lime", lw=0.9, ls="--", alpha=0.8,
        label=f"lock thr {config.SNR_LOCK_THRESHOLD_DB:.0f} dB",
    )
    ax_quality.set_ylabel("Sideband SNR (dB)", color="C0", fontsize=8)
    ax_quality.tick_params(axis="y", labelcolor="C0", labelsize=7)
    ax_quality.set_xlim(0, state.history_len)
    ax_quality.set_ylim(-5.0, max(40.0, config.SNR_LOCK_THRESHOLD_DB + 15.0))
    ax_quality.set_title("Signal Quality & CFO", pad=6, fontsize=9)
    ax_quality.grid(True, alpha=0.3)
    ax_cfo = ax_quality.twinx()
    (line_cfo,) = ax_cfo.plot([], [], lw=1.0, color="C1", alpha=0.85, label="CFO Hz")
    ax_cfo.axhline(0.0, color="0.6", lw=0.5, ls=":")
    ax_cfo.set_ylim(-600.0, 600.0)
    ax_cfo.set_ylabel("CFO (Hz)", color="C1", fontsize=8)
    ax_cfo.tick_params(axis="y", labelcolor="C1", labelsize=7)

    packet_bit_count = len(bytes_to_bit_list(config.PREAMBLE_BYTES + config.SYNC_BYTES)) + 8 + int(config.LIVE_DECODE_MAX_PAYLOAD_BYTES) * 8
    bit_axis = np.arange(packet_bit_count, dtype=np.float64)
    _preamble_bits = len(bytes_to_bit_list(config.PREAMBLE_BYTES))
    _header_bits = len(bytes_to_bit_list(config.PREAMBLE_BYTES + config.SYNC_BYTES))
    _len_field_end = _header_bits + 8
    (line_bits,) = ax_bits.step(bit_axis, np.zeros_like(bit_axis), where="mid", lw=1.2, color="C3")
    ax_bits.axvspan(0, _preamble_bits, color="C0", alpha=0.09)
    ax_bits.axvspan(_preamble_bits, _header_bits, color="C2", alpha=0.09)
    ax_bits.axvspan(_header_bits, _len_field_end, color="C5", alpha=0.12)
    ax_bits.axvspan(_len_field_end, packet_bit_count, color="C3", alpha=0.06)
    ax_bits.axvline(_preamble_bits, color="0.5", lw=0.8, alpha=0.6)
    ax_bits.axvline(_header_bits, color="0.5", lw=0.8, alpha=0.6)
    ax_bits.axvline(_len_field_end, color="0.5", lw=0.8, alpha=0.6)
    ax_bits.text(_preamble_bits / 2, 1.06, "pre", ha="center", va="bottom", fontsize=7, color="0.5", transform=ax_bits.get_xaxis_transform())
    ax_bits.text((_preamble_bits + _header_bits) / 2, 1.06, "sync", ha="center", va="bottom", fontsize=7, color="0.5", transform=ax_bits.get_xaxis_transform())
    ax_bits.text((_header_bits + _len_field_end) / 2, 1.06, "len", ha="center", va="bottom", fontsize=7, color="0.5", transform=ax_bits.get_xaxis_transform())
    ax_bits.text((_len_field_end + packet_bit_count) / 2, 1.06, "payload", ha="center", va="bottom", fontsize=7, color="0.5", transform=ax_bits.get_xaxis_transform())
    ax_bits.set_title("Best Decoded Packet Bits", pad=10)
    ax_bits.set_xlabel("Packet bit index")
    ax_bits.set_ylabel("Bit")
    ax_bits.set_xlim(0, max(1, len(bit_axis) - 1))
    ax_bits.set_ylim(-0.1, 1.1)
    ax_bits.grid(True, alpha=0.3)
    bits_text = ax_bits.text(0.01, 0.92, "Waiting for decoded packet...", transform=ax_bits.transAxes, va="top", ha="left", family="Consolas")

    status_text = fig.text(0.01, 0.01, "Waiting for RX status...", ha="left", va="bottom", family="Consolas")

    def _update(_frame: int):
        status = _read_status(status_path)
        _read_new_capture(capture_path, state)
        _update_live_decode(state)
        if status is None:
            return line_centered, waterfall_img, line_bits, bits_text, status_text

        frame = int(status.get("frame", 0))
        if state.frames and frame == state.frames[-1]:
            return line_centered, waterfall_img, line_bits, bits_text, status_text

        state.frames.append(frame)
        state.rx_ms.append(float(status.get("rx_ms", 0.0)))
        state.proc_ms.append(float(status.get("proc_ms", 0.0)))
        state.gap_ms.append(float(status.get("gap_ms", 0.0)))
        state.cfo_hz.append(float(status.get("cfo_hz", 0.0)))
        state.ncc.append(float(status.get("ncc_ema", 0.0)))

        row_vals = np.asarray(status.get("monitor_row_dbfs", []), dtype=np.float64)
        if row_vals.size == state.monitor_axis_khz.size:
            state.monitor_row_dbfs = row_vals
            line_centered.set_data(state.monitor_axis_khz, state.monitor_row_dbfs)

            state.monitor_waterfall[:-1] = state.monitor_waterfall[1:]
            if config.WATERFALL_ROWS > 1:
                row_vals = (
                    config.WATERFALL_ROW_BLEND * row_vals
                    + (1.0 - config.WATERFALL_ROW_BLEND) * state.monitor_waterfall[-2]
                )
            state.monitor_waterfall[-1] = row_vals
            waterfall_img.set_data(state.monitor_waterfall)

            noise_floor = float(status.get("monitor_noise_floor_dbfs", -120.0))
            row_peak = float(np.max(state.monitor_row_dbfs)) if state.monitor_row_dbfs.size else -60.0
            ax_centered.set_ylim(noise_floor - 6.0, max(noise_floor + 10.0, row_peak + 4.0))

            wf_axis_mask = np.abs(state.monitor_axis_khz) > 0.5
            valid = state.monitor_waterfall[:, wf_axis_mask] if np.any(wf_axis_mask) else state.monitor_waterfall
            valid = valid[valid > -139.0]
            if valid.size:
                nf = float(np.percentile(valid, 15))
                waterfall_img.set_clim(vmin=nf - 1.0, vmax=nf + config.WATERFALL_DYN_RANGE_DB)

        locked = bool(int(status.get("lock", 0)))
        state.locked = locked
        lock_text = "LOCK" if locked else "SEARCH"
        late = status.get("late_frames")
        slips = status.get("gap_slips")

        # ---- Update signal quality panel -----------------------------------
        snr_val = float(status.get("monitor_sideband_snr_db", 0.0))
        state.snr_db.append(snr_val)
        n_pts = len(state.snr_db)
        if n_pts >= 2:
            xq = np.arange(n_pts, dtype=np.float64)
            line_snr.set_data(xq, np.asarray(state.snr_db, dtype=np.float64))
            line_cfo.set_data(xq, np.asarray(state.cfo_hz, dtype=np.float64))
            ax_quality.set_xlim(0, max(n_pts - 1, 1))
            ax_cfo.set_xlim(0, max(n_pts - 1, 1))
            snr_arr = np.asarray(state.snr_db, dtype=np.float64)
            ax_quality.set_ylim(
                min(-5.0, float(np.min(snr_arr)) - 2.0),
                max(float(np.max(snr_arr)) + 4.0, config.SNR_LOCK_THRESHOLD_DB + 5.0),
            )
        ax_quality.set_facecolor("#e8f5e9" if locked else "#fff8e1")

        packet_bits = state.packet_bits
        if packet_bits:
            bit_vals = np.asarray(packet_bits, dtype=np.float64)
            bit_axis = np.arange(bit_vals.size, dtype=np.float64)
            line_bits.set_data(bit_axis, bit_vals)
            ax_bits.set_xlim(0, max(1, bit_vals.size - 1))
            bits_text.set_text(
                f"{state.packet_bits_label}\n"
                f"bits={bits_to_text(packet_bits)}\n"
                f"packet={config.PREAMBLE_BYTES.hex().upper()} {config.SYNC_BYTES.hex().upper()} [len] payload"
            )
        else:
            line_bits.set_data([0.0], [0.0])
            bits_text.set_text("Waiting for decoded packet...")

        events_text = " | ".join(state.packet_events) if state.packet_events else "No recent packet match"
        sb_pos = status.get("monitor_sideband_pos_dbfs")
        sb_neg = status.get("monitor_sideband_neg_dbfs")
        decode_summary = status.get("decode_summary") or state.last_packet_text

        status_text.set_color("#1b5e20" if locked else "#e65100")
        status_text.set_text(
            f"frame={status.get('frame')}  {lock_text}  phase={status.get('best_phase')}  chips={status.get('chips_seen')}  "
            f"peak={status.get('peak_hz')} Hz  CFO={status.get('cfo_hz')} Hz  NCC={status.get('ncc_ema')}  late={late}  slips={slips}\n"
            f"sidebands: snr={snr_val:.1f} dB  +1k={sb_pos} dBFS  -1k={sb_neg} dBFS  rx/proc/gap={status.get('rx_ms')}/{status.get('proc_ms')}/{status.get('gap_ms')} ms\n"
            f"best_decode={decode_summary}\n"
            f"events={events_text}"
        )
        return line_centered, waterfall_img, line_bits, bits_text, status_text

    _anim = FuncAnimation(fig, _update, interval=250, blit=False, cache_frame_data=False)
    plt.show()
    _ = _anim
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

