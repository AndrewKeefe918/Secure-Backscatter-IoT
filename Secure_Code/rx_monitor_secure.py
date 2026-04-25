#!/usr/bin/env python3
"""Decoupled live monitor for the RX-only FSK loop.

Run in a separate terminal:
    python -m Receiver_FSK.rx_monitor_fsk
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from . import config_secure as config
from .packet_decoder_secure import (
    bits_to_bytes,
    bytes_to_bit_list,
    bits_to_text,
    majority_decode_triplets,
)
from .secure_packet import SecureReceiver


class MonitorState:
    def __init__(self, history_len: int = 180) -> None:
        self.history_len = history_len
        self.frames = deque(maxlen=history_len)
        self.rx_ms = deque(maxlen=history_len)
        self.proc_ms = deque(maxlen=history_len)
        self.gap_ms = deque(maxlen=history_len)
        self.cfo_hz = deque(maxlen=history_len)
        self.ncc = deque(maxlen=history_len)
        self.monitor_axis_khz = np.array([], dtype=np.float64)
        self.monitor_row_dbfs = np.array([], dtype=np.float64)
        self.monitor_waterfall = np.full((config.WATERFALL_ROWS, 1), -140.0, dtype=np.float64)
        self.capture_offset = 0
        self.chips_by_phase: dict[int, list[int]] = defaultdict(list)
        self.last_announced_bit: dict[tuple[int, int], int] = {}
        self.last_packet_text = "No live decode yet"
        self.packet_events = deque(maxlen=6)
        self.logic_tail_bits: list[int] = []
        self.logic_tail_label = "phase=0 off=0"
        self.packet_bits: list[int] = []
        self.packet_bits_label = "phase=0 off=0"
        self.secure_rx = SecureReceiver(bytes.fromhex(config.SHARED_KEY_HEX), state_path=None)
        self.verified_packets: dict[tuple, object] = {}


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


def _weak_triplet_count(
    chips: list[int],
    decode_offset: int,
    bit_start: int,
    bit_count: int,
) -> int:
    """Count logical bits decided by a weak 2-of-3 majority in a window."""
    weak = 0
    step = int(config.REPETITION_CHIPS)
    for bit_idx in range(bit_start, bit_start + bit_count):
        chip_start = decode_offset + bit_idx * step
        chip_end = chip_start + step
        if chip_end > len(chips):
            break
        ones = sum(chips[chip_start:chip_end])
        if ones not in (0, step):
            weak += 1
    return weak


def _update_live_decode(state: MonitorState) -> None:
    header_bits = bytes_to_bit_list(config.PREAMBLE_BYTES + config.SYNC_BYTES)
    payload_len = int(config.LIVE_DECODE_PAYLOAD_BYTES) * 8
    packet_bits = len(header_bits) + payload_len

    best_text = "No live decode yet"
    best_score: tuple[int, ...] | None = None
    best_tail_bits: list[int] = []
    best_tail_label = "phase=0 off=0"
    best_packet_bits: list[int] = []
    best_packet_label = "phase=0 off=0"

    for phase, chips in sorted(state.chips_by_phase.items()):
        if len(chips) < config.REPETITION_CHIPS * packet_bits:
            continue

        for decode_offset in range(config.REPETITION_CHIPS):
            decoded_bits = majority_decode_triplets(chips, decode_offset)
            if decoded_bits and not best_tail_bits:
                best_tail_bits = decoded_bits[-96:]
                best_tail_label = f"phase={phase} off={decode_offset}"
            if len(decoded_bits) < packet_bits:
                continue

            recent_start = max(0, len(decoded_bits) - config.LIVE_DECODE_RECENT_BITS)
            last_start = len(decoded_bits) - packet_bits
            scan_start = max(0, min(recent_start, last_start))
            for header_idx in range(scan_start, last_start + 1):
                header_errors = sum(
                    1
                    for left, right in zip(
                        decoded_bits[header_idx : header_idx + len(header_bits)],
                        header_bits,
                    )
                    if left != right
                )
                if header_errors > config.LIVE_DECODE_MAX_HEADER_ERRORS:
                    continue

                payload_start = header_idx + len(header_bits)
                payload_end = payload_start + payload_len
                payload_bits = decoded_bits[payload_start:payload_end]
                payload = bits_to_bytes(payload_bits)
                recency = len(decoded_bits) - payload_end
                weak_bits = _weak_triplet_count(
                    chips,
                    decode_offset,
                    header_idx,
                    len(header_bits) + payload_len,
                )
                if weak_bits > int(config.LIVE_DECODE_MAX_WEAK_BITS):
                    continue

                # Verify once per (phase, offset, header_idx) position to avoid
                # re-consuming the replay counter on repeated monitor scans.
                cache_key = (phase, decode_offset, header_idx)
                if cache_key not in state.verified_packets:
                    result = state.secure_rx.verify_and_decrypt(payload)
                    state.verified_packets[cache_key] = result
                    if result.valid:
                        state.packet_events.appendleft(
                            f"AUTHENTICATED cnt={result.counter} "
                            f"phase={phase} off={decode_offset} bit={header_idx}"
                        )
                    elif "replay" not in result.reason:
                        state.packet_events.appendleft(
                            f"REJECTED [{result.reason[:30]}] phase={phase}"
                        )
                else:
                    result = state.verified_packets[cache_key]

                if result.valid:
                    score = (0, header_errors, weak_bits, recency)
                    text = (
                        f"phase={phase} off={decode_offset} bit={header_idx} "
                        f"herr={header_errors} weak={weak_bits} "
                        f"counter={result.counter} plaintext={result.plaintext!r}  AUTHENTICATED"
                    )
                else:
                    score = (1, header_errors, weak_bits, recency)
                    text = (
                        f"phase={phase} off={decode_offset} bit={header_idx} "
                        f"herr={header_errors} weak={weak_bits} "
                        f"payload={payload.hex().upper()} REJECTED: {result.reason}"
                    )

                if best_score is None or score < best_score:
                    best_score = score
                    best_text = text
                    best_tail_bits = decoded_bits[-96:]
                    best_tail_label = f"phase={phase} off={decode_offset}"
                    best_packet_bits = decoded_bits[header_idx:payload_end]
                    best_packet_label = f"phase={phase} off={decode_offset} bit={header_idx}"

    state.last_packet_text = best_text
    state.logic_tail_bits = best_tail_bits
    state.logic_tail_label = best_tail_label
    state.packet_bits = best_packet_bits
    state.packet_bits_label = best_packet_label


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

    fig, (ax_centered, ax_waterfall, ax_bits) = plt.subplots(3, 1, figsize=(11, 9))
    fig.subplots_adjust(top=0.92, bottom=0.14, hspace=0.62)
    fig.canvas.manager.set_window_title("FSK RX Monitor")
    fig.suptitle("FSK RX Monitor — decoupled spectrum, waterfall, and live decode")

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

    packet_bit_count = len(bytes_to_bit_list(config.PREAMBLE_BYTES + config.SYNC_BYTES)) + int(config.LIVE_DECODE_PAYLOAD_BYTES) * 8
    bit_axis = np.arange(packet_bit_count, dtype=np.float64)
    (line_bits,) = ax_bits.step(bit_axis, np.zeros_like(bit_axis), where="mid", lw=1.2, color="C3")
    ax_bits.axvline(len(bytes_to_bit_list(config.PREAMBLE_BYTES)), color="0.5", lw=0.8, alpha=0.6)
    ax_bits.axvline(len(bytes_to_bit_list(config.PREAMBLE_BYTES + config.SYNC_BYTES)), color="0.5", lw=0.8, alpha=0.6)
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

        axis_vals = np.asarray(status.get("monitor_axis_khz", []), dtype=np.float64)
        row_vals = np.asarray(status.get("monitor_row_dbfs", []), dtype=np.float64)
        if axis_vals.size and row_vals.size and axis_vals.size == row_vals.size:
            if axis_vals.size != state.monitor_axis_khz.size:
                state.monitor_waterfall = np.full((config.WATERFALL_ROWS, axis_vals.size), -140.0, dtype=np.float64)
                ax_centered.set_xlim(float(axis_vals[0]), float(axis_vals[-1]))
                waterfall_img.set_extent([float(axis_vals[0]), float(axis_vals[-1]), 0, config.WATERFALL_ROWS])
            state.monitor_axis_khz = axis_vals
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

        lock_text = "LOCK" if int(status.get("lock", 0)) else "SEARCH"
        late = status.get("late_frames")
        slips = status.get("gap_slips")

        packet_bits = state.packet_bits
        if packet_bits:
            bit_vals = np.asarray(packet_bits, dtype=np.float64)
            bit_axis = np.arange(bit_vals.size, dtype=np.float64)
            line_bits.set_data(bit_axis, bit_vals)
            ax_bits.set_xlim(0, max(1, bit_vals.size - 1))
            bits_text.set_text(
                f"{state.packet_bits_label}\n"
                f"bits={bits_to_text(packet_bits)}\n"
                f"packet={config.PREAMBLE_BYTES.hex().upper()} {config.SYNC_BYTES.hex().upper()} + {config.LIVE_DECODE_PAYLOAD_BYTES}B"
            )
        else:
            line_bits.set_data([0.0], [0.0])
            bits_text.set_text("Waiting for decoded packet...")

        events_text = " | ".join(state.packet_events) if state.packet_events else "No recent packet match"
        snr_db = status.get("monitor_sideband_snr_db")
        sb_pos = status.get("monitor_sideband_pos_dbfs")
        sb_neg = status.get("monitor_sideband_neg_dbfs")
        decode_summary = status.get("decode_summary") or state.last_packet_text

        status_text.set_text(
            f"frame={status.get('frame')}  {lock_text}  phase={status.get('best_phase')}  chips={status.get('chips_seen')}  "
            f"peak={status.get('peak_hz')} Hz  CFO={status.get('cfo_hz')} Hz  NCC={status.get('ncc_ema')}  late={late}  slips={slips}\n"
            f"sidebands: snr={snr_db} dB  +1k={sb_pos} dBFS  -1k={sb_neg} dBFS  rx/proc/gap={status.get('rx_ms')}/{status.get('proc_ms')}/{status.get('gap_ms')} ms\n"
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
