#!/usr/bin/env python3
"""Offline heavy packet decoder for RX-only FSK captures.

Usage:
    python -m Receiver_FSK.offline_decoder_fsk
    python -m Receiver_FSK.offline_decoder_fsk --capture Receiver_FSK/captures/chips_capture.ndjson --top 30
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from . import config_secure as config
from .packet_decoder_secure import (
    bits_to_bytes,
    bytes_to_bit_list,
    find_header_match,
    majority_decode_triplets,
    safe_ascii,
)


def _load_capture(path: Path) -> dict[int, list[int]]:
    chips_by_phase: dict[int, list[int]] = defaultdict(list)
    if not path.exists():
        raise FileNotFoundError(f"Capture file not found: {path}")

    with path.open("r", encoding="ascii") as handle:
        for line_num, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rec = json.loads(text)
            except json.JSONDecodeError:
                continue
            if "phase" not in rec or "chips" not in rec:
                continue
            phase = int(rec["phase"])
            chips = [1 if int(v) else 0 for v in rec["chips"]]
            chips_by_phase[phase].extend(chips)

    return chips_by_phase


def _decode_candidates(chips_by_phase: dict[int, list[int]], top: int) -> list[dict[str, object]]:
    header_bits = bytes_to_bit_list(config.PREAMBLE_BYTES + config.SYNC_BYTES)
    expected_payload_bits = bytes_to_bit_list(config.PAYLOAD_BYTES)
    payload_len = len(expected_payload_bits)

    rows: list[dict[str, object]] = []

    for phase, chips in sorted(chips_by_phase.items()):
        if len(chips) < config.REPETITION_CHIPS * (len(header_bits) + payload_len):
            continue

        for decode_offset in range(config.REPETITION_CHIPS):
            decoded_bits = majority_decode_triplets(chips, decode_offset)
            if len(decoded_bits) < len(header_bits) + payload_len:
                continue

            search_start = 0
            while True:
                header_idx, header_errors = find_header_match(
                    decoded_bits,
                    header_bits,
                    search_start,
                    config.HEADER_MAX_BIT_ERRORS,
                )
                if header_idx < 0:
                    break

                payload_start = header_idx + len(header_bits)
                payload_end = payload_start + payload_len
                if payload_end > len(decoded_bits):
                    break

                payload_bits = decoded_bits[payload_start:payload_end]
                payload = bits_to_bytes(payload_bits)
                payload_errors_normal = sum(
                    1
                    for left, right in zip(payload_bits, expected_payload_bits)
                    if left != right
                )
                if config.ALLOW_INVERTED_PAYLOAD_MATCH:
                    payload_errors_inverted = sum(
                        1
                        for left, right in zip(payload_bits, expected_payload_bits)
                        if (1 - left) != right
                    )
                else:
                    payload_errors_inverted = payload_errors_normal

                payload_errors = min(payload_errors_normal, payload_errors_inverted)
                polarity = "inverted" if payload_errors_inverted < payload_errors_normal else "normal"

                rows.append(
                    {
                        "phase": phase,
                        "chip_offset": decode_offset,
                        "bit": header_idx,
                        "header_errors": header_errors,
                        "payload_errors": payload_errors,
                        "payload_polarity": polarity,
                        "payload_hex": payload.hex().upper(),
                        "payload_ascii": safe_ascii(payload),
                    }
                )
                search_start = header_idx + 1

    rows.sort(key=lambda r: (int(r["payload_errors"]), int(r["header_errors"])))
    return rows[:top]


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline decode for RX-only FSK capture stream")
    parser.add_argument(
        "--capture",
        default=config.RX_CAPTURE_NDJSON,
        help="Path to NDJSON capture file produced by RX-only loop",
    )
    parser.add_argument("--top", type=int, default=25, help="Top-N candidates to print")
    args = parser.parse_args()

    path = Path(args.capture)
    chips_by_phase = _load_capture(path)
    if not chips_by_phase:
        print(f"No capture records found in {path}")
        return 1

    candidates = _decode_candidates(chips_by_phase, max(1, int(args.top)))
    if not candidates:
        print("No header candidates found.")
        return 0

    print(f"Loaded phases: {len(chips_by_phase)} from {path}")
    print(f"Top {len(candidates)} candidates:")
    for idx, row in enumerate(candidates, start=1):
        print(
            f"[{idx:02d}] phase={row['phase']} chip_offset={row['chip_offset']} bit={row['bit']} "
            f"header_errors={row['header_errors']} payload_errors={row['payload_errors']} "
            f"payload_polarity={row['payload_polarity']} payload_hex={row['payload_hex']} "
            f"payload_ascii={row['payload_ascii']!r}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
