"""Shared live-decode helpers for RX-only loop and decoupled monitor."""

from __future__ import annotations

from dataclasses import dataclass

from . import config as config
from .packet_decoder import (
    bits_to_bytes,
    bytes_to_bit_list,
    majority_decode_triplets,
)


@dataclass
class LiveDecodeAnalysis:
    best_text: str
    best_tail_bits: list[int]
    best_tail_label: str
    best_packet_bits: list[int]
    best_packet_label: str
    matches: list[tuple[int, int, int]]


def weak_triplet_count(
    chips: list[int],
    decode_offset: int,
    bit_start: int,
    bit_count: int,
) -> int:
    """Count logical bits decided by a weak majority in a window."""
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


def analyze_live_decode(
    chips_by_phase: dict[int, list[int]],
    tail_bits: int = 96,
    include_matches: bool = False,
) -> LiveDecodeAnalysis:
    """Evaluate packet candidates and return the best live-decode view.

    Packet format: [preamble+sync header] [1-byte length] [length bytes payload]
    """
    header_bits = bytes_to_bit_list(config.PREAMBLE_BYTES + config.SYNC_BYTES)
    max_payload_bytes = int(config.LIVE_DECODE_MAX_PAYLOAD_BYTES)
    # Minimum decodable packet: header + length byte + 1 payload byte
    min_packet_bits = len(header_bits) + 8 + 8
    # Maximum window size used to set scan bounds
    max_packet_bits = len(header_bits) + 8 + max_payload_bytes * 8

    best_text = "No live decode yet"
    best_score: tuple[int, ...] | None = None
    best_tail_bits: list[int] = []
    best_tail_label = "phase=0 off=0"
    best_packet_bits: list[int] = []
    best_packet_label = "phase=0 off=0"
    matches: list[tuple[int, int, int]] = []

    for phase, chips in sorted(chips_by_phase.items()):
        if len(chips) < config.REPETITION_CHIPS * min_packet_bits:
            continue

        for decode_offset in range(config.REPETITION_CHIPS):
            decoded_bits = majority_decode_triplets(chips, decode_offset)
            if decoded_bits and not best_tail_bits:
                best_tail_bits = decoded_bits[-tail_bits:]
                best_tail_label = f"phase={phase} off={decode_offset}"
            if len(decoded_bits) < min_packet_bits:
                continue

            recent_start = max(0, len(decoded_bits) - config.LIVE_DECODE_RECENT_BITS)
            last_start = len(decoded_bits) - min_packet_bits
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

                # Read the length field byte
                len_field_start = header_idx + len(header_bits)
                len_field_bits = decoded_bits[len_field_start : len_field_start + 8]
                if len(len_field_bits) < 8:
                    continue
                payload_byte_count = bits_to_bytes(len_field_bits)[0]
                if payload_byte_count == 0 or payload_byte_count > max_payload_bytes:
                    continue

                payload_len = payload_byte_count * 8
                payload_start = len_field_start + 8
                payload_end = payload_start + payload_len
                if payload_end > len(decoded_bits):
                    continue

                payload_bits = decoded_bits[payload_start:payload_end]
                payload = bits_to_bytes(payload_bits)
                recency = len(decoded_bits) - payload_end
                total_bits = len(header_bits) + 8 + payload_len
                weak_bits = weak_triplet_count(
                    chips,
                    decode_offset,
                    header_idx,
                    total_bits,
                )
                if weak_bits > int(config.LIVE_DECODE_MAX_WEAK_BITS):
                    continue

                score = (header_errors, weak_bits, recency)
                text = (
                    f"phase={phase} off={decode_offset} bit={header_idx} "
                    f"herr={header_errors} weak={weak_bits} len={payload_byte_count} "
                    f"payload={payload.hex().upper()} {payload!r}"
                )
                matched = header_errors == 0

                if best_score is None or score < best_score:
                    best_score = score
                    best_text = text
                    best_tail_bits = decoded_bits[-tail_bits:]
                    best_tail_label = f"phase={phase} off={decode_offset}"
                    best_packet_bits = decoded_bits[header_idx:payload_end]
                    best_packet_label = f"phase={phase} off={decode_offset} bit={header_idx}"

                if include_matches and matched:
                    matches.append((phase, decode_offset, header_idx))

    return LiveDecodeAnalysis(
        best_text=best_text,
        best_tail_bits=best_tail_bits,
        best_tail_label=best_tail_label,
        best_packet_bits=best_packet_bits,
        best_packet_label=best_packet_label,
        matches=matches,
    )

