"""Shared live-decode helpers for RX-only loop and decoupled monitor."""

from __future__ import annotations

from dataclasses import dataclass

from . import config as config
from .packet_decoder import (
    bits_to_bytes,
    bytes_to_bit_list,
    majority_decode_triplets,
)
from .secure_packet import SecureReceiver
from .secure_packet import DecodedPacket

# Module-level SecureReceiver for the live-decode path (no persistent state —
# the RX-only loop holds the authoritative replay state).
_monitor_rx: SecureReceiver | None = None


def _get_monitor_rx() -> SecureReceiver:
    global _monitor_rx
    if _monitor_rx is None:
        _monitor_rx = SecureReceiver(bytes.fromhex(config.SHARED_KEY_HEX), state_path=None)
    return _monitor_rx


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
    secure_rx: "SecureReceiver | None" = None,
    verified_cache: dict[bytes, DecodedPacket] | None = None,
    base_chip_index_by_phase: dict[int, int] | None = None,
) -> LiveDecodeAnalysis:
    """Evaluate packet candidates and return the best live-decode view.

    In SECURE_MODE the payload is a fixed 16-byte AEAD frame with no
    length prefix.  Outside secure mode the existing length-byte framing
    is used unchanged.
    """
    header_bits = bytes_to_bit_list(config.PREAMBLE_BYTES + config.SYNC_BYTES)

    payload_len = 0
    max_payload_bytes = 0
    rx: SecureReceiver | None = None
    verified: dict[bytes, DecodedPacket] = {}

    if config.SECURE_MODE:
        # Fixed 16-byte payload, no length byte.
        payload_len = int(config.LIVE_DECODE_PAYLOAD_BYTES) * 8
        min_packet_bits = len(header_bits) + payload_len
        rx = secure_rx if secure_rx is not None else _get_monitor_rx()
        verified = verified_cache if verified_cache is not None else {}
    else:
        max_payload_bytes = int(config.LIVE_DECODE_MAX_PAYLOAD_BYTES)
        # Minimum decodable packet: header + length byte + 1 payload byte
        min_packet_bits = len(header_bits) + 8 + 8

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

            decode_streams: list[tuple[str, list[int]]] = [("norm", decoded_bits)]
            if bool(config.LIVE_DECODE_TRY_INVERTED):
                decode_streams.append(("inv", [1 - bit for bit in decoded_bits]))

            for polarity, scan_bits in decode_streams:
                pol_penalty = 0 if polarity == "norm" else 1

                recent_start = max(0, len(scan_bits) - config.LIVE_DECODE_RECENT_BITS)
                last_start = len(scan_bits) - min_packet_bits
                scan_start = max(0, min(recent_start, last_start))
                # Prefer recent candidates for responsiveness, but fall back to the
                # full retained history when the packet drifts outside the tail window.
                search_ranges: list[tuple[int, int]] = [(scan_start, last_start)]
                if scan_start > 0:
                    search_ranges.append((0, scan_start - 1))

                if config.SECURE_MODE:
                    assert rx is not None
                    for range_start, range_end in search_ranges:
                        for header_idx in range(range_start, range_end + 1):
                            header_errors = sum(
                                1
                                for left, right in zip(
                                    scan_bits[header_idx : header_idx + len(header_bits)],
                                    header_bits,
                                )
                                if left != right
                            )
                            if header_errors > config.LIVE_DECODE_MAX_HEADER_ERRORS:
                                continue

                            payload_start = header_idx + len(header_bits)
                            payload_end = payload_start + payload_len
                            if payload_end > len(scan_bits):
                                continue

                            payload_bits = scan_bits[payload_start:payload_end]
                            payload = bits_to_bytes(payload_bits)
                            recency = len(scan_bits) - payload_end
                            weak_bits = weak_triplet_count(
                                chips,
                                decode_offset,
                                header_idx,
                                len(header_bits) + payload_len,
                            )
                            if weak_bits > int(config.LIVE_DECODE_MAX_WEAK_BITS):
                                continue

                            # Deduplicate by payload bytes, not bit position. In jittery
                            # windows the same physical packet can appear at slightly
                            # different header indices across frames; position-based keys
                            # would re-verify and trigger replay rejects on duplicates.
                            # However, once the receiver has accepted a newer packet
                            # (last_counter has advanced), evict stale valid entries so
                            # that a genuine replay of an older packet calls
                            # verify_and_decrypt() and is correctly REJECTED.
                            cache_key = payload
                            cache_hit = cache_key in verified
                            if cache_hit:
                                cached = verified[cache_key]
                                if cached.valid and cached.counter < rx.last_counter:
                                    del verified[cache_key]
                                    cache_hit = False
                            if not cache_hit:
                                verified[cache_key] = rx.verify_and_decrypt(payload)
                            result = verified[cache_key]

                            if result.valid:
                                score = (0, header_errors, weak_bits, pol_penalty, recency)
                                suffix = " [cached]" if cache_hit else ""
                                text = (
                                    f"phase={phase} off={decode_offset} bit={header_idx} pol={polarity} "
                                    f"herr={header_errors} weak={weak_bits} "
                                    f"counter={result.counter} plaintext={result.plaintext!r}  "
                                    f"AUTHENTICATED{suffix}"
                                )
                            else:
                                score = (1, header_errors, weak_bits, pol_penalty, recency)
                                suffix = " [cached]" if cache_hit else ""
                                text = (
                                    f"phase={phase} off={decode_offset} bit={header_idx} pol={polarity} "
                                    f"herr={header_errors} weak={weak_bits} "
                                    f"payload={payload.hex().upper()} REJECTED{suffix}: {result.reason}"
                                )

                            if best_score is None or score < best_score:
                                best_score = score
                                best_text = text
                                best_tail_bits = scan_bits[-tail_bits:]
                                best_tail_label = f"phase={phase} off={decode_offset} pol={polarity}"
                                best_packet_bits = scan_bits[header_idx:payload_end]
                                best_packet_label = (
                                    f"phase={phase} off={decode_offset} bit={header_idx} pol={polarity}"
                                )

                            if include_matches and result.valid:
                                matches.append((phase, decode_offset, header_idx))

                else:
                    # Original variable-length framing (non-secure firmware).
                    for range_start, range_end in search_ranges:
                        for header_idx in range(range_start, range_end + 1):
                            header_errors = sum(
                                1
                                for left, right in zip(
                                    scan_bits[header_idx : header_idx + len(header_bits)],
                                    header_bits,
                                )
                                if left != right
                            )
                            if header_errors > config.LIVE_DECODE_MAX_HEADER_ERRORS:
                                continue

                            len_field_start = header_idx + len(header_bits)
                            len_field_bits = scan_bits[len_field_start : len_field_start + 8]
                            if len(len_field_bits) < 8:
                                continue
                            payload_byte_count = bits_to_bytes(len_field_bits)[0]
                            if payload_byte_count == 0 or payload_byte_count > max_payload_bytes:
                                continue

                            var_payload_len = payload_byte_count * 8
                            payload_start = len_field_start + 8
                            payload_end = payload_start + var_payload_len
                            if payload_end > len(scan_bits):
                                continue

                            payload_bits = scan_bits[payload_start:payload_end]
                            payload = bits_to_bytes(payload_bits)
                            recency = len(scan_bits) - payload_end
                            total_bits = len(header_bits) + 8 + var_payload_len
                            weak_bits = weak_triplet_count(chips, decode_offset, header_idx, total_bits)
                            if weak_bits > int(config.LIVE_DECODE_MAX_WEAK_BITS):
                                continue

                            score = (header_errors, weak_bits, pol_penalty, recency)
                            text = (
                                f"phase={phase} off={decode_offset} bit={header_idx} pol={polarity} "
                                f"herr={header_errors} weak={weak_bits} len={payload_byte_count} "
                                f"payload={payload.hex().upper()} {payload!r}"
                            )
                            matched = header_errors == 0

                            if best_score is None or score < best_score:
                                best_score = score
                                best_text = text
                                best_tail_bits = scan_bits[-tail_bits:]
                                best_tail_label = f"phase={phase} off={decode_offset} pol={polarity}"
                                best_packet_bits = scan_bits[header_idx:payload_end]
                                best_packet_label = (
                                    f"phase={phase} off={decode_offset} bit={header_idx} pol={polarity}"
                                )

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

