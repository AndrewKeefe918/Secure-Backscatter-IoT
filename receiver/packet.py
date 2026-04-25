"""Packet pipeline: bit/byte codec, packet candidate model, and decode logic.

Combines the previous packet_decoder + packet_models + packet_pipeline modules
so the entire packet stage lives in one place.
"""

from dataclasses import dataclass
from typing import Any

from . import config


# ---------------------------------------------------------------------------
# Bit/byte primitives
# ---------------------------------------------------------------------------

def bytes_to_bit_list(data: bytes) -> list[int]:
    bits: list[int] = []
    for byte in data:
        for shift in range(7, -1, -1):
            bits.append((byte >> shift) & 1)
    return bits


def bits_to_bytes(bits: list[int]) -> bytes:
    if len(bits) % 8 != 0:
        return b""
    out = bytearray()
    for i in range(0, len(bits), 8):
        value = 0
        for bit in bits[i : i + 8]:
            value = (value << 1) | int(bit)
        out.append(value)
    return bytes(out)


def find_header_match(
    bits: list[int],
    pattern: list[int],
    start: int = 0,
    max_errors: int = 0,
) -> tuple[int, int]:
    if not pattern or len(bits) < len(pattern):
        return -1, len(pattern)
    best_idx = -1
    best_errors = len(pattern) + 1
    max_idx = len(bits) - len(pattern)
    for i in range(max(0, start), max_idx + 1):
        errors = sum(
            1 for left, right in zip(bits[i : i + len(pattern)], pattern) if left != right
        )
        if errors < best_errors:
            best_idx = i
            best_errors = errors
        if errors <= max_errors:
            return i, errors
    if best_errors <= max_errors:
        return best_idx, best_errors
    return -1, best_errors


def bits_to_text(bits: list[int]) -> str:
    return "".join("1" if b else "0" for b in bits)


def safe_ascii(data: bytes) -> str:
    return data.decode("ascii", errors="replace") if data else ""


def majority_decode_triplets(chips: list[int], start_offset: int) -> list[int]:
    span = max(1, int(config.REPETITION_CHIPS))
    threshold = max(1, int(config.MAJORITY_ONES_THRESHOLD))
    decoded: list[int] = []
    idx = start_offset
    while idx + span <= len(chips):
        window = chips[idx : idx + span]
        decoded.append(1 if sum(window) >= threshold else 0)
        idx += span
    return decoded


def packet_confidence(
    *,
    preamble_errors: int,
    sync_errors: int,
    preamble_transitions: int,
) -> float:
    """Return a bounded [0,1] confidence score from header quality metrics."""
    preamble_len = max(1, len(config.PREAMBLE_BYTES) * 8)
    sync_len = max(1, len(config.SYNC_BYTES) * 8)
    preamble_quality = 1.0 - (float(preamble_errors) / float(preamble_len))
    sync_quality = 1.0 - (float(sync_errors) / float(sync_len))
    trans_quality = float(preamble_transitions) / float(max(1, preamble_len - 1))

    score = (
        0.40 * max(0.0, min(1.0, preamble_quality))
        + 0.45 * max(0.0, min(1.0, sync_quality))
        + 0.15 * max(0.0, min(1.0, trans_quality))
    )
    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Packet candidate model
# ---------------------------------------------------------------------------

@dataclass
class PacketCandidate:
    phase: int
    chip_offset: int
    header_abs: int
    header_errors: int
    preamble_errors: int
    sync_errors: int
    preamble_transitions: int
    payload_errors: int
    payload: bytes
    phase_score: float
    observed_preamble: list[int]
    observed_sync: list[int]
    confidence: float


def packet_candidate_sort_key(candidate: PacketCandidate) -> tuple[float, ...]:
    return (
        -candidate.confidence,
        candidate.payload_errors,
        candidate.sync_errors,
        candidate.preamble_errors,
        candidate.header_errors,
        -candidate.phase_score,
        -candidate.preamble_transitions,
        candidate.header_abs,
    )


# ---------------------------------------------------------------------------
# Decode pipeline
# ---------------------------------------------------------------------------

def try_decode_packets(
    *,
    phase: int,
    st: Any,
    phase_score: float,
    packet_header_bits: list[int],
    preamble_bits: list[int],
    payload_patterns_bits: list[list[int]],
    payload_bits_len: int,
) -> list[PacketCandidate]:
    """Scan decoded bits for packet candidates on one phase."""
    candidates: list[PacketCandidate] = []
    for off in range(config.REPETITION_CHIPS):
        decoded = majority_decode_triplets(st.chips, off)
        base_bit = (st.base_chip_index + off) // config.REPETITION_CHIPS
        while True:
            idx, errs = find_header_match(
                decoded,
                packet_header_bits,
                st.search_start_by_offset[off],
                config.HEADER_MAX_BIT_ERRORS,
            )
            if idx < 0:
                st.search_start_by_offset[off] = max(0, len(decoded) - len(packet_header_bits) + 1)
                break

            header_abs = base_bit + idx
            if header_abs <= st.last_header_abs_by_offset[off]:
                st.search_start_by_offset[off] = idx + 1
                continue

            preamble = decoded[idx : idx + len(preamble_bits)]
            preamble_errors = sum(1 for left, right in zip(preamble, preamble_bits) if left != right)
            transitions = sum(1 for a, b in zip(preamble, preamble[1:]) if a != b)
            if transitions < config.PREAMBLE_MIN_TRANSITIONS:
                st.search_start_by_offset[off] = idx + 1
                continue
            if preamble_errors > config.PREAMBLE_MAX_BIT_ERRORS:
                st.search_start_by_offset[off] = idx + 1
                continue

            sync_start = idx + len(preamble_bits)
            sync_bits = decoded[sync_start : sync_start + len(packet_header_bits) - len(preamble_bits)]
            sync_expected = packet_header_bits[len(preamble_bits) :]
            sync_errors = sum(1 for left, right in zip(sync_bits, sync_expected) if left != right)
            if sync_errors > config.SYNC_MAX_BIT_ERRORS:
                st.search_start_by_offset[off] = idx + 1
                continue

            confidence = packet_confidence(
                preamble_errors=preamble_errors,
                sync_errors=sync_errors,
                preamble_transitions=transitions,
            )

            p_start = idx + len(packet_header_bits)
            p_end = p_start + payload_bits_len
            if p_end > len(decoded):
                st.search_start_by_offset[off] = idx
                break

            payload_bits_observed = decoded[p_start:p_end]
            payload = bits_to_bytes(payload_bits_observed)
            if payload_patterns_bits:
                payload_errors = min(
                    sum(1 for left, right in zip(payload_bits_observed, pattern) if left != right)
                    for pattern in payload_patterns_bits
                )
            else:
                payload_errors = 0
            if (
                header_abs > st.last_logged_header_abs_by_offset[off]
                and confidence >= config.HEADER_LOG_MIN_CONFIDENCE
            ):
                st.last_logged_header_abs_by_offset[off] = header_abs
                pre_seen  = bits_to_text(preamble)
                pre_exp   = bits_to_text(preamble_bits)
                sync_seen_txt = bits_to_text(sync_bits)
                sync_exp_txt  = bits_to_text(sync_expected)
                print(
                    f"\x1b[2K\r"
                    f"[HEADER?] conf={confidence:.3f}  err={errs}(pre={preamble_errors} sync={sync_errors} pay={payload_errors})  "
                    f"trans={transitions}  bit={header_abs}  phase={phase}\n"
                    f"         pre : {pre_seen} (exp {pre_exp})\n"
                    f"         sync: {sync_seen_txt} (exp {sync_exp_txt})",
                    flush=True,
                )
            candidates.append(
                PacketCandidate(
                    phase=phase,
                    chip_offset=off,
                    header_abs=header_abs,
                    header_errors=errs,
                    preamble_errors=preamble_errors,
                    sync_errors=sync_errors,
                    preamble_transitions=transitions,
                    payload_errors=payload_errors,
                    payload=payload,
                    phase_score=phase_score,
                    observed_preamble=preamble.copy(),
                    observed_sync=sync_bits.copy(),
                    confidence=confidence,
                )
            )
            st.search_start_by_offset[off] = idx + 1
    return candidates


def _current_bit_horizon(phase_state: dict[int, Any]) -> int:
    return max((st.base_chip_index + len(st.chips)) // config.REPETITION_CHIPS for st in phase_state.values())


def _accept_packet_candidate(
    *,
    candidate: PacketCandidate,
    phase_state: dict[int, Any],
    decoded_packets: int,
) -> tuple[int, str, int]:
    decoded_packets += 1
    packet_status_text = (
        f"Packet: accepted at phase {candidate.phase}, offset {candidate.chip_offset}, "
        f"payload_errors={candidate.payload_errors}"
    )
    print(
        f"\x1b[2K\r[RX PACKET {decoded_packets}] phase={candidate.phase} chip_offset={candidate.chip_offset} "
        f"bit={candidate.header_abs} header_errors={candidate.header_errors} "
        f"preamble_errors={candidate.preamble_errors} sync_errors={candidate.sync_errors} "
        f"confidence={candidate.confidence:.3f} payload_errors={candidate.payload_errors} "
        f"payload_hex={candidate.payload.hex().upper()} "
        f"payload_ascii={safe_ascii(candidate.payload)!r}",
        flush=True,
    )
    phase_state[candidate.phase].last_header_abs_by_offset[candidate.chip_offset] = candidate.header_abs
    return decoded_packets, packet_status_text, config.PACKET_STATUS_HOLD_FRAMES


def accept_best_packet_candidate(
    *,
    candidates: list[PacketCandidate],
    phase_state: dict[int, Any],
    pending_packet_candidate: PacketCandidate | None,
    last_packet_accept_bit: int,
    decoded_packets: int,
    ncc_lock: bool,
    ncc_abs_ema: float,
) -> tuple[PacketCandidate | None, int, int, str | None, int | None]:
    """Select and accept best packet candidate, returning updated packet state."""
    packet_status_text: str | None = None
    packet_status_hold: int | None = None

    current_bit_horizon = _current_bit_horizon(phase_state)
    payload_match_mode = config.PAYLOAD_MATCH_MODE.strip().lower()
    payload_match_enabled = payload_match_mode == "expected"
    payload_error_limit = (
        config.PAYLOAD_MAX_BIT_ERRORS
        if payload_match_enabled
        else config.PAYLOAD_ERROR_DISABLED_LIMIT
    )
    min_confidence = config.PACKET_MIN_CONFIDENCE
    max_preamble_errors_accept = config.PACKET_MAX_PREAMBLE_ERRORS_ACCEPT
    max_sync_errors_accept = config.PACKET_MAX_SYNC_ERRORS_ACCEPT

    if (
        config.PACKET_RELAX_WHEN_LOCKED
        and ncc_lock
        and ncc_abs_ema >= config.PACKET_LOCKED_MIN_NCC_EMA
    ):
        min_confidence = min(min_confidence, config.PACKET_LOCKED_MIN_CONFIDENCE)
        max_preamble_errors_accept = max(
            max_preamble_errors_accept,
            config.PACKET_LOCKED_MAX_PREAMBLE_ERRORS_ACCEPT,
        )
        max_sync_errors_accept = max(
            max_sync_errors_accept,
            config.PACKET_LOCKED_MAX_SYNC_ERRORS_ACCEPT,
        )

    viable = [
        cand
        for cand in candidates
        if cand.confidence >= min_confidence
        and cand.sync_errors <= max_sync_errors_accept
        and cand.preamble_errors <= max_preamble_errors_accept
        and cand.payload_errors <= payload_error_limit
        and cand.preamble_errors <= config.PREAMBLE_MAX_BIT_ERRORS
        and cand.sync_errors <= config.SYNC_MAX_BIT_ERRORS
        and cand.header_abs > last_packet_accept_bit + config.PACKET_MERGE_GAP_BITS
    ]
    if viable:
        viable.sort(key=packet_candidate_sort_key)
        best_new = viable[0]
        if pending_packet_candidate is None:
            pending_packet_candidate = best_new
        elif best_new.header_abs <= (
            pending_packet_candidate.header_abs + config.PACKET_CANDIDATE_SETTLE_BITS
        ):
            if packet_candidate_sort_key(best_new) < packet_candidate_sort_key(pending_packet_candidate):
                pending_packet_candidate = best_new
        else:
            decoded_packets, packet_status_text, packet_status_hold = _accept_packet_candidate(
                candidate=pending_packet_candidate,
                phase_state=phase_state,
                decoded_packets=decoded_packets,
            )
            last_packet_accept_bit = pending_packet_candidate.header_abs
            pending_packet_candidate = best_new

    if (
        pending_packet_candidate is not None
        and current_bit_horizon
        >= (pending_packet_candidate.header_abs + config.PACKET_CANDIDATE_SETTLE_BITS)
    ):
        decoded_packets, packet_status_text, packet_status_hold = _accept_packet_candidate(
            candidate=pending_packet_candidate,
            phase_state=phase_state,
            decoded_packets=decoded_packets,
        )
        last_packet_accept_bit = pending_packet_candidate.header_abs
        pending_packet_candidate = None

    return (
        pending_packet_candidate,
        last_packet_accept_bit,
        decoded_packets,
        packet_status_text,
        packet_status_hold,
    )
