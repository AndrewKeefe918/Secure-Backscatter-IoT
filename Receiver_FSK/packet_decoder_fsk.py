"""Bit-level and packet decoding utilities — FSK version.

Identical algorithms to the OOK packet_decoder; only the module
import changes (config_fsk instead of config).
"""

from . import config_fsk as config


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


def find_subsequence(bits: list[int], pattern: list[int], start: int = 0) -> int:
    if not pattern or len(bits) < len(pattern):
        return -1
    max_idx = len(bits) - len(pattern)
    for i in range(max(0, start), max_idx + 1):
        if bits[i : i + len(pattern)] == pattern:
            return i
    return -1


def find_header_match(
    bits: list[int],
    pattern: list[int],
    start: int = 0,
    max_errors: int = 0,
) -> tuple[int, int]:
    """Locate the best (lowest-error) match for `pattern` in `bits` from `start`.

    Returns (index, errors). If no match is at or below max_errors, returns
    (-1, best_errors_seen) so the caller can decide what to do.
    """
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
    """Collapse `REPETITION_CHIPS` consecutive chips into one bit by majority vote.

    With REPETITION_CHIPS = 1 (the FSK default) this is a passthrough — no
    voting happens, each chip is one bit. With higher values the same
    majority rule used by the OOK decoder applies.
    """
    decoded: list[int] = []
    idx = start_offset
    while idx + config.REPETITION_CHIPS <= len(chips):
        triplet = chips[idx : idx + config.REPETITION_CHIPS]
        decoded.append(1 if sum(triplet) >= config.MAJORITY_ONES_THRESHOLD else 0)
        idx += config.REPETITION_CHIPS
    return decoded
