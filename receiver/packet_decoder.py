"""Bit-level and packet decoding utilities."""

from . import config


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
    decoded: list[int] = []
    idx = start_offset
    while idx + config.REPETITION_CHIPS <= len(chips):
        triplet = chips[idx : idx + config.REPETITION_CHIPS]
        decoded.append(1 if sum(triplet) >= 2 else 0)
        idx += config.REPETITION_CHIPS
    return decoded
