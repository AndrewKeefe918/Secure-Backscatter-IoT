"""Bit-level and packet decoding utilities â€” FSK version.

Identical algorithms to the OOK packet_decoder; only the module
import changes (config instead of config).
"""

from . import config as config


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


def bits_to_text(bits: list[int]) -> str:
    return "".join("1" if b else "0" for b in bits)


def majority_decode_triplets(chips: list[int], start_offset: int) -> list[int]:
    """Collapse `REPETITION_CHIPS` consecutive chips into one bit by majority vote.

    With REPETITION_CHIPS = 1 (the FSK default) this is a passthrough â€” no
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

