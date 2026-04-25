#!/usr/bin/env python3
"""Demonstrate that the secure FSK link rejects forgery, replay, and tampering.

No SDR or tag needed - this builds packets in software using the same
construction as Msp_FSK_secure.c, then runs them through SecureReceiver
to show what gets accepted and what gets rejected.

Run from above the package:
    python -m Receiver_FSK_secure.demo_attacks

Or from inside the directory:
    python demo_attacks.py
"""

from __future__ import annotations

import os
import struct
import sys
import tempfile
from pathlib import Path

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.cmac import CMAC

# Allow running both as a package module and as a direct script.
try:
    from . import config_secure as config
    from .secure_packet import SecureReceiver, AIR_LEN, TAG_LEN
except ImportError:
    import config_secure as config
    from secure_packet import SecureReceiver, AIR_LEN, TAG_LEN


def build_packet(key: bytes, counter: int, plaintext: bytes = b"OPEN") -> bytes:
    """Build the 16-byte payload exactly the way Msp_FSK_secure.c does."""
    iv = struct.pack(">I", counter) + b"\x00" * 12
    enc = Cipher(algorithms.AES(key), modes.CTR(iv)).encryptor()
    ciphertext = enc.update(plaintext) + enc.finalize()

    counter_bytes = struct.pack(">I", counter)
    cmac = CMAC(algorithms.AES(key))
    cmac.update(counter_bytes + ciphertext)
    tag = cmac.finalize()[:TAG_LEN]

    return counter_bytes + ciphertext + tag


def banner(n: int, label: str) -> None:
    bar = "-" * max(4, 60 - len(label))
    print(f"\n---- Scenario {n}: {label} {bar}")


def show(description: str, payload: bytes, result) -> None:
    verdict = "ACCEPTED" if result.valid else "REJECTED"
    print(f"  {description}")
    print(f"    bytes:  {payload.hex()}")
    print(f"    -> {verdict:8s} counter={result.counter}  reason={result.reason}")


def main() -> int:
    key = bytes.fromhex(config.SHARED_KEY_HEX)
    print(f"Using key: {config.SHARED_KEY_HEX}")
    print("  (this must match SHARED_KEY[] in Msp_FSK_secure.c)")

    # Use a throwaway state file so the demo is repeatable.
    state = Path(tempfile.mkstemp(suffix=".json")[1])
    state.unlink()
    rx = SecureReceiver(key, state_path=state)

    # ----- 1. Legitimate ----------------------------------------------------
    banner(1, "Legitimate packet from the real tag")
    pkt = build_packet(key, counter=100)
    show("first packet, counter=100", pkt, rx.verify_and_decrypt(pkt))

    # ----- 2. Replay --------------------------------------------------------
    banner(2, "Replay attack (adversary records and re-emits)")
    show("same packet replayed", pkt, rx.verify_and_decrypt(pkt))
    older = build_packet(key, counter=50)
    show("older recording (counter=50) replayed", older,
         rx.verify_and_decrypt(older))

    # ----- 3. Tampering -----------------------------------------------------
    banner(3, "Tampering: single bit flip on a fresh-counter packet")
    fresh = build_packet(key, counter=200)
    tampered = bytearray(fresh)
    tampered[5] ^= 0x10
    print(f"    original:  {fresh.hex()}")
    show("after flipping one bit in the ciphertext", bytes(tampered),
         rx.verify_and_decrypt(bytes(tampered)))

    # ----- 4. Forgery -------------------------------------------------------
    banner(4, "Forgery: adversary builds a packet without the real key")
    forged = build_packet(b"\x00" * 16, counter=300)
    show("packet built with all-zero key", forged,
         rx.verify_and_decrypt(forged))
    show("16 random bytes", os.urandom(AIR_LEN),
         rx.verify_and_decrypt(os.urandom(AIR_LEN)))

    # ----- 5. Real traffic still works after all the attacks ---------------
    banner(5, "Legitimate traffic still works after the attacks")
    pkt5 = build_packet(key, counter=500)
    show("new packet, counter=500", pkt5, rx.verify_and_decrypt(pkt5))

    # ----- Summary ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("  ACCEPTED: 2 legitimate packets (counters 100 and 500)")
    print("  REJECTED: 2 replays, 1 tamper, 2 forgeries")
    print()
    print("Without auth + replay protection, the on-air bytes 'AA 7E O P E N'")
    print("are publicly observable, so scenarios 2-4 would all OPEN THE DOOR.")

    state.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
