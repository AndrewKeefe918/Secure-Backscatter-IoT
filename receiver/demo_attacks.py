#!/usr/bin/env python3
"""Demonstrate attacks against the live receiver security implementation.

No SDR or tag needed - this builds packets in software using the same
construction as BackscatterTag/Msp_FSK_secure.c, then runs them through the
working receiver path under receiver/ to show what gets accepted and what gets
rejected.

Run from the repo root:
    python -m receiver.demo_attacks

Or directly:
    python receiver/demo_attacks.py
"""

from __future__ import annotations

import os
import struct
import sys
import tempfile
from pathlib import Path

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.cmac import CMAC

if __package__ in (None, ""):
    REPO_ROOT = Path(__file__).resolve().parent.parent
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from receiver import config
    from receiver.secure_packet import AIR_LEN, TAG_LEN, DecodedPacket, SecureReceiver
else:
    from . import config
    from .secure_packet import AIR_LEN, TAG_LEN, DecodedPacket, SecureReceiver


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


def show(description: str, payload: bytes, result: DecodedPacket) -> None:
    verdict = "ACCEPTED" if result.valid else "REJECTED"
    print(f"  {description}")
    print(f"    bytes:  {payload.hex()}")
    print(f"    -> {verdict:8s} counter={result.counter}  reason={result.reason}")


def main() -> int:
    key = bytes.fromhex(config.SHARED_KEY_HEX)
    print(f"Using key: {config.SHARED_KEY_HEX}")
    print("  (this must match SHARED_KEY[] in BackscatterTag/Msp_FSK_secure.c)")
    print("  (demo is using the live receiver implementation under receiver/)")

    # Use a throwaway state file so the demo is repeatable.
    fd, state_path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    state = Path(state_path)
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
    random_payload = os.urandom(AIR_LEN)
    show("16 random bytes", random_payload,
         rx.verify_and_decrypt(random_payload))

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