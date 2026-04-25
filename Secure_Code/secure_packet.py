"""Receiver-side AEAD verification and decryption for the secure FSK link.

Wire format (after the 0xAA preamble and 0x7E sync):
    COUNTER (4B big-endian) || CIPHERTEXT (4B) || CMAC_TAG (8B)   = 16 bytes

Construction (must match Msp_FSK_secure.c exactly):
    - AES-128 with shared 16-byte key K
    - ciphertext = AES-CTR(K, IV = counter(4B BE) || 12*0x00, plaintext)
    - tag        = AES-CMAC(K, counter || ciphertext)[truncated to 8 bytes]

Replay protection:
    - The receiver tracks the highest counter ever accepted, persistently
      across runs, in a small JSON file.
    - Strictly require received_counter > last_accepted; reject otherwise.
    - The replay check runs BEFORE any AES work, so a flood of bogus
      packets gets rejected cheaply.

Integration:
    >>> from .secure_packet_fsk import SecureReceiver
    >>> rx = SecureReceiver(bytes.fromhex(config.SHARED_KEY_HEX),
    ...                     Path(config.SECURE_RX_STATE_PATH))
    >>> result = rx.verify_and_decrypt(payload_bytes)  # 16 bytes
    >>> if result.valid:
    ...     handle_open(result.counter)

State reset (when re-flashing the tag): delete the JSON state file. The
receiver will then accept any counter as fresh.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.cmac import CMAC


# ---- Constants (must match firmware) ----------------------------------------

KEY_LEN = 16
TAG_LEN = 8
COUNTER_LEN = 4
PT_LEN = 4
AIR_LEN = COUNTER_LEN + PT_LEN + TAG_LEN  # 16 bytes total payload

EXPECTED_PT = b"OPEN"


# ---- Result type ------------------------------------------------------------

@dataclass
class DecodedPacket:
    """Result of verify_and_decrypt.

    Always populated; check `valid` to decide whether to act. The `reason`
    string is intended for logs - it tells you exactly why a packet was
    rejected (length, replay, MAC, plaintext mismatch).
    """
    counter: int
    plaintext: bytes
    valid: bool
    reason: str


# ---- Receiver ---------------------------------------------------------------

class SecureReceiver:
    """Verifies and decrypts secured FSK packets, tracking replay state."""

    def __init__(self, key: bytes, state_path: Optional[Path] = None) -> None:
        if len(key) != KEY_LEN:
            raise ValueError(f"key must be {KEY_LEN} bytes, got {len(key)}")
        self.key = key
        self.state_path = Path(state_path) if state_path is not None else None
        self.last_counter = self._load_state()

    # ---- Persistent state ---------------------------------------------------

    def _load_state(self) -> int:
        if self.state_path is None or not self.state_path.exists():
            return -1
        try:
            data = json.loads(self.state_path.read_text())
            return int(data.get("last_counter", -1))
        except (OSError, ValueError, TypeError):
            return -1

    def _save_state(self) -> None:
        if self.state_path is None:
            return
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(
                json.dumps({"last_counter": self.last_counter})
            )
        except OSError:
            # Persisting failed; the in-memory counter is still authoritative
            # for the rest of this run. The next run will start fresh, which
            # is acceptable as long as the tag's flash counter has advanced.
            pass

    # ---- Crypto primitives --------------------------------------------------

    def _ctr_xcrypt(self, counter: int, data: bytes) -> bytes:
        # IV format: 4-byte BE counter || 12 zero bytes.
        # Matches Msp_FSK_secure.c build_packet().
        iv = struct.pack(">I", counter) + b"\x00" * 12
        cipher = Cipher(algorithms.AES(self.key), modes.CTR(iv))
        encryptor = cipher.encryptor()
        return encryptor.update(data) + encryptor.finalize()

    def _cmac_full(self, msg: bytes) -> bytes:
        mac = CMAC(algorithms.AES(self.key))
        mac.update(msg)
        return mac.finalize()  # full 16-byte tag; caller truncates

    # ---- Main entry point ---------------------------------------------------

    def verify_and_decrypt(self, payload: bytes) -> DecodedPacket:
        """Validate, decrypt, and update replay state for one packet payload.

        `payload` is the 16 bytes that follow the AA/7E header on the air.
        The header itself is not part of the MAC and should already be
        stripped by the caller's framing logic.
        """
        if len(payload) != AIR_LEN:
            return DecodedPacket(
                counter=0, plaintext=b"", valid=False,
                reason=f"length: {len(payload)} != {AIR_LEN}",
            )

        counter = struct.unpack(">I", payload[:COUNTER_LEN])[0]
        ciphertext = payload[COUNTER_LEN : COUNTER_LEN + PT_LEN]
        received_tag = payload[COUNTER_LEN + PT_LEN :]

        # 1. Replay check FIRST - cheap rejection of resends and floods.
        if counter <= self.last_counter:
            return DecodedPacket(
                counter=counter, plaintext=b"", valid=False,
                reason=(f"replay: counter {counter} "
                        f"<= last_accepted {self.last_counter}"),
            )

        # 2. MAC verification (constant-time compare).
        expected_tag = self._cmac_full(payload[: COUNTER_LEN + PT_LEN])[:TAG_LEN]
        if not _ct_eq(received_tag, expected_tag):
            return DecodedPacket(
                counter=counter, plaintext=b"", valid=False,
                reason="MAC verification failed",
            )

        # 3. Decrypt. Only safe to do AFTER MAC verifies (encrypt-then-MAC).
        plaintext = self._ctr_xcrypt(counter, ciphertext)

        # 4. Optional plaintext sanity check. Comment out if you want to
        #    accept any well-formed plaintext (e.g., for variable commands).
        if plaintext != EXPECTED_PT:
            return DecodedPacket(
                counter=counter, plaintext=plaintext, valid=False,
                reason=f"unexpected plaintext: {plaintext!r}",
            )

        # 5. Commit replay state and return success.
        self.last_counter = counter
        self._save_state()
        return DecodedPacket(
            counter=counter, plaintext=plaintext, valid=True, reason="OK",
        )


def _ct_eq(a: bytes, b: bytes) -> bool:
    """Constant-time bytes comparison. Avoids tag-comparison timing leaks."""
    if len(a) != len(b):
        return False
    diff = 0
    for x, y in zip(a, b):
        diff |= x ^ y
    return diff == 0


# ---- Self-test --------------------------------------------------------------

if __name__ == "__main__":
    # Round-trip check: synthesize a packet using the same construction the
    # firmware uses, then verify it. If this passes, the receiver is
    # internally consistent. To verify against the FIRMWARE specifically,
    # capture a real packet's 16-byte payload and feed it in here with the
    # same SHARED_KEY.

    import sys

    test_key = bytes.fromhex("2b7e151628aed2a6abf7158809cf4f3c")
    test_counter = 12345
    test_pt = b"OPEN"

    # Synthesize a packet exactly the way Msp_FSK_secure.c does.
    iv = struct.pack(">I", test_counter) + b"\x00" * 12
    enc = Cipher(algorithms.AES(test_key), modes.CTR(iv)).encryptor()
    test_ct = enc.update(test_pt) + enc.finalize()

    counter_bytes = struct.pack(">I", test_counter)
    cmac_full = CMAC(algorithms.AES(test_key))
    cmac_full.update(counter_bytes + test_ct)
    test_tag = cmac_full.finalize()[:TAG_LEN]

    payload = counter_bytes + test_ct + test_tag
    print(f"synthetic payload  ({len(payload)} bytes): {payload.hex()}")
    print(f"  counter    = {test_counter}")
    print(f"  ciphertext = {test_ct.hex()}")
    print(f"  tag        = {test_tag.hex()}")

    # Verify.
    rx = SecureReceiver(test_key, state_path=None)
    r1 = rx.verify_and_decrypt(payload)
    assert r1.valid, f"clean packet rejected: {r1.reason}"
    assert r1.plaintext == test_pt
    print(f"OK clean packet:   counter={r1.counter}, pt={r1.plaintext!r}")

    # Replay must be rejected.
    r2 = rx.verify_and_decrypt(payload)
    assert not r2.valid and "replay" in r2.reason, "replay check failed"
    print(f"OK replay rejected: {r2.reason}")

    # Tampering must be rejected (flip a bit anywhere in the payload).
    tampered = bytearray(payload)
    tampered[5] ^= 0x10                       # corrupt the ciphertext
    rx2 = SecureReceiver(test_key, state_path=None)
    r3 = rx2.verify_and_decrypt(bytes(tampered))
    assert not r3.valid and "MAC" in r3.reason, "MAC check failed"
    print(f"OK tamper rejected: {r3.reason}")

    # Wrong-length payload must be rejected.
    r4 = SecureReceiver(test_key).verify_and_decrypt(payload[:-1])
    assert not r4.valid and "length" in r4.reason
    print(f"OK length rejected: {r4.reason}")

    print("All self-tests passed.")
    sys.exit(0)
