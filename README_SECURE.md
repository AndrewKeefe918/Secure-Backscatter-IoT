# Secure FSK Backscatter — Run Guide

AES-128 + AES-CMAC + monotonic-counter version of the FSK backscatter link.
Same FSK modulation and bit timing as the original; the on-air payload
after the AA/7E header is now 16 bytes (`COUNTER || CIPHERTEXT || TAG`).

## File layout

The download contains 15 files. Put them in two places:

**Tag firmware** (Code Composer Studio project):
- `Msp_FSK_secure.c` — main firmware (replaces `Msp_FSK.c`)
- `crypto.h`         — AES-128 / CMAC / CTR API
- `crypto.c`         — implementations

Add all three to your CCS project. Build, flash to the G2553.

**Python receiver package.** Create a new directory next to the existing
`Receiver_FSK/` and drop the rest of the files in:

```
your_repo/
├── Receiver_FSK/             (existing, unchanged)
└── Receiver_FSK_secure/      (NEW)
    ├── __init__.py
    ├── config_secure.py
    ├── dsp_secure.py
    ├── gui_setup_secure.py
    ├── main_secure.py
    ├── packet_decoder_secure.py
    ├── receiver_loop_secure.py
    ├── receiver_loop_rx_only_secure.py
    ├── offline_decoder_secure.py
    ├── rx_monitor_secure.py
    ├── secure_packet.py
    └── demo_attacks.py
```

The `Receiver_FSK_secure/captures/` directory is created automatically
on first run.

## Pre-shared key — must match in two files

The 16-byte AES key is hard-coded in two places:

| File                    | Where               | Default                              |
| ----------------------- | ------------------- | ------------------------------------ |
| `Msp_FSK_secure.c`      | `SHARED_KEY[]`      | RFC 4493 demo key `2b 7e 15 16 ...`  |
| `config_secure.py`      | `SHARED_KEY_HEX`    | `"2b7e151628aed2a6abf7158809cf4f3c"` |

They must agree byte-for-byte. Change both before any deployment, then
re-flash and restart the receiver.

## How to run

**1. Flash the tag.** Build `Msp_FSK_secure.c` + `crypto.c` (with
`crypto.h` on the include path). Flash to the MSP430G2553. The red LED
on P1.0 blinks once per ~7-second packet.

**2. Start the exciter.** Same Pi + Pluto setup as before. No changes.

**3. Run the receiver:**
```
python -m Receiver_FSK_secure.main_secure
```
You should see lines like:
```
[RX PACKET 1] phase=0 chip_offset=0 bit=24 primary header_errors=0 \
              counter=100 plaintext=b'OPEN'  AUTHENTICATED
```
The counter increments by 1 per packet and jumps by ~100 across tag
reboots (that's the flash-skip-ahead behavior; expected).

**4. Demonstrate the security (no SDR needed):**
```
python -m Receiver_FSK_secure.demo_attacks
```
This builds legitimate and adversarial packets in software and feeds
them to the receiver. It shows which get accepted vs rejected and why.
Use it during a presentation to make the security claim concrete.

## Resetting replay state

If you re-flash the tag, its counter starts over from a low value, but
the receiver still remembers the highest counter ever accepted, so it
will reject the first batch of packets as replays. To reset:
```
rm Receiver_FSK_secure/captures/secure_rx_state.json
```
Then restart the receiver. It will accept the next packet whatever its
counter value, and start tracking forward from there.

## What the secure version actually buys you

| Attack                                | Original FSK          | Secure FSK                         |
| ------------------------------------- | --------------------- | ---------------------------------- |
| Eavesdrop the message                 | Reads "OPEN" trivially | Reads ciphertext (no useful info)  |
| Record one packet and replay later    | **Door opens**        | REJECT (counter ≤ last_accepted)   |
| Build a packet without the key        | **Door opens**        | REJECT (MAC verification failed)   |
| Flip one bit in transit               | Door opens or BER fix | REJECT (MAC verification failed)   |

`demo_attacks.py` runs through each of these in order and prints the
reject reason, so the security model is visible end-to-end.

## Quick sanity checks if something looks off

- Receiver prints `REJECT [MAC verification failed]` on every packet →
  the keys don't match. Re-check `SHARED_KEY[]` (firmware) vs
  `SHARED_KEY_HEX` (config) byte-for-byte.
- Receiver prints `REJECT [replay: ...]` on every packet immediately
  after re-flashing → delete `secure_rx_state.json` (see above).
- Receiver finds no headers at all → unrelated to crypto; check the
  spectrum/waterfall plots like you would for the non-secure system.
- `demo_attacks.py` accepts something it shouldn't → something is wrong
  with `secure_packet.py` or `crypto.c`. Re-run the self-test:
  `python -m Receiver_FSK_secure.secure_packet`.
