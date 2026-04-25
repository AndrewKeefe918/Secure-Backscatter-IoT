"""Secure PlutoSDR backscatter receiver package.

Authenticated, encrypted, replay-protected variant of the FSK receiver.
Uses AES-128 + AES-CTR + AES-CMAC with a per-packet monotonic counter.

The on-air packet format after the AA/7E header is:
    COUNTER(4 BE) || CIPHERTEXT(4) || CMAC_TAG(8)   = 16 bytes

The matching firmware is Msp_FSK_secure.c in the firmware/ folder.
"""
