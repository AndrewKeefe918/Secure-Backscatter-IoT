/*
 * crypto.c - AES-128 + CMAC + CTR for MSP430 backscatter tag.
 *
 * Self-contained, no external dependencies. ~280 lines of C, fits in
 * well under 3 KB of flash on MSP430G2553.
 *
 * Test vectors (verify with any reference AES implementation):
 *
 *   AES-128 ECB encrypt (NIST FIPS 197 Appendix A):
 *     Key:        00 01 02 03 04 05 06 07 08 09 0a 0b 0c 0d 0e 0f
 *     Plaintext:  00 11 22 33 44 55 66 77 88 99 aa bb cc dd ee ff
 *     Ciphertext: 69 c4 e0 d8 6a 7b 04 30 d8 cd b7 80 70 b4 c5 5a
 *
 *   AES-CMAC (RFC 4493 Section 4):
 *     Key:        2b 7e 15 16 28 ae d2 a6 ab f7 15 88 09 cf 4f 3c
 *     Empty msg:  bb 1d 69 29 e9 59 37 28 7f a3 7d 12 9b 75 67 46
 *     16-B msg "6bc1bee22e409f96e93d7e117393172a":
 *                 07 0a 16 b4 6b 4d 41 44 f7 9b dd 9d d0 4a 28 7c
 */

#include "crypto.h"
#include <string.h>

/* ---- AES S-box (Rijndael substitution table) ---------------------------- */

static const uint8_t sbox[256] = {
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
};

/* Round constants for AES-128 key expansion. Index 0 is unused. */
static const uint8_t Rcon[11] = {
    0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36
};

/* ---- AES round operations ----------------------------------------------- */

/* Multiply by x = {02} in GF(2^8) with reduction polynomial 0x11b. */
static uint8_t xtime(uint8_t b) {
    return (uint8_t)((b << 1) ^ (((b >> 7) & 1U) * 0x1bU));
}

static void add_round_key(uint8_t state[16], const uint8_t *rk) {
    uint8_t i;
    for (i = 0; i < 16; i++) state[i] ^= rk[i];
}

static void sub_bytes(uint8_t state[16]) {
    uint8_t i;
    for (i = 0; i < 16; i++) state[i] = sbox[state[i]];
}

/* AES state is column-major: bytes 0..3 are column 0 (rows 0..3 of column 0).
 * ShiftRows rotates row r left by r positions. */
static void shift_rows(uint8_t s[16]) {
    uint8_t t;
    /* Row 1: rotate left by 1 */
    t = s[1]; s[1] = s[5]; s[5] = s[9]; s[9] = s[13]; s[13] = t;
    /* Row 2: rotate left by 2 (= swap pairs) */
    t = s[2]; s[2] = s[10]; s[10] = t;
    t = s[6]; s[6] = s[14]; s[14] = t;
    /* Row 3: rotate left by 3 (= rotate right by 1) */
    t = s[3]; s[3] = s[15]; s[15] = s[11]; s[11] = s[7]; s[7] = t;
}

/* MixColumns: multiply each column by the AES MixColumns matrix in GF(2^8).
 * Identity used: x*a XOR a = xtime(a) XOR a, so for each new byte we can
 * write new = a ^ t ^ xtime(a ^ next), where t = a0 ^ a1 ^ a2 ^ a3. */
static void mix_columns(uint8_t s[16]) {
    uint8_t c, a0, a1, a2, a3, t;
    for (c = 0; c < 4; c++) {
        uint8_t *col = &s[c * 4];
        a0 = col[0]; a1 = col[1]; a2 = col[2]; a3 = col[3];
        t  = (uint8_t)(a0 ^ a1 ^ a2 ^ a3);
        col[0] ^= (uint8_t)(t ^ xtime((uint8_t)(a0 ^ a1)));
        col[1] ^= (uint8_t)(t ^ xtime((uint8_t)(a1 ^ a2)));
        col[2] ^= (uint8_t)(t ^ xtime((uint8_t)(a2 ^ a3)));
        col[3] ^= (uint8_t)(t ^ xtime((uint8_t)(a3 ^ a0)));
    }
}

/* ---- AES-128 key expansion + encrypt ------------------------------------ */

void aes128_init(aes128_ctx_t *ctx, const uint8_t key[16]) {
    uint8_t *rk = ctx->round_key;
    uint8_t i, t[4], tmp;

    /* Round key 0 is the user key. */
    for (i = 0; i < 16; i++) rk[i] = key[i];

    /* Generate 40 more 32-bit words to fill 11 round keys total. */
    for (i = 4; i < 44; i++) {
        /* t = previous word */
        t[0] = rk[(i - 1) * 4 + 0];
        t[1] = rk[(i - 1) * 4 + 1];
        t[2] = rk[(i - 1) * 4 + 2];
        t[3] = rk[(i - 1) * 4 + 3];

        if ((i & 0x03) == 0) {
            /* RotWord: cyclic byte rotation */
            tmp = t[0]; t[0] = t[1]; t[1] = t[2]; t[2] = t[3]; t[3] = tmp;
            /* SubWord: per-byte S-box */
            t[0] = sbox[t[0]];
            t[1] = sbox[t[1]];
            t[2] = sbox[t[2]];
            t[3] = sbox[t[3]];
            /* XOR with round constant */
            t[0] ^= Rcon[i >> 2];
        }

        /* W[i] = W[i-4] XOR t */
        rk[i * 4 + 0] = (uint8_t)(rk[(i - 4) * 4 + 0] ^ t[0]);
        rk[i * 4 + 1] = (uint8_t)(rk[(i - 4) * 4 + 1] ^ t[1]);
        rk[i * 4 + 2] = (uint8_t)(rk[(i - 4) * 4 + 2] ^ t[2]);
        rk[i * 4 + 3] = (uint8_t)(rk[(i - 4) * 4 + 3] ^ t[3]);
    }
}

void aes128_encrypt_block(const aes128_ctx_t *ctx,
                          const uint8_t in[16], uint8_t out[16]) {
    uint8_t state[16];
    uint8_t round;

    memcpy(state, in, 16);

    add_round_key(state, ctx->round_key);
    for (round = 1; round < 10; round++) {
        sub_bytes(state);
        shift_rows(state);
        mix_columns(state);
        add_round_key(state, ctx->round_key + round * 16);
    }
    /* Final round has no MixColumns. */
    sub_bytes(state);
    shift_rows(state);
    add_round_key(state, ctx->round_key + 160);

    memcpy(out, state, 16);
}

/* ---- AES-CMAC (RFC 4493) ------------------------------------------------ */

/* In-place left shift by one bit (16-byte big-endian integer). Returns the
 * MSB that was shifted out, so caller can do the conditional Rb XOR. */
static uint8_t left_shift_one(uint8_t v[16]) {
    uint8_t i, msb = (uint8_t)((v[0] >> 7) & 1U);
    uint8_t carry = 0;
    for (i = 16; i > 0; i--) {
        uint8_t b = v[i - 1];
        v[i - 1] = (uint8_t)((b << 1) | carry);
        carry = (uint8_t)((b >> 7) & 1U);
    }
    return msb;
}

/* Compute CMAC subkeys K1 and K2 from L = AES_K(0^128), per RFC 4493 §2.3. */
static void cmac_subkeys(const aes128_ctx_t *ctx, uint8_t K1[16], uint8_t K2[16]) {
    uint8_t L[16];
    uint8_t zero[16];
    memset(zero, 0, 16);
    aes128_encrypt_block(ctx, zero, L);

    /* K1 = L << 1 (XOR Rb if MSB(L) was 1). Rb for 128-bit is 0x...87. */
    memcpy(K1, L, 16);
    if (left_shift_one(K1)) K1[15] ^= 0x87;

    /* K2 = K1 << 1 (XOR Rb if MSB(K1) was 1). */
    memcpy(K2, K1, 16);
    if (left_shift_one(K2)) K2[15] ^= 0x87;
}

void aes128_cmac(const aes128_ctx_t *ctx,
                 const uint8_t *msg, size_t msg_len,
                 uint8_t mac[16]) {
    uint8_t K1[16], K2[16];
    uint8_t X[16], M_last[16];
    size_t n, i, last_off, last_len, j;
    uint8_t complete;

    cmac_subkeys(ctx, K1, K2);

    if (msg_len == 0) {
        n = 1;
        complete = 0;
    } else {
        n = (msg_len + 15U) / 16U;
        complete = (uint8_t)((msg_len % 16U) == 0U);
    }

    last_off = (n - 1U) * 16U;
    last_len = msg_len - last_off;

    /* Build M_last: either msg[last block] XOR K1 (complete final block)
     * or padded(msg[last block]) XOR K2 (incomplete final block). */
    memset(M_last, 0, 16);
    if (complete) {
        memcpy(M_last, msg + last_off, 16);
        for (j = 0; j < 16; j++) M_last[j] ^= K1[j];
    } else {
        if (last_len > 0) memcpy(M_last, msg + last_off, last_len);
        M_last[last_len] = 0x80;
        for (j = 0; j < 16; j++) M_last[j] ^= K2[j];
    }

    /* CBC-MAC chain over the first n-1 blocks (none in our use case). */
    memset(X, 0, 16);
    for (i = 0; i + 1 < n; i++) {
        for (j = 0; j < 16; j++) X[j] ^= msg[i * 16 + j];
        aes128_encrypt_block(ctx, X, X);
    }

    /* Final block. */
    for (j = 0; j < 16; j++) X[j] ^= M_last[j];
    aes128_encrypt_block(ctx, X, mac);
}

/* ---- AES-CTR (NIST SP 800-38A) ------------------------------------------ */

/* Increment the 128-bit counter as a big-endian integer in place. */
static void ctr_increment(uint8_t c[16]) {
    int i;
    for (i = 15; i >= 0; i--) {
        c[i] = (uint8_t)(c[i] + 1U);
        if (c[i] != 0U) break;
    }
}

void aes128_ctr_xcrypt(const aes128_ctx_t *ctx,
                       const uint8_t iv[16],
                       const uint8_t *in, uint8_t *out, size_t len) {
    uint8_t counter[16];
    uint8_t keystream[16];
    size_t pos = 0, i, chunk;

    memcpy(counter, iv, 16);

    while (pos < len) {
        aes128_encrypt_block(ctx, counter, keystream);
        chunk = (len - pos > 16U) ? 16U : (len - pos);
        for (i = 0; i < chunk; i++) {
            out[pos + i] = (uint8_t)(in[pos + i] ^ keystream[i]);
        }
        pos += chunk;
        ctr_increment(counter);
    }
}
