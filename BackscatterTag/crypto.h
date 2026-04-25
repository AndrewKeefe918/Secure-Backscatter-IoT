#ifndef CRYPTO_H
#define CRYPTO_H

/*
 * Minimal AES-128 + AES-CMAC + AES-CTR for MSP430 backscatter tag.
 *
 * Scope: encrypt-only AES-128, plus two modes derived from it:
 *   - AES-CTR for confidentiality
 *   - AES-CMAC for authentication
 *
 * Everything is built on a single primitive (aes128_encrypt_block).
 * All implementations follow published standards:
 *   - AES-128 ECB:  NIST FIPS 197
 *   - AES-CMAC:     RFC 4493
 *   - AES-CTR:      NIST SP 800-38A
 *
 * The corresponding test vectors are checked in crypto.c in comments.
 */

#include <stdint.h>
#include <stddef.h>

#define AES128_KEY_LEN     16U
#define AES128_BLOCK_LEN   16U

/* Round-key schedule storage. AES-128 has 11 round keys * 16 bytes = 176 B. */
typedef struct {
    uint8_t round_key[176];
} aes128_ctx_t;

/* ---- AES-128 ECB encrypt ------------------------------------------------- */

/* Expand the user key into the round-key schedule. Call once per key. */
void aes128_init(aes128_ctx_t *ctx, const uint8_t key[AES128_KEY_LEN]);

/* Encrypt one 16-byte block. `in` and `out` may alias. */
void aes128_encrypt_block(const aes128_ctx_t *ctx,
                          const uint8_t in[AES128_BLOCK_LEN],
                          uint8_t       out[AES128_BLOCK_LEN]);

/* ---- AES-CMAC (RFC 4493) ------------------------------------------------- */

/* Compute the full 16-byte CMAC tag over `msg`. Caller may truncate. */
void aes128_cmac(const aes128_ctx_t *ctx,
                 const uint8_t *msg, size_t msg_len,
                 uint8_t mac[AES128_BLOCK_LEN]);

/* ---- AES-CTR ------------------------------------------------------------- */

/* In-place or out-of-place CTR encrypt/decrypt (XOR with keystream).
 * The `iv` is the initial 128-bit counter; it is incremented as a
 * big-endian integer for each subsequent block, per NIST SP 800-38A. */
void aes128_ctr_xcrypt(const aes128_ctx_t *ctx,
                       const uint8_t iv[AES128_BLOCK_LEN],
                       const uint8_t *in, uint8_t *out, size_t len);

#endif /* CRYPTO_H */
