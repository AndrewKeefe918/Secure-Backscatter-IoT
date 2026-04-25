#include <msp430.h>
#include <stdint.h>
#include <string.h>
#include "crypto.h"

/*
 * Msp_FSK_secure.c - drop-in replacement for Msp_FSK.c with AEAD + replay protection.
 *
 * The FSK transmission machinery is unchanged: binary FSK with 1.0 kHz for '1'
 * bits and 1.7 kHz for '0' bits, 50 ms per bit, subcarrier always on. What
 * changes:
 *
 *   1. The fixed 6-byte packet { 0xAA, 0x7E, 'O', 'P', 'E', 'N' } is replaced
 *      by an 18-byte packet:
 *           PREAMBLE(0xAA) || SYNC(0x7E) || COUNTER(4 BE) || CIPHERTEXT(4) || CMAC_TAG(8)
 *
 *   2. The 3-chip repetition coding (transmit_byte_rep3) is dropped: the CMAC
 *      tag provides cryptographic integrity that strictly dominates 3-chip
 *      majority voting. Single-chip transmit_byte is used for every bit.
 *      Air-time per packet: 18 * 8 * 50 ms = 7.2 s (matches the original
 *      6-byte * 3-chip rep packet duration).
 *
 *   3. A monotonic 32-bit counter is persisted to flash Info Segment B
 *      (0x1080) every COUNTER_PERSIST_INTERVAL transmissions. On boot the
 *      stored value is read, advanced by the persistence interval, and
 *      written back - so the counter never goes backwards across reboots
 *      even though we don't write to flash on every transmission. With
 *      INTERVAL = 100, segment endurance (~10K erase cycles) translates to
 *      ~1M transmissions, decades for a low-duty access-control tag.
 *
 *   4. The 16-byte pre-shared key SHARED_KEY[] must match the receiver's
 *      SHARED_KEY_HEX in config_fsk.py.
 *
 * Note on low-power: this firmware busy-waits on the timer; LPM3 between
 * packets is a separate change. Counter persistence is already correct for
 * battery operation since flash is non-volatile.
 */

/* ---- FSK transmission constants (unchanged from Msp_FSK.c) -------------- */

#define BIT_DURATION_MS             50U
#define CCR0_FOR_1_BIT              499U   /* 1.0 kHz subcarrier */
#define CCR0_FOR_0_BIT              293U   /* 1.7 kHz subcarrier */
#define TICKS_PER_1_BIT             100U
#define TICKS_PER_0_BIT             170U
#define GAP_BETWEEN_PACKETS_MS      2000U

/* ---- Pre-shared key (MUST match receiver) ------------------------------- */
/* Demo key from RFC 4493 test vectors. CHANGE THIS for any real deployment.
 * Production should provision a per-device key at programming time and
 * derive K_enc and K_mac via a one-line KDF (see crypto.c comments). */
static const uint8_t SHARED_KEY[16] = {
    0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
    0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c
};

/* ---- Plaintext command -------------------------------------------------- */
/* P1.3 is an active-low reed input with the internal pull-up enabled:
 *   - switch open / disconnected -> logic 1 -> transmit "OPEN"
 *   - switch shut / grounded     -> logic 0 -> stay silent */
#define REED_SWITCH_BIT  BIT3
#define PT_LEN  4U
static const uint8_t PT_OPEN[PT_LEN] = { 'O', 'P', 'E', 'N' };

/* ---- Packet layout ------------------------------------------------------ */
#define PREAMBLE_BYTE       0xAAU
#define SYNC_BYTE           0x7EU
#define COUNTER_LEN         4U
#define TAG_LEN             8U
#define AIR_PAYLOAD_LEN     (COUNTER_LEN + PT_LEN + TAG_LEN)   /* 16 bytes */
#define PACKET_LEN          (2U + AIR_PAYLOAD_LEN)             /* 18 bytes */

static uint8_t packet_buf[PACKET_LEN];

/* ---- Counter persistence (Info Segment B) ------------------------------- */

/* Info Segment B at 0x1080-0x10BF is reserved for user data on G2553.
 * DO NOT use Info Segment A (0x10C0-0x10FF) - it holds the DCO calibration
 * data we read at boot. Erasing A would brick the device. */
#define COUNTER_FLASH_ADDR        0x1080U
#define COUNTER_PERSIST_INTERVAL  100UL

static aes128_ctx_t aes_ctx;
static uint32_t     tx_counter;

static uint32_t flash_read_counter(void) {
    /* Read 4 bytes big-endian to match the on-air counter encoding. */
    volatile uint8_t *p = (volatile uint8_t *)COUNTER_FLASH_ADDR;
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] <<  8) | ((uint32_t)p[3]);
}

static void flash_write_counter(uint32_t value) {
    /* MSP430G2553 flash timing generator must be in 257-476 kHz range.
     * SMCLK = 1 MHz, divider = 3 (FN1 set) -> 333 kHz. */
    volatile uint8_t *p = (volatile uint8_t *)COUNTER_FLASH_ADDR;

    __disable_interrupt();
    FCTL2 = FWKEY | FSSEL_2 | FN1;          /* SMCLK / 3 */
    FCTL3 = FWKEY;                           /* unlock (LOCK = 0) */

    /* Erase the whole segment (64 bytes -> all 0xFF). */
    FCTL1 = FWKEY | ERASE;
    *p = 0;                                  /* dummy write triggers erase */
    while (FCTL3 & BUSY) { /* spin */ }

    /* Write 4 bytes big-endian. */
    FCTL1 = FWKEY | WRT;
    p[0] = (uint8_t)(value >> 24); while (FCTL3 & BUSY) { }
    p[1] = (uint8_t)(value >> 16); while (FCTL3 & BUSY) { }
    p[2] = (uint8_t)(value >>  8); while (FCTL3 & BUSY) { }
    p[3] = (uint8_t) value;        while (FCTL3 & BUSY) { }

    FCTL1 = FWKEY;                           /* exit write mode */
    FCTL3 = FWKEY | LOCK;                    /* re-lock */
    __enable_interrupt();
}

/* On boot: read stored counter, skip ahead by INTERVAL (in case we
 * transmitted up to INTERVAL-1 unsaved counters before the last reboot),
 * and persist the new starting value. The receiver tolerates any monotonic
 * jump, so a skipped range is harmless. */
static void counter_init(void) {
    uint32_t stored = flash_read_counter();
    if (stored == 0xFFFFFFFFUL) {
        stored = 0UL;                        /* fresh / erased segment */
    }
    tx_counter = stored + COUNTER_PERSIST_INTERVAL;
    flash_write_counter(tx_counter);
}

static uint32_t counter_next(void) {
    uint32_t c = tx_counter;
    tx_counter++;
    if ((tx_counter % COUNTER_PERSIST_INTERVAL) == 0UL) {
        flash_write_counter(tx_counter);
    }
    return c;
}

/* ---- FSK transmission primitives (verbatim from Msp_FSK.c) -------------- */

static void wait_subcarrier_ticks(uint16_t ticks) {
    while (ticks--) {
        while ((TA0CCTL0 & CCIFG) == 0) { /* spin */ }
        TA0CCTL0 &= ~CCIFG;
    }
}

static void transmit_bit(uint8_t bit_value) {
    if (bit_value) {
        TA0CCR0 = CCR0_FOR_1_BIT;
        P1OUT |= BIT6;
        wait_subcarrier_ticks(TICKS_PER_1_BIT);
    } else {
        TA0CCR0 = CCR0_FOR_0_BIT;
        P1OUT &= ~BIT6;
        wait_subcarrier_ticks(TICKS_PER_0_BIT);
    }
}

static void transmit_byte(uint8_t b) {
    int8_t i;
    for (i = 7; i >= 0; i--) {
        transmit_bit((b >> i) & 0x01U);
    }
}

/* Note: transmit_byte_rep3 is intentionally absent. CMAC handles integrity. */

/* ---- Packet construction ------------------------------------------------ */

static uint8_t reed_switch_is_open(void) {
    return (P1IN & REED_SWITCH_BIT) != 0U;
}

static void subcarrier_enable(void) {
    TA0CCR0  = CCR0_FOR_1_BIT;
    TA0CCTL0 &= ~CCIFG;
    TA0CTL   = TASSEL_2 | MC_1 | TACLR;
    P1SEL   |= BIT1;
}

static void subcarrier_disable(void) {
    P1SEL   &= ~BIT1;
    P1OUT   &= ~BIT1;
    TA0CTL   = TASSEL_2 | MC_0 | TACLR;
    TA0CCTL0 &= ~CCIFG;
}

static void build_packet(void) {
    uint32_t ctr;
    uint8_t  iv[16];
    uint8_t  full_tag[16];

    packet_buf[0] = PREAMBLE_BYTE;
    packet_buf[1] = SYNC_BYTE;

    ctr = counter_next();

    /* Counter (4 bytes, big-endian) at packet_buf[2..5]. */
    packet_buf[2] = (uint8_t)(ctr >> 24);
    packet_buf[3] = (uint8_t)(ctr >> 16);
    packet_buf[4] = (uint8_t)(ctr >>  8);
    packet_buf[5] = (uint8_t) ctr;

    /* IV = counter (4B BE) || 12 zero bytes. Unique per packet because the
     * counter never repeats. The trailing 12 bytes leave room for in-block
     * counter increments if pt_len ever grows beyond 16 bytes (NIST CTR). */
    iv[0] = packet_buf[2];
    iv[1] = packet_buf[3];
    iv[2] = packet_buf[4];
    iv[3] = packet_buf[5];
    memset(&iv[4], 0, 12);

    /* Encrypt the fixed OPEN command with AES-CTR -> packet_buf[6..9]. */
    aes128_ctr_xcrypt(&aes_ctx, iv, PT_OPEN, &packet_buf[6], PT_LEN);

    /* CMAC over (counter || ciphertext) = packet_buf[2..9], 8 bytes total. */
    aes128_cmac(&aes_ctx, &packet_buf[2], COUNTER_LEN + PT_LEN, full_tag);

    /* Truncate the 16-byte CMAC tag to 8 bytes -> packet_buf[10..17]. */
    memcpy(&packet_buf[10], full_tag, TAG_LEN);
}

/* ---- Packet transmission ------------------------------------------------ */

static void transmit_packet(void) {
    uint8_t i;
    P1OUT |= BIT0;                           /* red LED on during packet */
    /* Single chip per bit - no repetition. */
    for (i = 0; i < PACKET_LEN; i++) {
        transmit_byte(packet_buf[i]);
    }
    P1OUT &= ~BIT0;
}

static void delay_ms(uint16_t ms) {
    while (ms--) {
        __delay_cycles(1000);
    }
}

/* ---- Main --------------------------------------------------------------- */

int main(void) {
    WDTCTL = WDTPW | WDTHOLD;

    /* Load 1 MHz DCO calibration. Fall back to slow blink if erased. */
    {
        volatile unsigned char *p_calbc1 = (volatile unsigned char *)0x10FFu;
        volatile unsigned char *p_caldco = (volatile unsigned char *)0x10FEu;
        unsigned char cal_bc1 = *p_calbc1;
        unsigned char cal_dco = *p_caldco;
        if (cal_bc1 == 0xFF || cal_dco == 0xFF) {
            P1DIR |= BIT0;
            while (1) {
                P1OUT ^= BIT0;
                __delay_cycles(50000);
            }
        }
        DCOCTL  = 0;
        BCSCTL1 = cal_bc1;
        DCOCTL  = cal_dco;
        BCSCTL2 = 0;
    }

    /* GPIO setup */
    P1DIR  |=  (BIT0 | BIT1 | BIT6);
    P1DIR  &= ~REED_SWITCH_BIT;
    P1OUT  &= ~(BIT0 | BIT1 | BIT6);
    P1OUT  |= REED_SWITCH_BIT;
    P1REN  |= REED_SWITCH_BIT;
    P1SEL  &= ~REED_SWITCH_BIT;
    P1SEL2 &= ~(BIT1 | REED_SWITCH_BIT);

    /* Timer_A0 setup: leave the timer disconnected until the reed opens. */
    TA0CCR0  = CCR0_FOR_1_BIT;
    TA0CCTL0 = OUTMOD_4;
    subcarrier_disable();

    /* Crypto + counter init must come AFTER clock setup (flash needs SMCLK). */
    aes128_init(&aes_ctx, SHARED_KEY);
    counter_init();

    while (1) {
        if (reed_switch_is_open()) {
            subcarrier_enable();
            build_packet();
            transmit_packet();
            delay_ms(GAP_BETWEEN_PACKETS_MS);
        } else {
            subcarrier_disable();
            P1OUT &= ~(BIT0 | BIT6);
            delay_ms(100U);
        }
    }
}
