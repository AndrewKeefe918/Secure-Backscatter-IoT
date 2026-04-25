#include <msp430.h>
#include <stdint.h>

/*
 * FSK Backscatter Transmitter (MSP430G2553).
 *
 * Modulation: Binary FSK on the subcarrier.
 *   '1' bit: subcarrier toggles at 1.0 kHz
 *   '0' bit: subcarrier toggles at 1.7 kHz
 *
 * Frequencies chosen so neither is a harmonic of the other:
 *   1 kHz square wave: spectral lines at 1, 3, 5, 7, 9 kHz
 *   1.7 kHz square wave: spectral lines at 1.7, 5.1, 8.5 kHz
 *   No overlap -> receiver can cleanly distinguish the two states.
 *
 * Each bit is transmitted for exactly BIT_DURATION_MS regardless of
 * which frequency is active. Since CCR0 ticks are different durations
 * at different frequencies, we count ticks differently per bit value
 * to keep wall-clock bit duration constant.
 *
 * Packet: [0xAA preamble] [0x7E sync] [O][P][E][N]   (48 data bits)
 *
 * The subcarrier is ALWAYS routed to P1.1 (no P1SEL gating). The two
 * bit states are both "on," just at different frequencies. This is one
 * of the main advantages of FSK over OOK: the receiver always sees
 * signal power, never has to distinguish "signal present" from "noise."
 */

#define PACKET_LENGTH               9U   /* AA + 7E + len(1) + 6 payload */
#define BIT_DURATION_MS             50U

/* CCR0 values for the two subcarrier frequencies.
 * CCR0 = (SMCLK / (2 * f_sub)) - 1, where SMCLK = 1 MHz.
 *   1.0 kHz -> CCR0 = 499  (each tick = 500 us)
 *   1.7 kHz -> CCR0 = 293  (each tick = 294 us, gives 1700.7 Hz)
 */
#define CCR0_FOR_1_BIT              499U   /* 1 kHz subcarrier */
#define CCR0_FOR_0_BIT              293U   /* 1.7 kHz subcarrier */

/* Tick counts to make each bit exactly 50 ms wall-clock regardless
 * of which frequency is active.
 *   At 1.0 kHz: 50 ms / 500 us = 100 ticks
 *   At 1.7 kHz: 50 ms / 294 us = 170 ticks (= 49.98 ms, close enough)
 */
#define TICKS_PER_1_BIT             100U
#define TICKS_PER_0_BIT             170U

/* 2-second gap between packets so the receiver sees a clear silence,
 * making packet-edge detection easier. */
#define GAP_BETWEEN_PACKETS_MS      2000U

const uint8_t packet[PACKET_LENGTH] = { 0xAA, 0x7E, 6, 'H', 'E', 'L', 'P', 'E', 'R' };

/* Block for N CCR0 events. Each tick = (CCR0+1) * 1us at 1 MHz SMCLK. */
static void wait_subcarrier_ticks(uint16_t ticks) {
    while (ticks--) {
        while ((TA0CCTL0 & CCIFG) == 0) { /* spin */ }
        TA0CCTL0 &= ~CCIFG;
    }
}

/* Set the subcarrier to f_1 (for '1') or f_0 (for '0') for one bit period. */
static void transmit_bit(uint8_t bit_value) {
    if (bit_value) {
        TA0CCR0 = CCR0_FOR_1_BIT;
        P1OUT |= BIT6;          /* green LED on for '1' bits */
        wait_subcarrier_ticks(TICKS_PER_1_BIT);
    } else {
        TA0CCR0 = CCR0_FOR_0_BIT;
        P1OUT &= ~BIT6;         /* green LED off for '0' bits */
        wait_subcarrier_ticks(TICKS_PER_0_BIT);
    }
}

static void transmit_byte(uint8_t b) {
    int8_t i;
    for (i = 7; i >= 0; i--) {
        transmit_bit((b >> i) & 0x01);
    }
}

static void transmit_packet(void) {
    uint8_t i;
    P1OUT |= BIT0;              /* red LED on during packet */
    /* Single-bit mode: each packet bit is transmitted exactly once. */
    for (i = 0; i < PACKET_LENGTH; i++) {
        transmit_byte(packet[i]);
    }
    P1OUT &= ~BIT0;
}

/* Approximate ms delay using __delay_cycles at 1 MHz. */
static void delay_ms(uint16_t ms) {
    while (ms--) {
        __delay_cycles(1000);
    }
}

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
    P1OUT  &= ~(BIT0 | BIT1 | BIT6);
    P1SEL2 &= ~BIT1;

    /* Timer_A0: SMCLK / up mode / toggle on CCR0.
     * Start with the '1'-bit frequency. CCR0 will be updated per bit. */
    TA0CCR0  = CCR0_FOR_1_BIT;
    TA0CCTL0 = OUTMOD_4;
    TA0CTL   = TASSEL_2 | MC_1 | TACLR;
    TA0CCTL0 &= ~CCIFG;

    /* Route Timer_A0 output to P1.1 ONCE and never gate it again.
     * Both '1' and '0' bits transmit subcarrier; only frequency changes. */
    P1SEL |= BIT1;

    while (1) {
        transmit_packet();
        delay_ms(GAP_BETWEEN_PACKETS_MS);
    }
}
