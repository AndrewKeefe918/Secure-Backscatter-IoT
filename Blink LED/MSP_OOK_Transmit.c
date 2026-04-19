#include <msp430.h>
#include <stdint.h>

/**
 * SECURE BACKSCATTER TRANSMITTER
 * 
 * Protocol: On-Off Keying (OOK) of a 1 kHz subcarrier
 * 
 * Packet: [0xAA preamble] [0x7E sync] [O][P][E][N]
 * Total: 6 bytes = 48 bits per packet
 * 
 * Bit encoding:
 *   '1' bit: 1 kHz subcarrier ON  (sidebands present)
 *   '0' bit: 1 kHz subcarrier OFF (no sidebands)
 * 
 * Timing: 50 subcarrier periods per bit (nominally 50 ms at 1 kHz)
 */

#define PACKET_LENGTH           6
#define SUBCARRIER_HZ_NOMINAL        1000U
/* CCR0 events occur every 500 us (toggle points), i.e. half a full
 * 1 kHz square-wave period. Use "ticks" to avoid half-period mistakes. */
#define BIT_SUBCARRIER_TICKS         100U   /* 50 ms nominal */
#define GAP_BETWEEN_PACKETS_TICKS    4000U  /* 2 s nominal */

const uint8_t packet[PACKET_LENGTH] = {
    0xAA,
    0x7E,
    'O',
    'P',
    'E',
    'N'
};

/* Wait for N subcarrier timer periods using CCR0 flag events.
 * This keeps OOK bit timing locked to the same timer that generates
 * the subcarrier, avoiding software delay drift and clock mismatch. */
static void wait_subcarrier_ticks(uint16_t ticks) {
    while (ticks--) {
        while ((TA0CCTL0 & CCIFG) == 0) {
            /* spin */
        }
        TA0CCTL0 &= ~CCIFG;
    }
} /* end wait_subcarrier_ticks */

/* Route the free-running 1 kHz timer output onto P1.1.
 *
 * The timer is started once in main() and NEVER stopped or cleared.
 * We gate the subcarrier only by (dis)connecting the pin via P1SEL,
 * so the 1 kHz phase stays continuous across bit boundaries. Stopping
 * or TACLR'ing the timer on every '1' (as the previous version did)
 * restarted the phase each bit, producing broadband spectral splatter
 * around the +/- 1 kHz sidebands.
 */
static void subcarrier_on(void) {
    P1SEL |= BIT1;
    P1OUT |= BIT6;
} /* end subcarrier_on */

static void subcarrier_off(void) {
    P1SEL &= ~BIT1;
    P1OUT &= ~BIT1;
    P1OUT &= ~BIT6;
} /* end subcarrier_off */

void transmit_byte(uint8_t byte) {
    int8_t i;
    for (i = 7; i >= 0; i--) {
        if ((byte >> i) & 0x01) {
            subcarrier_on();
        } else {
            subcarrier_off();
        } /* end if */
        wait_subcarrier_ticks(BIT_SUBCARRIER_TICKS);
    } /* end for */
} /* end transmit_byte */

void transmit_packet(void) {
    uint8_t i;
    P1OUT |= BIT0;
    for (i = 0; i < PACKET_LENGTH; i++) {
        transmit_byte(packet[i]);
    } /* end for */
    subcarrier_off();
    P1OUT &= ~BIT0;
} /* end transmit_packet */

int main(void) {
    WDTCTL = WDTPW | WDTHOLD;

    if (CALBC1_1MHZ != 0xFF) {
        DCOCTL = 0;
        BCSCTL1 = CALBC1_1MHZ;
        DCOCTL = CALDCO_1MHZ;
    } /* end if */

    P1DIR  |= (BIT0 | BIT1 | BIT6);
    P1OUT  &= ~(BIT0 | BIT1 | BIT6);
    P1SEL  &= ~BIT1;                /* start as GPIO low (subcarrier off) */
    P1SEL2 &= ~BIT1;

    /* Timer_A0: SMCLK (1 MHz), up mode, toggle on CCR0.
     * CCR0 = (SMCLK / (2*SUBCARRIER_HZ_NOMINAL)) - 1.
     * With SMCLK=1 MHz and SUBCARRIER_HZ_NOMINAL=1 kHz: CCR0=499.
     * Start it ONCE and leave it running forever. */
    TA0CCR0  = (1000000U / (2U * SUBCARRIER_HZ_NOMINAL)) - 1U;
    TA0CCTL0 = OUTMOD_4;
    TA0CTL   = TASSEL_2 | MC_1 | TACLR;

    /* Clear any stale CCR0 event before timing bits from CCIFG edges. */
    TA0CCTL0 &= ~CCIFG;

    while (1) {
        transmit_packet();
        wait_subcarrier_ticks(GAP_BETWEEN_PACKETS_TICKS);
    } /* end while */

    return 0;
} /* end main */