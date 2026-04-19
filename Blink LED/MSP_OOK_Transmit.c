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
 * Timing: 50 ms per bit
 */

#define BIT_DURATION_MS         50
#define PACKET_LENGTH           6
#define GAP_BETWEEN_PACKETS_MS  2000

const uint8_t packet[PACKET_LENGTH] = {
    0xAA,
    0x7E,
    'O',
    'P',
    'E',
    'N'
};

static void delay_ms(uint16_t ms) {
    while (ms--) {
        __delay_cycles(1000);
    }
} /* end delay_ms */

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
        delay_ms(BIT_DURATION_MS);
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
     * CCR0 = 499 -> toggle every 500 us -> 1 kHz square wave.
     * Start it ONCE and leave it running forever. */
    TA0CCR0  = 499;
    TA0CCTL0 = OUTMOD_4;
    TA0CTL   = TASSEL_2 | MC_1 | TACLR;

    while (1) {
        transmit_packet();
        delay_ms(GAP_BETWEEN_PACKETS_MS);
    } /* end while */

    return 0;
} /* end main */