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

static void subcarrier_on(void) {
    TA0CTL |= TACLR;
    P1OUT &= ~BIT1;
    P1SEL |= BIT1;
    TA0CTL = TASSEL_2 | MC_1;
    P1OUT |= BIT6;
} /* end subcarrier_on */

static void subcarrier_off(void) {
    TA0CTL = TASSEL_2 | MC_0;
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

    P1DIR |= (BIT0 | BIT1 | BIT6);
    P1OUT &= ~(BIT0 | BIT1 | BIT6);
    P1SEL2 &= ~BIT1;

    TA0CCR0 = 499;
    TA0CCTL0 = OUTMOD_4;
    TA0CTL = TASSEL_2 | MC_0;

    while (1) {
        transmit_packet();
        delay_ms(GAP_BETWEEN_PACKETS_MS);
    } /* end while */

    return 0;
} /* end main */