#include <msp430.h>
#include <stdint.h>

#define MESSAGE 0x4F70656E   // "Open"
#define BIT_COUNT 32
#define BIT_DURATION 100000  // 100 ms at 1 MHz MCLK

int main(void) {
    WDTCTL = WDTPW | WDTHOLD;

    // Calibrate DCO to exactly 1 MHz
    if (CALBC1_1MHZ != 0xFF) {
        DCOCTL = 0;
        BCSCTL1 = CALBC1_1MHZ;
        DCOCTL = CALDCO_1MHZ;
    }

    P1DIR |= (BIT0 | BIT1 | BIT6);
    P1OUT &= ~(BIT0 | BIT1 | BIT6);

    TA0CCR0 = 499;              // 500 cycles/toggle → 1 kHz square wave
    TA0CCTL0 = OUTMOD_4;
    TA0CTL = TASSEL_2 | MC_0;   // SMCLK, stopped

    while(1) {
        P1OUT |= BIT0;
        transmit_packet(MESSAGE);
        P1OUT &= ~BIT0;
        __delay_cycles(3000000);
    }
}

void transmit_packet(uint32_t packet) {
    int8_t i;
    for (i = BIT_COUNT - 1; i >= 0; i--) {
        if ((packet >> i) & 0x01) {
            TA0CTL |= TACLR;
            P1OUT &= ~BIT1;
            P1SEL |= BIT1;
            P1OUT |= BIT6;
            TA0CTL = TASSEL_2 | MC_1;
        } else {
            TA0CTL = TASSEL_2 | MC_0;
            P1SEL &= ~BIT1;
            P1OUT &= ~(BIT1 | BIT6);
        }
        __delay_cycles(BIT_DURATION);
    }
    TA0CTL = TASSEL_2 | MC_0;
    P1SEL &= ~BIT1;
    P1OUT &= ~(BIT1 | BIT6);
}