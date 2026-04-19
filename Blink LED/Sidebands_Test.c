#include <msp430.h>

int main(void) {
    WDTCTL = WDTPW | WDTHOLD;

    // Calibrate DCO to 1 MHz
    if (CALBC1_1MHZ != 0xFF) {
        DCOCTL = 0;
        BCSCTL1 = CALBC1_1MHZ;
        DCOCTL = CALDCO_1MHZ;
    }

    P1DIR |= (BIT0 | BIT1 | BIT6);
    P1OUT &= ~(BIT0 | BIT1 | BIT6);
    P1SEL |= BIT1;
    P1SEL2 &= ~BIT1;

    TA0CCR0 = 499;
    TA0CCTL0 = OUTMOD_4;
    TA0CTL = TASSEL_2 | MC_1 | TACLR;

    P1OUT |= BIT0;  // Red LED on = running

    while(1) {
        __bis_SR_register(LPM0_bits);
    }
}