#include <msp430.h>
#include <stdint.h>

/*
 * Continuous 1 kHz subcarrier test.
 *
 * Goal: output a steady 1 kHz square wave on P1.1 forever. No bits, no
 * packets, no gating — the subcarrier is just always on. This should
 * produce persistent, easy-to-see ±1 kHz sidebands on the receiver
 * around the 2.48 GHz carrier.
 *
 * If the receiver still can't see sidebands with this firmware running,
 * the problem is downstream of the MSP (wiring, ADG902 CTRL path,
 * ADG902 power, antenna topology).
 *
 * LEDs:
 *   P1.0 (red)   steady on  = firmware alive
 *   P1.6 (green) steady on  = subcarrier active (P1.1 driving CTRL)
 *
 * Clock setup mirrors the OOK firmware exactly so DCO/Timer_A0 config
 * is not a variable between the two tests.
 */

#define SUBCARRIER_HZ_NOMINAL  1000U

int main(void) {
    WDTCTL = WDTPW | WDTHOLD;

    /* Load 1 MHz DCO calibration. If CAL bytes are erased, blink P1.0
     * fast instead of transmitting at an unknown clock rate. */
    {
        volatile unsigned char *p_calbc1 = (volatile unsigned char *)0x10FFu;
        volatile unsigned char *p_caldco = (volatile unsigned char *)0x10FEu;
        unsigned char cal_bc1 = *p_calbc1;
        unsigned char cal_dco = *p_caldco;
        if (cal_bc1 == 0xFF || cal_dco == 0xFF) {
            P1DIR |= BIT0;
            while (1) { P1OUT ^= BIT0; __delay_cycles(50000); }
        }
        DCOCTL  = 0;
        BCSCTL1 = cal_bc1;
        DCOCTL  = cal_dco;
        BCSCTL2 = 0;  /* SMCLK = DCOCLK / 1, MCLK = DCOCLK / 1 */
    }

    /* Configure outputs. */
    P1DIR  |=  (BIT0 | BIT1 | BIT6);
    P1OUT  &= ~(BIT0 | BIT1 | BIT6);
    P1SEL2 &= ~BIT1;

    /* Timer_A0: SMCLK / up mode / toggle on CCR0 for 1 kHz square wave.
     * CCR0 = SMCLK / (2 * f_sub) - 1 = 1e6 / 2000 - 1 = 499. */
    TA0CCR0  = (1000000U / (2U * SUBCARRIER_HZ_NOMINAL)) - 1U;
    TA0CCTL0 = OUTMOD_4;
    TA0CTL   = TASSEL_2 | MC_1 | TACLR;

    /* Route Timer_A0 output onto P1.1 and leave it there forever.
     * No gating, no P1SEL toggling — subcarrier is always on. */
    P1SEL  |= BIT1;

    /* Status LEDs on. */
    P1OUT  |= (BIT0 | BIT6);

    /* Sit idle. Timer_A0 keeps toggling P1.1 in hardware. */
    while (1) {
        /* __bis_SR_register(LPM0_bits); could be used here but plain
         * spinning is fine and keeps SMCLK guaranteed-on regardless of
         * low-power mode handling on the particular chip/silicon. */
    }
}