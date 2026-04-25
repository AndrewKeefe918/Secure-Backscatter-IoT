/*
 * backscatter_1khz.c  –  MSP430 EXP-G2ET backscatter tag
 *
 * Produces a continuous 1 kHz square wave on P1.1.
 * P1.1 is the data control (RF switch / load-modulation) pin.
 *
 * Method: Timer_A0 in UP mode with output-unit 0 in TOGGLE mode.
 *   DCO = 1 MHz  ->  SMCLK = 1 MHz
 *   CCR0 = 499   ->  toggles every 500 us  ->  1 kHz square wave
 *   Hardware does all toggling; CPU enters LPM0 and never wakes.
 *
 * Pin   Function
 * ----  --------
 * P1.1  TA0.0 output  (1 kHz square wave to RF switch / antenna load)
 * P1.0  Red LED       (heartbeat blink = running)
 */

#include <msp430.h>

static volatile unsigned char hb_div = 0;

/*
 * ADG901 control test mode on P1.1:
 *   0 = force LOW
 *   1 = force HIGH
 *   2 = 1 kHz square wave (normal operation)
 *
 * Change CTRL_TEST_MODE, rebuild, and flash for A/B/C RF checks.
 */
#define CTRL_MODE_LOW 0
#define CTRL_MODE_HIGH 1
#define CTRL_MODE_TOGGLE_1KHZ 2

/* Optional manual override example:
 * #define CTRL_TEST_MODE CTRL_MODE_LOW
 */
#ifndef CTRL_TEST_MODE
#define CTRL_TEST_MODE CTRL_MODE_TOGGLE_1KHZ
#endif

int main(void)
{
    /* ---- Watchdog off ---- */
    WDTCTL = WDTPW | WDTHOLD;

    /* ---- Clock: 1 MHz DCO using factory calibration ---- */
    if (CALBC1_1MHZ == 0xFF || CALDCO_1MHZ == 0xFF) {
        /* Calibration constants erased – blink the red LED as a fault
         * indicator and halt. Never proceeds to RF activity. */
        P1DIR |= BIT0;
        for (;;) {
            P1OUT ^= BIT0;
            __delay_cycles(100000UL);
        }
    }
    BCSCTL1 = CALBC1_1MHZ;
    DCOCTL  = CALDCO_1MHZ;   /* SMCLK = MCLK = 1 MHz */

    /* ---- P1.0: red LED output (heartbeat blink) ---- */
    P1DIR |= BIT0;
    P1OUT &= ~BIT0;

    /* ---- P1.1: ADG901 control output in selected test mode ---- */
    P1DIR  |= BIT1;
    P1SEL  &= ~BIT1;
    P1SEL2 &= ~BIT1;

#if CTRL_TEST_MODE == CTRL_MODE_LOW
    /* Mode A: constant LOW */
    P1OUT &= ~BIT1;
    TA0CCTL0 = 0;
    TA0CTL = MC_0;
#elif CTRL_TEST_MODE == CTRL_MODE_HIGH
    /* Mode B: constant HIGH */
    P1OUT |= BIT1;
    TA0CCTL0 = 0;
    TA0CTL = MC_0;
#elif CTRL_TEST_MODE == CTRL_MODE_TOGGLE_1KHZ
    /* Mode C: 1 kHz square wave via Timer_A0 (TA0.0 on P1.1)
     *  Period = 2 * (CCR0 + 1) / f_SMCLK
     *         = 2 * 500 / 1 000 000 = 1 ms -> 1 kHz
     */
    P1SEL  |= BIT1;           /* peripheral function: TA0.0 */
    P1SEL2 &= ~BIT1;
    TA0CCR0  = 499;
    TA0CCTL0 = OUTMOD_4;
    TA0CTL   = TASSEL_2 | MC_1 | TACLR;
#else
#error Invalid CTRL_TEST_MODE value. Use 0, 1, or 2.
#endif

    /* ---- WDT interval timer for heartbeat LED ----
     * WDT_MDLY_32 triggers about every 32 ms from SMCLK.
     * Toggle LED every 16 ticks => about 0.5 s per LED edge.
     */
    WDTCTL = WDT_MDLY_32;
    IE1 |= WDTIE;

    /* ---- Enter low-power mode; timer hardware runs autonomously ---- */
    __bis_SR_register(GIE);
    __bis_SR_register(LPM0_bits);

    /* Never reached */
    return 0;
}

#pragma vector=WDT_VECTOR
__interrupt void wdt_isr(void)
{
    hb_div++;
    if (hb_div >= 16) {
        hb_div = 0;
        P1OUT ^= BIT0;
    }
}
