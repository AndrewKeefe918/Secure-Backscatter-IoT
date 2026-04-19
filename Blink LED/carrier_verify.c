#include <msp430.h>

/**
 * HARDWARE SANITY CHECK
 * This code bypasses the reed switch and sends a constant 1kHz carrier.
 * Use this to verify your wiring and SDR settings.
 */
void main(void) {
    WDTCTL = WDTPW | WDTHOLD;     // Stop watchdog timer

    // 1. Setup P1.1 for Hardware Timer Output
    P1DIR |= BIT1;                // Set P1.1 as output
    P1SEL |= BIT1;                // Connect Timer_A to P1.1

    // 2. Setup Status LED
    P1DIR |= BIT0;                // Red LED
    P1OUT |= BIT0;                // Solid Red = "Test Signal Active"

    // 3. Configure Timer_A to toggle P1.1 at 1kHz
    // SMCLK is 1MHz. 500 cycles = 0.5ms. 
    // Toggling every 0.5ms creates a 1ms period (1kHz frequency).
    TA0CCR0 = 500;                
    TA0CCTL0 = OUTMOD_4;          // Toggle mode
    TA0CTL = TASSEL_2 + MC_1;     // Source: SMCLK, Mode: Up (Start Timer)

    // CPU sleeps, Timer hardware keeps flipping the switch forever
    __bis_SR_register(LPM0_bits); 
}