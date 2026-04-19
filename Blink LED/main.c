#include <msp430.h>

/**
 * Backscatter IoT Test Code
 * Red LED (P1.0) = Magnet Detected
 * Green LED (P1.6) & P1.1 = 1kHz Modulation active
 * Trigger = Reed Switch on P1.3
 */
void main(void) {
    WDTCTL = WDTPW | WDTHOLD;   // Stop watchdog timer [cite: 780]
    
    // Set P1.0 (Red LED), P1.6 (Green LED), and P1.1 (RF CTRL) as outputs [cite: 128]
    P1DIR |= (BIT0 | BIT1 | BIT6); 
    P1OUT &= ~(BIT0 | BIT1 | BIT6); // Ensure they start OFF

    // Set P1.3 (Reed Switch) as input with internal pull-up resistor [cite: 158, 754]
    P1DIR &= ~BIT3;             
    P1REN |= BIT3;              // Enable pull-up/down resistor
    P1OUT |= BIT3;              // Set as Pull-up

    while(1) {
        if(!(P1IN & BIT3)) {    // If Magnet is near (Reed Switch closed) [cite: 158]
            P1OUT |= BIT0;      // Turn on Red LED 
            P1OUT ^= BIT6;      // Toggle Green LED visually
            P1OUT ^= BIT1;      // Toggle the actual RF CTRL pin (P1.1)
            __delay_cycles(1000); // Pulse timing for approx 1kHz modulation
        } else {
            // Turn everything off when magnet is removed
            P1OUT &= ~(BIT0 | BIT6 | BIT1); 
        }
    }
}