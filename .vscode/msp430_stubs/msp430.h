/*
 * IntelliSense-only stub for MSP430G2553 register map.
 * NOT used by CCS at build time — CCS uses the real TI headers.
 * Only intended so ms-vscode.cpptools can parse MSP_OOK_Transmit.c
 * without flooding the Problems panel with "undefined symbol" noise.
 */
#ifndef MSP430_INTELLISENSE_STUB_H
#define MSP430_INTELLISENSE_STUB_H

#include <stdint.h>

/* ---- SFR types ------------------------------------------------------- */
typedef volatile uint8_t  sfrb_t;
typedef volatile uint16_t sfrw_t;

/* ---- Watchdog -------------------------------------------------------- */
extern sfrw_t WDTCTL;
#define WDTPW     (0x5A00)
#define WDTHOLD   (0x0080)

/* ---- DCO / BCS ------------------------------------------------------- */
extern sfrb_t DCOCTL;
extern sfrb_t BCSCTL1;
extern sfrb_t BCSCTL2;
extern sfrb_t BCSCTL3;
extern const sfrb_t CALBC1_1MHZ;
extern const sfrb_t CALDCO_1MHZ;

/* ---- Port 1 ---------------------------------------------------------- */
extern sfrb_t P1DIR;
extern sfrb_t P1OUT;
extern sfrb_t P1SEL;
extern sfrb_t P1SEL2;
extern sfrb_t P1IN;
extern sfrb_t P1REN;

/* ---- Timer A0 -------------------------------------------------------- */
extern sfrw_t TA0CTL;
extern sfrw_t TA0CCTL0;
extern sfrw_t TA0CCTL1;
extern sfrw_t TA0CCR0;
extern sfrw_t TA0CCR1;
extern sfrw_t TA0R;

/* TACTL / TACCTL bits */
#define TASSEL_2  (0x0200)  /* SMCLK */
#define ID_0      (0x0000)
#define MC_1      (0x0010)  /* up-mode */
#define TACLR     (0x0004)
#define OUTMOD_4  (0x0080)  /* toggle */
#define CCIE      (0x0010)
#define CCIFG     (0x0001)
#define OUT       (0x0004)

/* ---- Interrupts ------------------------------------------------------ */
#define __interrupt
#define __bis_SR_register(x)         ((void)0)
#define __bic_SR_register_on_exit(x) ((void)0)
#define _BIS_SR(x)                   ((void)0)
#define GIE      (0x0008)
#define LPM0_bits (0x0010)

/* ---- Misc ------------------------------------------------------------ */
#define __no_operation()  ((void)0)

#endif /* MSP430_INTELLISENSE_STUB_H */
