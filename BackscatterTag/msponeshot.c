#include <msp430.h>

#define BIT_DURATION_US 50000UL
#define PACKET_GAP_US 200000UL
#define LED_MARKER_PULSES 3U
#define LED_MARKER_ON_US 20000UL
#define LED_MARKER_OFF_US 20000UL

/* LED telemetry modes on P1.0
 * 0: toggle once per packet (legacy heartbeat)
 * 1: mirror chip TX state (recommended for debug)
 */
#define LED_MODE_PACKET_TOGGLE 0
#define LED_MODE_CHIP_MIRROR 1

#ifndef LED_TELEMETRY_MODE
#define LED_TELEMETRY_MODE LED_MODE_CHIP_MIRROR
#endif

#ifndef ALL_ONES_TEST
#define ALL_ONES_TEST 1
#endif

#define TX_MODE_PACKET 0
#define TX_MODE_CONTINUOUS_ON 1
#define TX_MODE_CONTINUOUS_75_25 2

#ifndef TX_TEST_MODE
#define TX_TEST_MODE TX_MODE_PACKET
#endif

static const unsigned char PREAMBLE_BYTES[] = {0xAA, 0xAA};
static const unsigned char SYNC_BYTES[] = {0xD3, 0x91};
#if ALL_ONES_TEST
static const unsigned char PAYLOAD_BYTES[] = {0xFF, 0xFF, 0xFF, 0xFF};
#else
static const unsigned char PAYLOAD_BYTES[] = {'O', 'P', 'E', 'N'};
#endif

static void emit_packet_start_marker(void)
{
#if LED_TELEMETRY_MODE == LED_MODE_CHIP_MIRROR
    unsigned char i;
    for (i = 0; i < LED_MARKER_PULSES; i++) {
        P1OUT |= BIT0;
        __delay_cycles(LED_MARKER_ON_US);
        P1OUT &= ~BIT0;
        __delay_cycles(LED_MARKER_OFF_US);
    }
#endif
}

static void subcarrier_off(void)
{
    P1SEL &= ~BIT1;
    P1SEL2 &= ~BIT1;
    P1OUT &= ~BIT1;
#if LED_TELEMETRY_MODE == LED_MODE_CHIP_MIRROR
    P1OUT &= ~BIT0;
#endif
}

static void subcarrier_on(void)
{
    P1SEL |= BIT1;
    P1SEL2 &= ~BIT1;
#if LED_TELEMETRY_MODE == LED_MODE_CHIP_MIRROR
    P1OUT |= BIT0;
#endif
}

static void send_bit(unsigned char bit)
{
    if (bit) {
        subcarrier_on();
    } else {
        subcarrier_off();
    }
    __delay_cycles(BIT_DURATION_US);
}

static void send_byte(unsigned char value)
{
    unsigned char i;
    for (i = 0; i < 8; i++) {
        unsigned char bit = (value & 0x80U) ? 1U : 0U;
        send_bit(bit);
        value <<= 1;
    }
}

static void send_packet(void)
{
    unsigned int i;
    for (i = 0; i < (sizeof(PREAMBLE_BYTES) / sizeof(PREAMBLE_BYTES[0])); i++) {
        send_byte(PREAMBLE_BYTES[i]);
    }
    for (i = 0; i < (sizeof(SYNC_BYTES) / sizeof(SYNC_BYTES[0])); i++) {
        send_byte(SYNC_BYTES[i]);
    }
    for (i = 0; i < (sizeof(PAYLOAD_BYTES) / sizeof(PAYLOAD_BYTES[0])); i++) {
        send_byte(PAYLOAD_BYTES[i]);
    }
    subcarrier_off();
}

static void send_continuous_ratio_75_25(void)
{
    /* Repeat logical pattern 1110 forever -> 75% ones / 25% zeros. */
    static const unsigned char pattern[] = {1U, 1U, 1U, 0U};
    static unsigned char idx = 0U;
    send_bit(pattern[idx]);
    idx = (unsigned char)((idx + 1U) & 0x03U);
}

int main(void) {
    WDTCTL = WDTPW | WDTHOLD;

    if (CALBC1_1MHZ == 0xFF || CALDCO_1MHZ == 0xFF) {
        P1DIR |= BIT0;
        for (;;) {
            P1OUT ^= BIT0;
            __delay_cycles(100000UL);
        }
    }
    DCOCTL = 0;
    BCSCTL1 = CALBC1_1MHZ;
    DCOCTL = CALDCO_1MHZ;

    P1DIR |= (BIT0 | BIT1);
    P1OUT &= ~(BIT0 | BIT1);

    /* Timer_A0 toggle output at 1 kHz on TA0.0 (P1.1 when selected). */
    TA0CCR0 = 499;
    TA0CCTL0 = OUTMOD_4;
    TA0CTL = TASSEL_2 | MC_1 | TACLR;
    subcarrier_off();

    while (1) {
    #if TX_TEST_MODE == TX_MODE_CONTINUOUS_ON
        /* Continuous ON test: force perpetual 1 kHz switching, no framing/gap. */
        subcarrier_on();
    #if LED_TELEMETRY_MODE == LED_MODE_PACKET_TOGGLE
        P1OUT ^= BIT0;
    #endif
        __delay_cycles(50000UL);
    #elif TX_TEST_MODE == TX_MODE_CONTINUOUS_75_25
        /* Continuous ratio test: 1110 pattern, no framing/gap. */
        send_continuous_ratio_75_25();
    #if LED_TELEMETRY_MODE == LED_MODE_PACKET_TOGGLE
        P1OUT ^= BIT0;
    #endif
    #else
        emit_packet_start_marker();
        send_packet();
#if LED_TELEMETRY_MODE == LED_MODE_PACKET_TOGGLE
        P1OUT ^= BIT0;
#else
        P1OUT &= ~BIT0;
#endif
        __delay_cycles(PACKET_GAP_US);
    #endif
    }
}
