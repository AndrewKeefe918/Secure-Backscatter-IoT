#include <msp430.h>

#define BIT_DURATION_US 50000UL
#define PACKET_GAP_US 200000UL
#define REPETITION_CHIPS 3U

static const unsigned char PREAMBLE_BYTES[] = {0xAA, 0xAA};
static const unsigned char SYNC_BYTES[] = {0xD3, 0x91};
static const unsigned char PAYLOAD_BYTES[] = {'O', 'P', 'E', 'N'};

static void subcarrier_off(void)
{
    P1SEL &= ~BIT1;
    P1SEL2 &= ~BIT1;
    P1OUT &= ~BIT1;
}

static void subcarrier_on(void)
{
    P1SEL |= BIT1;
    P1SEL2 &= ~BIT1;
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

static void send_coded_bit(unsigned char bit)
{
    unsigned char chip;
    for (chip = 0; chip < REPETITION_CHIPS; chip++) {
        send_bit(bit);
    }
}

static void send_byte(unsigned char value)
{
    unsigned char i;
    for (i = 0; i < 8; i++) {
        unsigned char bit = (value & 0x80U) ? 1U : 0U;
        send_coded_bit(bit);
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
        send_packet();
        P1OUT ^= BIT0;
        __delay_cycles(PACKET_GAP_US);
    }
}
