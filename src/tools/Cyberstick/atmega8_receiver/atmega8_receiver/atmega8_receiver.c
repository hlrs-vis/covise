/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * atmega8_receiver.c
 *
 * Created: 26.06.2014 12:22:25
 *  Author: covise
 */

#define KEYBOARD
//#ifndef F_CPU
//#define F_CPU 12800000UL
#define F_CPU 12000000UL
//#endif
/*! \brief Pin number of IRQ contact on RFM73 module.*/
#define RFM73_IRQ_PIN DDD3
/*! \brief PORT register to IRQ contact on RFM73 module.*/
#define RFM73_IRQ_PORT PORTD
/*! \brief PIN register of IRQ contact on RFM73 module.*/
#define RFM73_IRQ_IN PIND
/*! \brief DDR register of IRQ contact on RFM73 module.*/
#define RFM73_IRQ_DIR DDRD
/*! \brief Pin number of CE contact on RFM73 module.*/
#define RFM73_CE_PIN DDB2
/*! \brief PORT register to CE contact on RFM73 module.*/
#define RFM73_CE_PORT PORTB
/*! \brief PIN register of CE contact on RFM73 module.*/
#define RFM73_CE_IN PINB
/*! \brief DDR register of CE contact on RFM73 module.*/
#define RFM73_CE_DIR DDRB
/*! \brief Pin number of CSN contact on RFM73 module.*/
#define RFM73_CSN_PIN DDB0
/*! \brief PORT register to CSN contact on RFM73 module.*/
#define RFM73_CSN_PORT PORTB
/*! \brief PIN register of CSN contact on RFM73 module.*/
#define RFM73_CSN_IN PINB
/*! \brief DDR register of CSN contact on RFM73 module.*/
#define RFM73_CSN_DIR DDRB

/*! \brief Setting high level on CE line.*/
#define RFM73_CE_HIGH RFM73_CE_PORT |= (1 << RFM73_CE_PIN)
/*! \brief Setting low level on CE line.*/
#define RFM73_CE_LOW RFM73_CE_PORT &= ~(1 << RFM73_CE_PIN)

#define MOD_CONTROL_LEFT (1 << 0)
#define MOD_SHIFT_LEFT (1 << 1)
#define MOD_ALT_LEFT (1 << 2)
#define MOD_GUI_LEFT (1 << 3)
#define MOD_CONTROL_RIGHT (1 << 4)
#define MOD_SHIFT_RIGHT (1 << 5)
#define MOD_ALT_RIGHT (1 << 6)
#define MOD_GUI_RIGHT (1 << 7)

#define KEY_1 30
#define KEY_2 31
#define KEY_3 32
#define KEY_4 33
#define KEY_5 34
#define KEY_6 35
#define KEY_7 36
#define KEY_8 37
#define KEY_9 38
#define KEY_0 39
#define KEY_RETURN 40

#include <avr/io.h>
#include <util/delay.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <avr/interrupt.h>
#include <avr/pgmspace.h>
#include <avr/eeprom.h>

#include "usbdrv.h"
#include "oddebug.h"
//#include "lcd.h"
#define sbi(port, bit) (port) |= (1 << (bit)) // set bit
#define cbi(port, bit) (port) &= ~(1 << (bit)) // clear bit
//#include "adc-atmega8.h"
#include "spi_init.h"
#define LED_RED 1 // red led is connected to pin 2 port d receiver only
#include "rfm70.h"

#define UTIL_BIN4(x) (uchar)((0##x & 01000) / 64 + (0##x & 0100) / 16 + (0##x & 010) / 4 + (0##x & 1))
#define UTIL_BIN8(hi, lo) (uchar)(UTIL_BIN4(hi) * 16 + UTIL_BIN4(lo))

#ifndef NULL
#define NULL ((void *)0)
#endif

/* ------------------------------------------------------------------------- */

static uchar reportBuffer[2]; /* buffer for HID reports */
static uchar idleRate; /* in 4 ms units */

static uchar adcPending;
static uchar isRecording;

static uchar valueBuffer[16];
static uchar *nextDigit;

/* ------------------------------------------------------------------------- */

const PROGMEM char usbHidReportDescriptor[USB_CFG_HID_REPORT_DESCRIPTOR_LENGTH] = { /* USB report descriptor */
                                                                                    0x05, 0x01, // USAGE_PAGE (Generic Desktop)
                                                                                    0x09, 0x06, // USAGE (Keyboard)
                                                                                    0xa1, 0x01, // COLLECTION (Application)
                                                                                    0x05, 0x07, //   USAGE_PAGE (Keyboard)
                                                                                    0x19, 0xe0, //   USAGE_MINIMUM (Keyboard LeftControl)
                                                                                    0x29, 0xe7, //   USAGE_MAXIMUM (Keyboard Right GUI)
                                                                                    0x15, 0x00, //   LOGICAL_MINIMUM (0)
                                                                                    0x25, 0x01, //   LOGICAL_MAXIMUM (1)
                                                                                    0x75, 0x01, //   REPORT_SIZE (1)
                                                                                    0x95, 0x08, //   REPORT_COUNT (8)
                                                                                    0x81, 0x02, //   INPUT (Data,Var,Abs)
                                                                                    0x95, 0x01, //   REPORT_COUNT (1)
                                                                                    0x75, 0x08, //   REPORT_SIZE (8)
                                                                                    0x25, 0x65, //   LOGICAL_MAXIMUM (101)
                                                                                    0x19, 0x00, //   USAGE_MINIMUM (Reserved (no event indicated))
                                                                                    0x29, 0x65, //   USAGE_MAXIMUM (Keyboard Application)
                                                                                    0x81, 0x00, //   INPUT (Data,Ary,Abs)
                                                                                    0xc0 // END_COLLECTION
};
bool LEDonoff = false;
bool LEDonoff_2 = false;
bool counting = false;

int counter = 6;

static void buildReport(void)
{
    uchar key = 0;

    if (nextDigit != NULL)
    {
        key = *nextDigit;
    }
    reportBuffer[0] = 0; /* no modifiers */
    reportBuffer[1] = key;
}

/* ------------------------------------------------------------------------- */
/* ------------------------ interface to USB driver ------------------------ */
/* ------------------------------------------------------------------------- */

uchar usbFunctionSetup(uchar data[8])
{
    usbRequest_t *rq = (void *)data;

    usbMsgPtr = reportBuffer;
    if ((rq->bmRequestType & USBRQ_TYPE_MASK) == USBRQ_TYPE_CLASS)
    { /* class request type */
        if (rq->bRequest == USBRQ_HID_GET_REPORT)
        { /* wValue: ReportType (highbyte), ReportID (lowbyte) */
            /* we only have one report type, so don't look at wValue */
            buildReport();
            return sizeof(reportBuffer);
        }
        else if (rq->bRequest == USBRQ_HID_GET_IDLE)
        {
            usbMsgPtr = &idleRate;
            return 1;
        }
        else if (rq->bRequest == USBRQ_HID_SET_IDLE)
        {
            idleRate = rq->wValue.bytes[1];
        }
    }
    else
    {
        /* no vendor specific requests implemented */
    }
    return 0;
}

ISR(TIMER1_COMPA_vect)
{
    if (LEDonoff == false)
    {
        LEDonoff_2 = !LEDonoff_2;

        if (counting == true)
        {
            counter -= 1;
        }
    }
    LEDonoff = !LEDonoff;
}
void Timer_io_init(void)
{

    TCCR1A |= (0 << WGM11) | (1 << WGM10);
    TCCR1B |= (1 << WGM13) | (0 << WGM12);

    TCCR1B |= (1 << CS12) | (0 << CS11) | (1 << CS10);

#define DELAY (F_CPU / 2048 / 2 / 2)
    OCR1A = DELAY;
    TCNT1 = -(unsigned char)DELAY;

    TIMSK1 = (1 << OCIE1A);
}

//----------------------------------------------------------------------------------
// Receive Payload
//----------------------------------------------------------------------------------

uint8_t rfm70ReceivePayload(uint8_t *payload)
{
    uint8_t len;
    uint8_t status;
    //uint8_t detect;
    uint8_t fifo_status;
    uint8_t rx_buf[32];

    status = rfm70ReadRegValue(RFM70_REG_STATUS);

    /*	
	detect = rfm70ReadRegValue(RFM70_REG_CD);       // Read value of Carrier Detection register
	if (fifo_status & RFM70_FIFO_STATUS_RX_FULL) {
		return false;
	}
	
	if ((detect & RFM70_CARRIER_DETECTION) == 0x01)			// Confirm that the CD bit is set high/low  
	{
		cbi(PORTC, LED_YELLOW);
		lcd_goto(1,0);
		lcd_writeText("CD found        ", 16);
	}
	else
	{
		sbi(PORTC, LED_YELLOW);
		lcd_goto(1,0);
		lcd_writeText("CD not found    ", 16);
	}
	*/

    //char charValue [6] = "      ";

    // check if receive data ready (RX_DR) interrupt
    if (status & RFM70_IRQ_STATUS_RX_DR)
    {

        do
        {
            // read length of playload packet
            len = rfm70ReadRegValue(RFM70_CMD_RX_PL_WID);

            if (len <= 32) // 32 = max packet length
            {
                // read data from FIFO Buffer
                rfm70ReadRegPgmBuf(RFM70_CMD_RD_RX_PLOAD, rx_buf, len);

                sbi(PORTD, LED_RED);

                _delay_ms(100);
                cbi(PORTD, LED_RED);
                nextDigit = &valueBuffer[sizeof(valueBuffer)];
                *--nextDigit = 0xff; /* terminate with 0xff */
                *--nextDigit = 0;
                *--nextDigit = KEY_RETURN;
                *--nextDigit = (rx_buf[0] & 0xf) + 4;
                /*if(rx_buf[0]&1)
				{
					*--nextDigit = KEY_1;
				}
				else
				{
					*--nextDigit = KEY_0;
				}
				if(rx_buf[0]&2)
				{
					*--nextDigit = KEY_1;
				}
				else
				{
					*--nextDigit = KEY_0;
				}
				if(rx_buf[0]&4)
				{
					*--nextDigit = KEY_1;
				}
				else
				{
					*--nextDigit = KEY_0;
				}
				if(rx_buf[0]&8)
				{
					*--nextDigit = KEY_1;
				}
				else
				{
					*--nextDigit = KEY_0;
				}*/

                //itoa(len, charValue, 16);
                //lcd_clear();
                //lcd_goto(1,0);
                //lcd_writeText(charValue, 2);

                //itoa(rx_buf[0], charValue, 16);
                //lcd_goto(1,3);
                //lcd_writeText(charValue, 2);

                //itoa(rx_buf[1], charValue, 16);
                //lcd_goto(1,6);
                //lcd_writeText(charValue, 2);

                //itoa(rx_buf[2], charValue, 16);
                //lcd_goto(1,9);
                //lcd_writeText(charValue, 2);
            }
            else
            {
                // flush RX FIFO
                rfm70WriteRegPgmBuf((uint8_t *)RFM70_CMD_FLUSH_RX, sizeof(RFM70_CMD_FLUSH_RX));
            }

            fifo_status = rfm70ReadRegValue(RFM70_REG_FIFO_STATUS);
        } while ((fifo_status & RFM70_FIFO_STATUS_RX_EMPTY) == 0);

        if ((rx_buf[0] == 0xAA) && (rx_buf[1] == 0x80))
        {
            //sbi(PORTD, LED_RED);
            _delay_ms(100);
            //cbi(PORTD, LED_RED);

            rfm70SetModeRX();
        }
    }
    rfm70WriteRegValue(RFM70_CMD_WRITE_REG | RFM70_REG_STATUS, status);

    return 0;
}

// ----------------------------------------------------------------------------
// main
// ----------------------------------------------------------------------------
int main(void)
{
    ///////////////////////////////////////////////////////////////////////////
    // low level init
    // IO-pins and pullups

    _delay_ms(100);

    //all unused ports as input with pullups

    DDRC &= ~((1 << DDC0) | (1 << DDC1) | (1 << DDC2) | (1 << DDC3) | (1 << DDC4) | (1 << DDC5));
    PORTC |= ((1 << DDC0) | (1 << DDC1) | (1 << DDC2) | (1 << DDC3) | (1 << DDC4) | (1 << DDC5));
    DDRB &= ~((1 << DDB1));
    PORTB |= ((1 << DDB1));
    DDRD &= ~((1 << DDD4) | (1 << DDD5) | (1 << DDD6) | (1 << DDD7));
    PORTD |= ((1 << DDD4) | (1 << DDD5) | (1 << DDD6) | (1 << DDD7));

    // leds
    // DDD2 - red led
    DDRD |= (1 << DDD1);
    PORTD &= ~((1 << DDD1));

    // do SPI init after seting CE to LOW, this is important, otherwise the RFM73 module does not always answer to SPI requests.

    //out
    RFM73_CSN_DIR |= (1 << RFM73_CSN_PIN);
    RFM73_CE_DIR |= (1 << RFM73_CE_PIN);
    //in
    RFM73_IRQ_DIR &= ~(1 << RFM73_IRQ_PIN);
    RFM73_CE_LOW;

    DDRD |= (1 << USB_CFG_DMINUS_BIT) | (1 << USB_CFG_DPLUS_BIT);

    uchar i;
    odDebugInit();
    usbDeviceDisconnect();
    for (i = 0; i < 20; i++)
    { /* 300 ms disconnect */
        _delay_ms(15);
    }
    usbDeviceConnect();

    usbInit();
    sei();
    while (1)
    {

        usbPoll();
        _delay_ms(10);
        if (usbInterruptIsReady() && nextDigit != NULL)
        { // we can send another key
            buildReport();
            usbSetInterrupt(reportBuffer, sizeof(reportBuffer));
            if (*++nextDigit == 0xff) // this was terminator character
                nextDigit = NULL;
        }
    }

    // low level init
    // IO-pins and pullups
    spiInit();
    spiSelect(csNONE);
    cbi(PORTD, LED_RED);
    _delay_ms(500);

    // ADC
    // TODO

    // TIMER INIT
    // TODO

    //////////////////////////////////////////////////////////////////////////
    // schrott

    unsigned char buf[32];
    // char charValue [6] = "      ";
    // int len = 6;

    //////////////////////////////////////////////////////////////////////////
    // start main loop

    // state machine
    // state 0  - exit
    // state 1  - choose transmitter or receiver
    // state 2  - init of transmitter module
    // state 3  - wait for button press and send something
    // state 10 - init and power up RX modules

    uint8_t smState = 1;

    //sei();

    sbi(PORTD, LED_RED);

    // RFM70
    // write registers
    // light green led if rfm70 found
    // light red led if rfm70 not found
    if (rfm70InitRegisters())
    {
        cbi(PORTD, LED_RED);
    }
    else
    {
        sbi(PORTD, LED_RED);
    }

    while (smState != 0)
    {
        while (1)
        {

            usbPoll();
            _delay_ms(10);
            if (usbInterruptIsReady() && nextDigit != NULL)
            { // we can send another key
                buildReport();
                usbSetInterrupt(reportBuffer, sizeof(reportBuffer));
                if (*++nextDigit == 0xff) // this was terminator character
                    nextDigit = NULL;
            }
        }
        //------------------------------------------------------------------------
        // Receiver State Machine
        //------------------------------------------------------------------------
        switch (smState)
        {
        case 1:
        {
            // start state
            // we are the receiver
            smState = 10;
            break;
        }
        case 10:
        {

            // init and power up modules
            // goto RX mode
            rfm70SetModeRX();

            smState = 11;
            break;
        }
        case 11:
        {
            // try to receive something

            // _delay_ms(100);
            rfm70ReceivePayload(buf);

            break;
        }
        }
    }
}