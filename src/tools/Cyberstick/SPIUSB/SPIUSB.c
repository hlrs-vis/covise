/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * GccApplication1.c
 *
 * Created: 05.08.2014 13:26:30
 *  Author: hpcwoess
 */

// done in Project#define F_CPU 12000000L
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
#define RFM73_CSN_PIN DDB1
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

#define sbi(port, bit) (port) |= (1 << (bit)) // set bit
#define cbi(port, bit) (port) &= ~(1 << (bit)) // clear bit

#define LED_RED 1 // red led is connected to pin 2 port d receiver only

#define USB_LED_ON 1
#define USB_LED_OFF 0
#define USB_DATA_OUT 2
#define USB_INIT 3

#include <avr/io.h>
#include <avr/interrupt.h>
#include <avr/wdt.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <util/delay.h>

#include "usbdrv.h"

#include "spi_init.h"

//static report_t reportBuffer;
static int8_t reportBuffer[8];
static int8_t lastReportBuffer[8];

static unsigned char replyBuf[128] = "Hello, USB!";
static int8_t numData = 0;

ISR(PCINT0_vect, ISR_BLOCK)
{
    if ((PINB & (1 << DDB0)) == 0)
    { // data is ready, read it
        //spiReadMaster(&val,2);
        //spiReadMaster(&val,3);
        //spiReadMaster(&val,4);
        //spiReadMaster(&val,5);
    }
}

int main()
{
    uchar i;

    //wdt_enable(WDTO_1S); // enable 1s watchdog timer

    //all unused ports as input with pullups

    DDRC &= ~((1 << DDC0) | (1 << DDC1) | (1 << DDC2) | (1 << DDC3) | (1 << DDC4) | (1 << DDC5));
    PORTC |= ((1 << DDC0) | (1 << DDC1) | (1 << DDC2) | (1 << DDC3) | (1 << DDC4) | (1 << DDC5));
    DDRB &= ~((1 << DDB1));
    PORTB |= ((1 << DDB1));
    DDRB &= ~((1 << DDB0));
    PORTB |= ((1 << DDB0));
    DDRD &= ~((1 << DDD4) | (1 << DDD5) | (1 << DDD6) | (1 << DDD7));
    PORTD |= ((1 << DDD4) | (1 << DDD5) | (1 << DDD6) | (1 << DDD7));

    cbi(PORTB, DDB1); // no pullup on chip select

    //PORTC &= ~((1<<DDC5));
    //PCICR |= (1<<PCIE0); // enable PCINT0..7
    //PCMSK0 |= (1<<PCINT0); // enable PCINT1
    // leds
    // DDD2 - red led
    DDRD |= (1 << DDD1);
    PORTD &= ~((1 << DDD1));

    PORTB |= ((1 << DDB1));
    DDRB |= ((1 << DDB1));

    sbi(PORTD, LED_RED);

    // setup TIMER
    TCCR1A |= (0 << WGM11) | (0 << WGM10);
    // prescaler = 8
    TCCR1B |= (0 << WGM13) | (0 << WGM12) | (0 << CS12) | (1 << CS11) | (0 << CS10);

    usbInit();
    cli();
    usbDeviceDisconnect(); // enforce re-enumeration
    for (i = 0; i < 250; i++)
    { // wait 500 ms
        //wdt_reset(); // keep the watchdog happy
        _delay_ms(2);
    }
    usbDeviceConnect();
    sei(); // Enable interrupts after re-enumeration
    // do SPI init after seting CE to LOW, this is important, otherwise the RFM73 module does not always answer to SPI requests.

    //out
    RFM73_CSN_DIR |= (1 << RFM73_CSN_PIN);
    RFM73_CE_DIR |= (1 << RFM73_CE_PIN);
    //in
    //RFM73_IRQ_DIR &=~(1 << RFM73_IRQ_PIN);
    RFM73_CE_HIGH;
    _delay_ms(100);
    // low level init
    // IO-pins and pullups
    spiInitMaster();
    cbi(PORTB, DDB2);
    _delay_ms(10);
    spiSelect(csNONE);
    sbi(PORTD, LED_RED);

    _delay_ms(20);

    uchar changed = 0;

    cbi(PORTD, LED_RED);
    uint16_t tmpint = 0x5ABA;
    spiSelect(csTOUCH);
    spiSendMsg(tmpint);

    spiSelect(csNONE);
    uint8_t oldDataReady = 0;

    while (1)
    {
        wdt_reset(); // keep the watchdog happy
        usbPoll();
        if ((PINB & (1 << DDB0)) != 0)
        {
            /*if(oldDataReady == 0)
			{
				
				spiSelect(csTOUCH);
				spiSendMsg(tmpint);
				
				spiSelect(csNONE);
			}*/
            oldDataReady = 1;
        }
        if (((PINB & (1 << DDB0)) == 0) && (oldDataReady != 0) && (numData < 100))
        { // data is ready, read it
            oldDataReady = 0;
            //_delay_us(10);
            spiSelect(csTOUCH);
            uint8_t res = spiReadMaster(&replyBuf[numData], 2);
            spiSelect(csNONE);

            wdt_reset(); // keep the watchdog happy
            //_delay_us(60);
            //numData+=res;
            spiSelect(csTOUCH);
            spiReadMaster(&replyBuf[numData], 3);
            spiSelect(csNONE);
            wdt_reset(); // keep the watchdog happy
            //_delay_us(60);
            numData += res;
            spiSelect(csTOUCH);
            spiReadMaster(&replyBuf[numData], 4);
            spiSelect(csNONE);
            wdt_reset(); // keep the watchdog happy
            //_delay_us(60);
            numData += res;
            spiSelect(csTOUCH);
            spiReadMaster(&replyBuf[numData], 2);
            spiSelect(csNONE);
            wdt_reset(); // keep the watchdog happy
            //numData+=res;
        }

        if (false) //while((PINB & (1<<DDB1))==0)
        {
            wdt_reset(); // keep the watchdog happy

            unsigned int oldTime = TCNT1;
            int res = spiReadMsg(&replyBuf[numData]);

            unsigned int currentTime = TCNT1;
            unsigned int diff = currentTime - oldTime;
            numData += res;
            if (res > 0)
            {

                if (res == 1)
                {

                    replyBuf[numData] = 0;
                    numData++;
                }
                replyBuf[numData] = res;
                numData++;
                replyBuf[numData] = diff & 0xff;
                numData++;
                replyBuf[numData] = (diff >> 8) & 0xff;
                numData++;
                replyBuf[numData] = (PINB & (1 << DDB1));
                numData++;
                if (numData > 100)
                    break;
            }
            else
            {
                break;
            }
        }
        if (tmpint == 0)
        {
            //numData=10;
        }
        tmpint++;
        for (i = 0; i < sizeof(reportBuffer); i++)
        {
            if (reportBuffer[i] != lastReportBuffer[i])
            {
                lastReportBuffer[i] = reportBuffer[i];
                changed = 1;
                sbi(PORTD, LED_RED);
            }
        }
        if (changed)
        {
            if (usbInterruptIsReady())
            {
                changed = 0;
                // called after every poll of the interrupt endpoint
                usbSetInterrupt((void *)&reportBuffer, sizeof(reportBuffer));
            }
        }
        else
        {
            //cbi(PORTD, LED_RED);
        }
    }

    return 0;
}

usbMsgLen_t usbFunctionSetup(uchar data[8])
{
    usbRequest_t *rq = (void *)data;

    switch (rq->bRequest)
    { // custom command is in the bRequest field
    case USB_LED_ON:
    {

        sbi(PORTD, LED_RED);
        //uint16_t tmpint = 0xBA5A;
        //	spiSelect(csTOUCH);
        //	spiSendMsg(tmpint);

        //	spiSelect(csNONE);
        return 0;
    }
    case USB_LED_OFF:
    {

        cbi(PORTD, LED_RED);
        //uint16_t tmpint = 0xBA5A;
        //	spiSelect(csTOUCH);
        //	spiSendMsg(tmpint);

        //	spiSelect(csNONE);
        return 0;
    }
    case USB_INIT:
    {

        uint16_t tmpint = 0x5ABA;
        spiSelect(csTOUCH);
        spiSendMsg(tmpint);

        spiSelect(csNONE);
        return 0;
    }
    case USB_DATA_OUT: // send data to PC
        usbMsgPtr = replyBuf;
        uint16_t oldNum = numData;
        numData = 0;
        return oldNum;
    }

    return 0; // by default don't return any data
}