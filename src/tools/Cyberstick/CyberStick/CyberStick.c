/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * atmega8_cyberstick.c
 *
 * Created: 02.07.2014 10:51:21
 *  Author: covise
 */

#ifndef F_CPU
#define F_CPU 8000000UL
#endif

#define sbi(port, bit) (port) |= (1 << (bit)) // set bit
#define cbi(port, bit) (port) &= ~(1 << (bit)) // clear

#define CYBERSTICK_ID 1;
//buttons
#define BUTTON_1 DDB0 // Trigger
#define BUTTON_2 DDD7 // Right
#define BUTTON_3 DDD6 // Middle
#define BUTTON_4 DDD5 // Left
#define BUTTON_5 DDC1 // Left

#define LED_2 DDD0 // red LED Button 2 right
#define LED_3 DDD1 // green LED Button 3 middle
#define LED_4 DDD2 // yellow LED Button 4 left

#define BUTTON_BIT0 0 // position of button in button databyte
#define BUTTON_BIT1 1 //
#define BUTTON_BIT2 2 //
#define BUTTON_BIT3 3 //
#define BUTTON_BIT4 4 //

#define JOYSTICK_X DDC0
#define JOYSTICK_Y DDC1

#define LED2ON sbi(PORTD, LED_2)
#define LED2OFF cbi(PORTD, LED_2)
#define LED3ON sbi(PORTD, LED_3)
#define LED3OFF cbi(PORTD, LED_3)
#define LED4ON sbi(PORTD, LED_4)
#define LED4OFF cbi(PORTD, LED_4)

#define ADCTHRESHOLD 30
/*! \brief Pin number of IRQ contact on RFM73 module.*/
#define RFM73_IRQ_PIN DDD3
/*! \brief PORT register to IRQ contact on RFM73 module.*/
#define RFM73_IRQ_PORT PORTD
/*! \brief PIN register of IRQ contact on RFM73 module.*/
#define RFM73_IRQ_IN PIND
/*! \brief DDR register of IRQ contact on RFM73 module.*/
#define RFM73_IRQ_DIR DDRD
/*! \brief Pin number of CE contact on RFM73 module.*/
#define RFM73_CE_PIN DDB6
/*! \brief PORT register to CE contact on RFM73 module.*/
#define RFM73_CE_PORT PORTB
/*! \brief PIN register of CE contact on RFM73 module.*/
#define RFM73_CE_IN PINB
/*! \brief DDR register of CE contact on RFM73 module.*/
#define RFM73_CE_DIR DDRB
/*! \brief Pin number of CSN contact on RFM73 module.*/
#define RFM73_CSN_PIN DDB7
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

#include <avr/io.h>
#include <util/delay.h>
#include <stdbool.h>
#include <stdlib.h>
#include <avr/interrupt.h>
#include <avr/pgmspace.h>
#include <avr/wdt.h>
#include <compat/twi.h>
#include <inttypes.h>
#include <string.h>

#include "spi_init.h"
#include "rfm73.h"

bool LEDonoff = false;
bool LEDonoff_2 = false;
bool counting = false;

int counter = 6;

// ----------------------------------------------------------------------------
// rfm70SendPayload
// queries the FIFO status
// if its not full, write payload to FIFO
// ----------------------------------------------------------------------------
uint8_t rfm70SendPayload(uint8_t *payload, uint8_t len, uint8_t toAck)
{
    uint8_t status;

    rfm70SetModeTX();
    _delay_ms(1);

    // read status register
    status = rfm70ReadRegValue(RFM70_REG_FIFO_STATUS);

    // if the FIFO is full, do nothing just return false
    if (status & RFM70_FIFO_STATUS_TX_FULL)
    {
        return false;
    }

    // set CE low
    // rfm needs a CE high pulse of at least 10 us and stays in
    // TX mode as long CE==1 and FIFO is not empty. He returns
    // to standbyII if FIFO is empty.
    // rfm70SetCE(0);

    // enable CSN
    spiSelect(csRFM73);
    _delay_ms(0);

    // send TX cmd via SPI
    if (toAck == -1)
    {
        // cmd: write TX payload
        spiSendMsg(RFM70_CMD_W_ACK_PAYLOAD);
    }
    else if (toAck == 0)
    {
        // cmd: write TX payload and disable AUTOACK
        spiSendMsg(RFM70_CMD_W_TX_PAYLOAD_NOACK);
    }
    else
    {
        // cmd: write TX payload with defined ACK packet
        spiSendMsg(RFM70_CMD_WR_TX_PLOAD);
    }

    // send payload
    while (len--)
    {
        spiSendMsg(*(payload));
        payload++;
    }

    // disable CSN
    spiSelect(csNONE);
    _delay_ms(0);

    // now reset CE=1 to initiate transmitting the FIFO content
    //rfm70SetCE(1);

    return true;
}

// ----------------------------------------------------------------------------
// main
// ----------------------------------------------------------------------------
int main(void)
{
    _delay_ms(100);

    struct
    {
        uint8_t buttons; // state of all buttons
        // bit 2: button 3 pressed = 0, not pressed = 1;
        // bit 1: button 2 pressed = 0, not pressed = 1;
        // bit 0: button 1 pressed = 0, not pressed = 1;
        int8_t touchpadX; // x-coord of the touchpad
        int8_t touchpadY; // y-coord of the touchpad
        uint8_t wheelCounter; // wheel counter, if we happen to have a wheel some time
        uint8_t ID; // ID of the device

    } stateCurrent, stateLast;

    bool stateChanged;

    ///////////////////////////////////////////////////////////////////////////

    //in
    DDRB &= ~((1 << BUTTON_1));
    PORTB |= (1 << BUTTON_1); // pullup
    DDRD &= ~((1 << BUTTON_2) | (1 << BUTTON_3) | (1 << BUTTON_4));
    PORTD |= (1 << BUTTON_2) | (1 << BUTTON_3) | (1 << BUTTON_4); // pullup

    DDRC &= ~((1 << BUTTON_5));
    PORTC |= (1 << BUTTON_5); // pullup

    // all unused Ports need to be input with pullup, otherwise it does not start up propperly
    DDRB &= ~((1 << DDB1) | (1 << DDB2));
    PORTB |= ((1 << DDB1) | (1 << DDB2));
    DDRD &= ~((1 << DDD4));
    PORTD |= (1 << DDD4);
    DDRC &= ~((1 << DDC0) | (1 << DDC1) | (1 << DDC2) | (1 << DDC3) | (1 << DDC4) | (1 << DDC5));
    PORTC |= ((1 << DDC2) | (1 << DDC3) | (1 << DDC4) | (1 << DDC5));

    cbi(DDRC, DDC0); // DataReady is on C0
    sbi(PORTC, DDC0); // pullup
    cbi(DDRC, DDC1); // Touchpad button is on C1
    sbi(PORTC, DDC1); // pullup

    //out
    DDRD |= (1 << LED_2);
    DDRD |= (1 << LED_3);
    DDRD |= (1 << LED_4);

    LED2OFF;
    LED3OFF;
    LED4OFF;

    //out
    RFM73_CSN_DIR |= (1 << RFM73_CSN_PIN);

    spiSelect(csNONE);
    RFM73_CE_DIR |= (1 << RFM73_CE_PIN);
    //in
    RFM73_IRQ_DIR &= ~(1 << RFM73_IRQ_PIN);
    RFM73_CE_LOW;

    // low level init
    // IO-pins and pullups
    spiInit();
    spiSelect(csNONE);
    LED3ON;

    if (rfm70InitRegisters())
    {
        LED2OFF;
        LED3OFF;
    }
    else
    {

        LED2ON;
        LED3OFF;
    }

    sei();
    cbi(PORTB, DDB2); // chip enable of the touchpad to low
    _delay_ms(10);

    uint16_t initStreaming = 0x5ABA;
    spiSelect(csTOUCH);
    spiSendMsg16(initStreaming);

    spiSelect(csNONE);
    cbi(PORTB, DDB2); // chip enable of the touchpad to low

    //////////////////////////////////////////////////////////////////////////

    unsigned char buf[32];
    uint8_t value;

    stateCurrent.buttons = 0x00;
    stateCurrent.touchpadX = 0;
    stateCurrent.touchpadY = 0;

    stateLast = stateCurrent;
    stateChanged = false;

    //////////////////////////////////////////////////////////////////////////
    // start main loop

    uint8_t firsttime = 2;

    sei();

    while (true)
    {

        if ((PINC & (1 << DDC0)) == 0)
        {
            int8_t tmpdata[2];
            // data is ready, read it
            spiSelect(csTOUCH);
            spiReadMaster(&tmpdata, 2);
            spiSelect(csNONE);
            spiSelect(csTOUCH);
            spiReadMaster(&tmpdata, 3);
            spiSelect(csNONE);
            stateCurrent.touchpadY += tmpdata[1];
            spiSelect(csTOUCH);
            spiReadMaster(&tmpdata, 4);
            spiSelect(csNONE);
            stateCurrent.touchpadX += tmpdata[1];
            spiSelect(csTOUCH);
            spiReadMaster(&tmpdata, 2);
            spiSelect(csNONE);
        }
        else
        {
        }
        // get current state of all inputs
        // buttons

        if ((PINB & (1 << BUTTON_1)) == 0)
        {
            stateCurrent.buttons |= (1 << BUTTON_BIT0);
        }
        else
        {
            stateCurrent.buttons &= ~(1 << BUTTON_BIT0);
        }

        if ((PIND & (1 << BUTTON_2)) == 0)
        {
            stateCurrent.buttons |= (1 << BUTTON_BIT1);
        }
        else
        {
            stateCurrent.buttons &= ~(1 << BUTTON_BIT1);
        }

        if ((PIND & (1 << BUTTON_3)) == 0)
        {
            stateCurrent.buttons |= (1 << BUTTON_BIT2);
        }
        else
        {
            stateCurrent.buttons &= ~(1 << BUTTON_BIT2);
        }

        if ((PIND & (1 << BUTTON_4)) == 0)
        {
            stateCurrent.buttons |= (1 << BUTTON_BIT3);
        }
        else
        {
            stateCurrent.buttons &= ~(1 << BUTTON_BIT3);
        }

        if ((PINC & (1 << BUTTON_5)) == 0)
        {
            stateCurrent.buttons |= (1 << BUTTON_BIT4);
        }
        else
        {
            stateCurrent.buttons &= ~(1 << BUTTON_BIT4);
        }

        //check if something changed

        if ((stateCurrent.buttons != stateLast.buttons) || (stateCurrent.touchpadX != stateLast.touchpadX) || (stateCurrent.touchpadY != stateLast.touchpadY))
        {
            stateChanged = true;
            stateLast = stateCurrent;
        }

        if (stateChanged)
        {
            /*if(firsttime>0)
			{
				//firsttime --;
				uint16_t initStreaming = 0x5ABA;
				spiSelect(csTOUCH);
				spiSendMsg16(initStreaming);
				
				spiSelect(csNONE);
			}*/
            stateChanged = false;
            // copy current state to buffer

            buf[0] = stateCurrent.buttons;
            buf[1] = stateCurrent.touchpadX;
            buf[2] = stateCurrent.touchpadY;
            buf[3] = CYBERSTICK_ID;

            LED4ON;
            // send buffer

            if (rfm70SendPayload(buf, 32, 0) == true)
            {
                _delay_ms(10);
            }

            value = rfm70ReadRegValue(RFM70_REG_STATUS);
            if ((value & 0x20) == 0x00)
            {
                _delay_ms(50);
            }
        }
        else
        {
            LED4OFF;
        }
    }
}
