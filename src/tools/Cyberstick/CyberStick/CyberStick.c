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
#define F_CPU 1000000UL
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
#include <avr/sleep.h>
#include <avr/pgmspace.h>
#include <avr/wdt.h>
#include <compat/twi.h>
#include <inttypes.h>
#include <string.h>



#include "spi_init.h"
#include "rfm73.h"
//#include "rfmSendReceivePayload.h"

bool stateChanged;

unsigned char buf[32];
//buf[3] = CYBERSTICK_ID;
// ----------------------------------------------------------------------------
// Variable declaration for timiongs
// ----------------------------------------------------------------------------
double beacon_msg_time = 0.0;
double msg_receive_timeout = 0.0;

// ----------------------------------------------------------------------------
// Variable declaration for wake up ISR
// ----------------------------------------------------------------------------

double time_out_cyberstick = 0.0;
bool power_down = false;


// ----------------------------------------------------------------------------
// Struct for saving states of buttons on the CyberStick
// ----------------------------------------------------------------------------

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


// ----------------------------------------------------------------------------
// 8-bit timer0 initialization and its ISR implementation
// ----------------------------------------------------------------------------

void timer0_init()
{
	TIMSK0 |= (1<<TOIE0);			// set timer overflow(=255) interrupt

	TCCR0A |= (1<<CS02) | (1<<CS00);	// Set prescale value Clk(8Mhz)/1024
						// 1 count = 0.128 ms
						// 1 timer overflow = 255*0.128ms =32.64ms
}


ISR(TIMER0_OVF_vect)
{
	msg_receive_timeout += 32.64;
	beacon_msg_time += 32.64;
	time_out_cyberstick += 32.64;

}


// ----------------------------------------------------------------------------
// ISR for button pressed on CyberStick and time out if no button pressed
// for 5 sec
// ----------------------------------------------------------------------------

void enable_PCINT()
{
	// Enable interrupt on pin[23..16], pins[15..8], pin[1..0]
	PCICR = (1<<PCIE2) | (1<<PCIE0) | (1<<PCIE1);
	//sei();

	// enable interrupt for buttons right, middle, left, touchpad
	PCMSK2 = (1<<PCINT23) | (1<<PCINT22) | (1<<PCINT21);
	PCMSK0 = (1<<PCINT0);
	PCMSK1 = (1<<PCINT9);

}

void disable_PCINT()
{
	// Disable interrupt on pin[23..16], pins[15..8], pin[1..0]
	PCICR = (0<<PCIE2) | (0<<PCIE0) | (0<<PCIE1);

	// disable interrupt on for buttons right, middle, left
	PCMSK2 = (0<<PCINT23) | (0<<PCINT22) | (0<<PCINT21);
	PCMSK0 = (0<<PCINT0);
	PCMSK1 = (0<<PCINT9);
}

ISR(PCINT2_vect)
{
	time_out_cyberstick = 0.0;
	disable_PCINT();

}

// ----------------------------------------------------------------------------
// power down settings for RFM73 and Attiny88
// ----------------------------------------------------------------------------

void power_down_mode()
{
	// Power down for rfm73, PWR_UP = 0
    rfm70WriteRegValue(RFM70_CMD_WRITE_REG | RFM70_REG_CONFIG, (0<<1));

    // Power down mode for attiny88
    cli();
    SMCR |= (1<<SE);
    SMCR |= (1<<SM1) | (0<<SM0);
    sei();
    sleep_cpu();
}

// ----------------------------------------------------------------------------
// rfm70SendPayload
// queries the FIFO status
// if its not full, write payload to FIFO
// ----------------------------------------------------------------------------
uint8_t rfm70SendPayload(uint8_t *payload, uint8_t len, uint8_t toAck, int pipe)
{
    uint8_t status;

    // read status register
    status = rfm70ReadRegValue(RFM70_REG_FIFO_STATUS);

    // if the FIFO is full, do nothing just return false
    if (status & RFM70_FIFO_STATUS_TX_FULL)
    {
    	return false;
    }

    // enable CSN
    spiSelect(csRFM73);
    _delay_ms(0);

    if (pipe==0)
    {
    	spiSendMsg(RFM70_CMD_W_ACK_PAYLOAD_P0);
    }

    if (pipe==1)
    {
    	spiSendMsg(RFM70_CMD_W_ACK_PAYLOAD_P1);
    }
    	// send TX cmd via SPI
    if (toAck == -1)
    {
        // cmd: write payload with ack of received message used in RX mode
    	//if (pipe == 0)
    	//	spiSendMsg(RFM70_CMD_W_ACK_PAYLOAD_P0);
    	//else
    	//	spiSendMsg(RFM70_CMD_W_ACK_PAYLOAD_P1);
    }
    else if (toAck == 0)
    {
        // cmd: write TX payload and disable AUTOACK
     //   spiSendMsg(RFM70_CMD_W_TX_PAYLOAD_NOACK);
    }
    else
    {
        // cmd: write TX payload with defined ACK packet
      //  spiSendMsg(RFM70_CMD_WR_TX_PLOAD);
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

    return true;
}

// ----------------------------------------------------------------------------
// rfm70ReceivePayload
// ----------------------------------------------------------------------------

uint8_t rfm70ReceivePayload()
{
    uint8_t len;
    uint8_t status;

    uint8_t fifo_status;
    uint8_t rx_buf[32];

    bool msg_received = false;

    fifo_status = rfm70ReadRegValue(RFM70_REG_FIFO_STATUS);

    status = rfm70ReadRegValue(RFM70_REG_STATUS);

    // check if receive data ready (RX_DR) interrupt
    if (status & RFM70_IRQ_STATUS_RX_DR)
    {
    	msg_received = true;
    	do
        {
            // read length of playload packet
            len = rfm70ReadRegValue(RFM70_CMD_RX_PL_WID);

            if (len >= 5 && len <= 32) // 32 = max packet length
            {
                // read data from FIFO Buffer
                rfm70ReadRegPgmBuf(RFM70_CMD_RD_RX_PLOAD, rx_buf, len);

		// Send message with ack payload of the beacon message
		// if receiver allows and button is pressed
		if (stateChanged)
		{
			stateChanged = false;
			rfm70SendPayload(buf, 32, -1, 1);
		}

		}
            else
            {
                // flush RX FIFO
                rfm70WriteRegPgmBuf((uint8_t *)RFM70_CMD_FLUSH_RX, sizeof(RFM70_CMD_FLUSH_RX));
            }

            fifo_status = rfm70ReadRegValue(RFM70_REG_FIFO_STATUS);
        } while ((fifo_status & RFM70_FIFO_STATUS_RX_EMPTY) == 0);


    }
    rfm70WriteRegValue(RFM70_CMD_WRITE_REG | RFM70_REG_STATUS, status);

    return msg_received;
}


// ----------------------------------------------------------------------------
// Search current receiver operating frequency
// ----------------------------------------------------------------------------

bool find_receiver_frequency()
{
	bool msg_received = false;
	int frequency = 0;
	TCNT0 = 0x00;

	// start communication with receiver at pipe0
	rfm70SendPayload(buf, 32, -1, 0);
	while(true)
	{
		if (frequency == 83)  // max possible frequencies 2400Mhz-2483Mhz
		{
			frequency = 0;
		}

		if (rfm70ReceivePayload())
		{
			msg_received = true;
			break;
		}

		msg_receive_timeout = TCNT0 * 0.128;

		if ((msg_receive_timeout >= 5) & (msg_received == false))
		{
			msg_receive_timeout = 0.0;
			TCNT0 = 0x00;
			frequency++;

			// change frequency
			rfm70WriteRegValue(RFM70_CMD_WRITE_REG | 0x05, frequency);
			_delay_ms(1);
		}


	}

	return msg_received;
}


// ----------------------------------------------------------------------------
// Main Function for CyberStick
// ----------------------------------------------------------------------------

void CyberStick_Start()
{
    //bool stateChanged;
	uint8_t value;

	buf[3] = CYBERSTICK_ID;

    stateCurrent.buttons = 0x00;
    stateCurrent.touchpadX = 0;
    stateCurrent.touchpadY = 0;

    stateLast = stateCurrent;
    stateChanged = false;

    sei();

    bool beacon_msg_received = false;

    // chnage frequency to check find_receiver_frequency() can find the right frequency
	rfm70WriteRegValue(RFM70_CMD_WRITE_REG | 0x05, 0x40);
	_delay_ms(2);

	rfm70SetModeRX();
	_delay_ms(2);

	timer0_init();
	LED3ON;
	find_receiver_frequency();

    while (true)
    {
    	LED3OFF;
    	LED4ON;   // always means connection is established with the receiver
    	//rfm70ReceivePayload();
    	// check if beacon message is received
    	// if not for 100ms that means receiver has switched to another
    	// frequency and we need to find receiver frequency again
    	beacon_msg_received = rfm70ReceivePayload();
    	if ( beacon_msg_received == false)
    	{
    		if (beacon_msg_time>= 100)
    		{
				LED4OFF;
				find_receiver_frequency();
				beacon_msg_time = 0.0;
				LED4ON;
    		}
    	}
    	else
    	{
    		beacon_msg_time =0.0;
    	}

    	// if button is not pressed for 20s go to power down mode
    	/*if (time_out_cyberstick > 20000)  // time is in ms
    	{
    		LED4OFF;
    		power_down = true;
    		time_out_cyberstick = 0.0;
    		enable_PCINT();
    		power_down_mode();

    	}*/

       /* if ((PINC & (1 << DDC0)) == 0)
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
        }*/
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

        /*if ((PINC & (1 << BUTTON_5)) == 0)
        {
            stateCurrent.buttons |= (1 << BUTTON_BIT4);
        }
        else
        {
            stateCurrent.buttons &= ~(1 << BUTTON_BIT4);
        }*/

        //check if something changed

        if ((stateCurrent.buttons != stateLast.buttons) || (stateCurrent.touchpadX != stateLast.touchpadX) || (stateCurrent.touchpadY != stateLast.touchpadY))
        {
            stateChanged = true;
            stateLast = stateCurrent;
        }

        if (stateChanged)
        {
        	time_out_cyberstick = 0.0;       //restart time, button is pressed

            // copy current state to buffer
            buf[0] = stateCurrent.buttons;
            buf[1] = stateCurrent.touchpadX;
            buf[2] = stateCurrent.touchpadY;
            buf[3] = CYBERSTICK_ID;
        }


    }
}



// ----------------------------------------------------------------------------
// main
// ----------------------------------------------------------------------------
int main(void)
{
    _delay_ms(100);

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

    cbi(DDRC, DDC0);  // DataReady is on C0
    sbi(PORTC, DDC0); // pullup
    cbi(DDRC, DDC1);  // Touchpad button is on C1
    sbi(PORTC, DDC1); // pullup

    //power reduction by stopping the clock to different peripherals
    //PRR =  (1<<PRTWI) |(1<<PRTIM1)| (1<<PRADC);

    //out
    DDRD |= (1 << LED_2);
    DDRD |= (1 << LED_3);
    DDRD |= (1 << LED_4);

    LED2OFF;
    LED3OFF;
    LED4OFF;

    //enable_PCINT();   // enable pin change interrupt

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
    //LED3ON;
    //LED2ON;

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

    _delay_ms(50);

    sei();
    cbi(PORTB, DDB2); // chip enable of the touchpad to low
    _delay_ms(10);


    spiSelect(csNONE);
    cbi(PORTB, DDB2); // chip enable of the touchpad to low

    //////////////////////////////////////////////////////////////////////////

    CyberStick_Start();
}



