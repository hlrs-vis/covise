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

#define F_CPU 12000000UL // system clock in Hz -


#define BAUD 9600UL     // Baud rate

// Calculations UART serial communication BAUDRATE
#define UBRR_VAL ((F_CPU + BAUD * 8) / (16 * BAUD) -1) // clever round
#define BAUD_REAL (F_CPU / (16 * (UBRR_VAL + 1))) // Reale baud
#define BAUD_ERROR ((BAUD_REAL * 1000) / BAUD) // error in per mil, 1000 = no error.



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

#define sbi(port, bit) (port) |= (1 << (bit)) // set bit
#define cbi(port, bit) (port) &= ~(1 << (bit)) // clear bit

#define LED_RED 4 //1 // red led is connected to pin 6 port d receiver only

#include <avr/io.h>
#include <avr/interrupt.h>
#include <avr/wdt.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <util/delay.h>

#include "usbdrv.h"

#include "spi_init.h"
#include "rfm73.h"

#define USB_LED_OFF 0
#define USB_LED_ON 1

const PROGMEM char usbHidReportDescriptor[52] = { /* USB report descriptor, size must match usbconfig.h */
                                                  0x05, 0x01, // USAGE_PAGE (Generic Desktop)
                                                  0x09, 0x02, // USAGE (Mouse)
                                                  0xa1, 0x01, // COLLECTION (Application)
                                                  0x09, 0x01, //   USAGE (Pointer)
                                                  0xA1, 0x00, //   COLLECTION (Physical)
                                                  0x05, 0x09, //     USAGE_PAGE (Button)
                                                  0x19, 0x01, //     USAGE_MINIMUM
                                                  0x29, 0x03, //     USAGE_MAXIMUM
                                                  0x15, 0x00, //     LOGICAL_MINIMUM (0)
                                                  0x25, 0x01, //     LOGICAL_MAXIMUM (1)
                                                  0x95, 0x05, //     REPORT_COUNT (5)
                                                  0x75, 0x01, //     REPORT_SIZE (1)
                                                  0x81, 0x02, //     INPUT (Data,Var,Abs)
                                                  0x95, 0x01, //     REPORT_COUNT (1)
                                                  0x75, 0x05, //     REPORT_SIZE (5)
                                                  0x81, 0x03, //     INPUT (Const,Var,Abs)
                                                  0x05, 0x01, //     USAGE_PAGE (Generic Desktop)
                                                  0x09, 0x30, //     USAGE (X)
                                                  0x09, 0x31, //     USAGE (Y)
                                                  0x09, 0x38, //     USAGE (Wheel)
                                                  0x15, 0x81, //     LOGICAL_MINIMUM (-127)
                                                  0x25, 0x7F, //     LOGICAL_MAXIMUM (127)
                                                  0x75, 0x08, //     REPORT_SIZE (8)
                                                  0x95, 0x03, //     REPORT_COUNT (3)
                                                  0x81, 0x06, //     INPUT (Data,Var,Rel)
                                                  0xC0, //   END_COLLECTION
                                                  0xC0, // END COLLECTION
};

typedef struct
{
    uchar buttonMask;
    char dx;
    char dy;
    char dWheel;
} report_t;

static report_t reportBuffer;
static uchar 	idleRate; /* repeat rate for keyboards, never used for mice */

uint8_t oldDX = 0;
uint8_t oldDY = 0;
uint8_t CyberStick_allowed = 1;

double carrier_detect_time = 0.0;
double ack_timeout = 0.0;
double cyberstick_switch_time = 0.0;

unsigned char occupied_frequencies[84];

bool CyberStick1_detected = false;
bool CyberStick2_detected = false;
bool timer0_initiated = false;
bool next_message = true;


//----------------------------------------------------------------------------------
// UART INIT and TRANSMIT
//----------------------------------------------------------------------------------


// function to send data
void uart_transmit (unsigned char data)
{
    while (!( UCSR0A & (1<<UDRE0)));                // wait while register is free
    UDR0 = data;                                   // load data in the register
}

void uart_init (void)
  {
	UBRR0 = UBRR_VAL;
	UCSR0B |= (1 << TXEN0); // Frame Format: Asynchronous 8N1
	UCSR0C = (1 << UCSZ01) | (1 << UCSZ00);
  }





// ----------------------------------------------------------------------------
// Convert floating point value into two integers for serial transmission
// Function for the size of the integers
// ----------------------------------------------------------------------------

void double2Ints(double f, int p, int *i, int *d)
{
  // f = float, p=decimal precision, i=integer, d=decimal
  int   li;
  int   prec=1;

  for(int x=p;x>0;x--)
  {
    prec*=10;
  };  // same as power(10,p)

  li = (int) f;              // get integer part
  *d = (int) ((f-li)*prec);  // get decimal part
  *i = li;
}

int lenHelper(unsigned x) {
    if(x>=1000000000) return 10;
    if(x>=100000000) return 9;
    if(x>=10000000) return 8;
    if(x>=1000000) return 7;
    if(x>=100000) return 6;
    if(x>=10000) return 5;
    if(x>=1000) return 4;
    if(x>=100) return 3;
    if(x>=10) return 2;
    return 1;
}


//----------------------------------------------------------------------------------
// Send beacon message
//----------------------------------------------------------------------------------

bool rfm70SendBeaconMsg(int toAck)
{
	uint8_t beacon_payload[32]; // acknowledgment message to transmit
	uint8_t status;
	bool ack_received = false;


    rfm70SetModeTX();
    _delay_ms(1);

    beacon_payload[0]= 0xAA;  // 0xAA defined code for beacon message
    beacon_payload[1]= 1;

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

    // send TX cmd via SPI
    if (toAck == -1)
    {
        // cmd: write payload with ack of received message used in RX mode
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

    int len = 0;
    while(len < 32)
    {
    	spiSendMsg(beacon_payload[len]);
    	len++;
    }
    // disable CSN
    spiSelect(csNONE);
    _delay_ms(0);

    uint8_t value = rfm70ReadRegValue(RFM70_REG_STATUS);
    if ((value & 0x20) == 0x00)
    {

       _delay_ms(2);
    }
    rfm70SetModeRX();
    //_delay_ms(2);   	// critical delay, do not do anything until 2ms after
    					// changing to RX mode

    return true;
}

// ----------------------------------------------------------------------------
// 8-bit timer0 initialization and its ISR implementation
// ----------------------------------------------------------------------------

void timer0_init()
{
	TIMSK0 |= (1<<TOIE0);			// set timer overflow(=255) interrupt

	TCCR0B |= (1<<CS02) | (1<<CS00);	// Set prescale value Clk(12Mhz)/1024
						// 1 count = 0.0853 ms
						// 1 timer overflow = 255*0.0853ms =21.76ms
}


ISR(TIMER0_OVF_vect)
{
	wdt_reset();
	cyberstick_switch_time = cyberstick_switch_time + 21.76 ;
}


//----------------------------------------------------------------------------------
// Receive Payload
//----------------------------------------------------------------------------------

int rfm70ReceivePayload()
{
    uint8_t len;
    uint8_t status;
  
    uint8_t fifo_status;
    unsigned char rx_buf[32];
    bool ack_received = false;
    status = rfm70ReadRegValue(RFM70_REG_STATUS);

    // check if receive data ready (RX_DR) interrupt
    if (status & RFM70_IRQ_STATUS_RX_DR)
    {
        do
        {
            // read length of playload packet
            len = rfm70ReadRegValue(RFM70_CMD_RX_PL_WID);

            if (len >= 5 && len <= 32) // 32 = max packet length
            {
                // read data from FIFO Buffer
                rfm70ReadRegPgmBuf(RFM70_CMD_RD_RX_PLOAD, rx_buf, len);


               	ack_received = true;

		reportBuffer.buttonMask = rx_buf[0];
		reportBuffer.dx = rx_buf[1] - oldDX;
		reportBuffer.dy = rx_buf[2] - oldDY;
		oldDX = rx_buf[1];
		oldDY = rx_buf[2];

		if (rx_buf[3] == 1) 
		{
			sbi(PORTD, LED_RED);
			_delay_ms(10);
			cbi(PORTD, LED_RED);
		}


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
            rfm70SetModeRX();
        }
    }
    rfm70WriteRegValue(RFM70_CMD_WRITE_REG | RFM70_REG_STATUS, status);

    return ack_received;
}



int main()
{
    uchar i;

    wdt_enable(WDTO_1S); // enable 1s watchdog timer
    //unsigned char p[] = {red, green, blue};
    //all unused ports as input with pullups

    DDRC &= ~((1 << DDC0) | (1 << DDC1) | (1 << DDC2) | (1 << DDC3) | (1 << DDC4) | (1 << DDC5));
    PORTC |= ( (1 << DDC0) | (1 << DDC1) |(1 << DDC2) | (1 << DDC3) | (1 << DDC4) | (1 << DDC5));
    DDRB &= ~((1 << DDB1));
    PORTB |= ((1 << DDB1));
    DDRD &= ~((1 << DDD0)  | (1 << DDD5) | (1 << DDD6) );
    PORTD |= ((1 << DDD0)  | (1 << DDD5) | (1 << DDD6) );

    // leds
    // DDD1 changed to DDD4 - red led
    DDRD |= (1 << DDD4) | (1 << DDD1);
    PORTD &= ~((1 << DDD4)| (1 << DDD1));

    // uart initialize and check on terminal serial communication is working
    uart_init();


    usbInit();
    cli();
    usbDeviceDisconnect(); // enforce re-enumeration
    for (i = 0; i < 250; i++)
    { 
        wdt_reset(); // keep the watchdog happy
        _delay_ms(2);
    }
    usbDeviceConnect();
    sei(); // Enable interrupts after re-enumeration
    // do SPI init after seting CE to LOW, this is important, otherwise the RFM73 module does not always answer to SPI requests.

    //out
    RFM73_CSN_DIR |= (1 << RFM73_CSN_PIN);
    RFM73_CE_DIR |= (1 << RFM73_CE_PIN);
    //in
    RFM73_IRQ_DIR &= ~(1 << RFM73_IRQ_PIN);
    RFM73_CE_LOW;
    // low level init
    // IO-pins and pullups
    spiInit();
    spiSelect(csNONE);
    //sbi(PORTD, LED_RED);

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


    _delay_ms(50);
    // init and power up modules
    // goto RX mode
    wdt_reset(); // keep the watchdog happy
    rfm70SetModeRX();
    wdt_reset(); // keep the watchdog happy
    _delay_ms(50);

    int rand = 1234;
    timer0_init();
    while (1)
    {
    	//sbi(PORTD, LED_RED);
    	if (cyberstick_switch_time >= 20) 
    	{
    		rfm70SendBeaconMsg(0);
    		cyberstick_switch_time = 0.0;
    	}


    	rfm70ReceivePayload();

    	wdt_reset(); // keep the watchdog happy
        usbPoll();

        if (usbInterruptIsReady())
        { // if the interrupt is ready, feed data
            // pseudo-random sequence generator, thanks to Dan Frederiksen @AVRfreaks
            // http://en.wikipedia.org/wiki/Linear_congruential_generator
            rand = (rand * 109 + 89) % 251;

            // move to a random direction
            //reportBuffer.dx = (rand&0xf)-8;
            //reportBuffer.dy = ((rand&0xf0)>>4)-8;

            usbSetInterrupt((void *)&reportBuffer, sizeof(reportBuffer));

            reportBuffer.dx = 0;
            reportBuffer.dy = 0;
        }
    }

    return 0;
}


usbMsgLen_t usbFunctionSetup(uchar data[8])
{
    usbRequest_t *rq = (void *)data;

    // The following requests are never used. But since they are required by
    // the specification, we implement them in this example.
    if ((rq->bmRequestType & USBRQ_TYPE_MASK) == USBRQ_TYPE_CLASS)
    {
        if (rq->bRequest == USBRQ_HID_GET_REPORT)
        {
            // wValue: ReportType (highbyte), ReportID (lowbyte)
            usbMsgPtr = (void *)&reportBuffer; // we only have this one
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

    return 0; // by default don't return any data
}




