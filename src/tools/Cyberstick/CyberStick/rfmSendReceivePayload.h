/*
 * rfmSendReceivePayload.h
 *
 *  Created on: Nov 17, 2015
 *      Author: hpcdsaad
 */

#ifndef RFMSENDRECEIVEPAYLOAD_H_
#define RFMSENDRECEIVEPAYLOAD_H_

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


// ----------------------------------------------------------------------------
// rfm70SendPayload
// queries the FIFO status
// if its not full, write payload to FIFO
// ----------------------------------------------------------------------------
bool rfm70SendPayload2(uint8_t *payload, uint8_t len, uint8_t toAck)
{
    uint8_t status;
	bool ack_received = false;
	uint8_t retr_msg_count = 0xF5;
	int frequency = 0;

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

	while(true)//frequency < 84)
    {

		//if (frequency == 83)
		//{
			//frequency = 0;
		//}
		//if (retr_msg_count == 0xF5)
		//{
			retr_msg_count = 0x00;
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

			// send payload
			while (len--)
			{
				spiSendMsg(*(payload));
				payload++;
			}

		//}

		// disable CSN
		spiSelect(csNONE);
		_delay_ms(0);

		status = rfm70ReadRegValue(RFM70_REG_STATUS);
		// When Auto ack is on
		// check if data is sent and ack is received (TX_DS) interrupt
		if (status & RFM70_IRQ_STATUS_TX_DS)
		{
			rfm70SetModeRX();
			return true;
		}


		/*if (toAck == 2)
		{
			retr_msg_count = rfm70ReadRegValue(0x08) | 0xF0;

			if ((retr_msg_count == 0xF5))
			{
				//rfm70SetModeRX();
				//ack_received = false;
				frequency++;

				// clear flag TX_FULL
				rfm70WriteRegValue(RFM70_CMD_WRITE_REG | RFM70_REG_STATUS, rfm70ReadRegValue(RFM70_REG_STATUS)|0x10 );

				// change frequency
				rfm70WriteRegValue(RFM70_CMD_WRITE_REG | 0x05, frequency);

			}
		}*/
    }

	rfm70SetModeRX();
    return false;
}


#endif /* RFMSENDRECEIVEPAYLOAD_H_ */
