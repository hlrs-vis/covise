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
	uint8_t retr_msg_count = 0xFF;
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

	while((frequency < 84) & (ack_received == false))
    {

		if (retr_msg_count == 0xFF)
		{
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

		}

		// disable CSN
		spiSelect(csNONE);
		_delay_ms(0);

		status = rfm70ReadRegValue(RFM70_REG_STATUS);
		// When Auto ack is on
		// check if data is sent and ack is received (TX_DS) interrupt
		if (status & RFM70_IRQ_STATUS_TX_DS)
		{
			ack_received = true;
			return ack_received;
		}


		retr_msg_count = rfm70ReadRegValue(0x08) | 0xF0;

		if ((retr_msg_count == 0xFF))
		{
			ack_received = false;
			frequency++;

			// clear flag TX_FULL
			rfm70WriteRegValue(RFM70_CMD_WRITE_REG | RFM70_REG_STATUS, rfm70ReadRegValue(RFM70_REG_STATUS)|0x10 );

			// change frequency
			rfm70WriteRegValue(RFM70_CMD_WRITE_REG | 0x05, frequency);

		}

    }

    return ack_received;
}

// ----------------------------------------------------------------------------
// rfm70ReceivePayload
// ----------------------------------------------------------------------------

uint8_t rfm70ReceivePayload()
{
    uint8_t len;
    uint8_t status;
    //uint8_t detect;
    uint8_t fifo_status;
    uint8_t rx_buf[32];
    int ack_received = 0;
    fifo_status = rfm70ReadRegValue(RFM70_REG_FIFO_STATUS);


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

				if (rx_buf[0] == 0xFF) 	// 0xFF user defined ack msg code
				{
					/* 1 count = 0.128 ms
				   //ack_time_current += TCNT0 * 0.128;//*(10^(-3));
				   ack_time_previous = ack_time_current;

				//	LED3ON;
				//	_delay_ms(10);
				//	LED3OFF;*/
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



    return true;
}


#endif /* RFMSENDRECEIVEPAYLOAD_H_ */
