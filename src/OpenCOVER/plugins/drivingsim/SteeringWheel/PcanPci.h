/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef PCANPCI_H
#define PCANPCI_H

#include "CanInterface.h"
#include <libpcan.h>
#include <iostream>
#include <stdio.h>
#include <string.h>

///Implementation of CanInterface interface for the Pcan PCI card (Peak Systems).

class PcanPci : public CanInterface
{

public:
    ///Connect to Pcan PCI card and set can bus speed.
    /**
		\param port is the number of the Pcan card port (1..8).
		\param speed is the communication speed of the can bus. Following options are available:
			- CAN_BAUD_1M
			- CAN_BAUD_500K
			- CAN_BAUD_250K
			- CAN_BAUD_125K
			- CAN_BAUD_100K
			- CAN_BAUD_50K
			- CAN_BAUD_20K
			- CAN_BAUD_10K
			- CAN_BAUD_5K
	**/
    PcanPci(unsigned char port, unsigned short speed);

    ///Disconnect Pcan PCI card
    ~PcanPci();

    ///Send Pcan can frame.
    bool sendFrame(TPCANMsg &msg);

    ///Read Pcan can frame.
    /**
		Reads Pcan can frame from the message buffer of the card.
		If there's no message in the buffer, blocks until either a new message arrives or an error occurs.
	**/
    bool readFrame(TPCANMsg &msg);

    ///Empty read queue of PCAN PCI card.
    void emptyReadQueue();

    ///Print Pcan can message to stdout.
    void printMsg(TPCANMsg &msg);

    ///Get status of the PCAN PCI card.
    /**
		\return Status code:
               - CAN_ERR_OK:             no error
               - CAN_ERR_XMTFULL:        transmit buffer full
               - CAN_ERR_OVERRUN:        overrun in receive buffer
               - CAN_ERR_BUSLIGHT:       bus error, errorcounter limit reached
               - CAN_ERR_BUSHEAVY:       bus error, errorcounter limit reached
               - CAN_ERR_BUSOFF:         bus error, 'bus off' state entered
               - CAN_ERR_QRCVEMPTY:      receive queue is empty
               - CAN_ERR_QOVERRUN:       receive queue overrun
               - CAN_ERR_QXMTFULL:       transmit queue full 
               - CAN_ERR_REGTEST:        test of controller registers failed
               - CAN_ERR_NOVXD:          Win95/98/ME only
               - CAN_ERR_RESOURCE:       can't create resource
               - CAN_ERR_ILLPARAMTYPE:   illegal parameter
               - CAN_ERR_ILLPARAMVAL:    value out of range
               - CAN_ERRMASK_ILLHANDLE:  wrong handle, handle error
	**/
    int getStatus();

    ///Get last stored error of the PCAN library.
    int getError();

    ///Set message filter of the PCAN PCI card for standard frames.
    /**
		\param fromID Any ID below will be filtered.
		\param toID Any ID above will be filtered.
	**/
    bool setMsgFilter(int fromID, int toID);

    ///Reset message filter.
    bool resetMsgFilter();

protected:
    HANDLE pcanHandle;
};

#endif
