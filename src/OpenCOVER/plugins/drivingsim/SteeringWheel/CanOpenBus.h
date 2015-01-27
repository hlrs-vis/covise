/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CANOPENBUS_H
#define CANOPENBUS_H

#ifdef HAVE_PCAN
#include "CanInterface.h"
#include <iostream>

///Loose implementation of the CanOpen standard.
/**
	All functions reading from the can bus are blocking!
**/
class CanOpenBus
{
public:
    /// Constructor.
    /**
		\param can Pointer to a PC-CAN-Interface implementation, e.g. PcanPci (Peak Systems PCAN PCI Interface).
	**/
    CanOpenBus(CanInterface *can);

    /// Send NMT message to the bus.
    /**
		\param id Id of the slave node which the NMT message will be sent to.
		\param cs Command specifier:
			- cs=129: reset node
			- cs=1: start node
			- cs=2: stop node
	**/
    bool sendNMT(unsigned char id, unsigned char cs);

    /// Send RTR message.
    /**
		\param cob COB-ID of the requested can message.
		\param length Length of the requested can message.
	**/
    bool sendRTR(unsigned short cob, unsigned char length);

    /// Send a SYNC message to the bus.
    bool sendSYNC();

    /// Read a object from the dictionary of a node.
    /**
		\param id Id of the node the dictionary will be accessed from.
		\param index Index of the dictionary entry.
		\param subindex Subindex of the dictionary entry.
	**/
    TPCANMsg *readObject(unsigned char id, unsigned short index, unsigned char subindex);

    /// Write to an object in the dictionary of a node.
    /**
		\param id Id of the node the dictionary will be accessed from.
		\param index Index of the dictionary entry.
		\param subindex Subindex of the dictionary entry.
		\param length Length of the data array (in byte).
		\param data Array of data to be written into the dictionary entry.

		The length of the data array must not exceed four bytes.
	**/
    bool writeObject(unsigned char id, unsigned short index, unsigned char subindex, unsigned char length, unsigned char *data);

    /// Send PDO message to a node.
    /**
		\param id Id of the node the PDO will be sent to.
		\param pdonum Number of the Receiving PDO (RPDO) of the node the PDO will be sent to.
		\param length Length of the data array (in byte).
		\param data Data array of the PDO.

		The length of the data array must not exceed eight bytes.
		There's no check against unreasonable values.
	**/
    bool sendPDO(unsigned char id, unsigned char pdonum, unsigned char length, unsigned char *data);

    /// Receive a PDO message from the bus.
    /**
		\return	A pointer to the received PDO message.
					Returns NULL if an error occured.
	**/
    TPCANMsg *recvPDO();

    /// Receive a PDO message from the bus.
    /**
		\param id Id of the node the PDO message shall be received from.
		\param pdonum Number of the Transmit PDO (TPDO) of the node a message shall be received from.
		\param data Pointer to a data array the received data will be written to. Make sure, there's enough space allocated to receive the indicated PDO.

		\return	Returns false if another than the indicated PDO has been received or if an error occured.

		This function is blocking until a message arrives!
	**/
    bool recvPDO(unsigned char id, unsigned char pdonum, unsigned char *data);

    /// Receive a Emergency Object message from the bus.
    /**
		\param id Id of the node the Emergency Object shall be received from.
		\param data Pointer to a data array the received data will be written to. Make sure, there's enough space allocated to receive the indicated PDO.

		\return	Returns false if a message has been received, but not an Emergency Object from the indicated node or if an error occured.

		This function is blocking until a message arrives!
	**/
    bool recvEmergencyObject(unsigned char id, unsigned char *data);

protected:
    CanInterface *can;

    TPCANMsg msgNMT;
    TPCANMsg msgRTR;
    TPCANMsg msgSYNC;
    TPCANMsg msgSDO;
    TPCANMsg msgPDO;

    TPCANMsg recvMsg;

    void initMsg();

    TPCANMsg *sendSDO(unsigned char id, bool write, unsigned short index, unsigned char subindex, unsigned char length, unsigned char *data);
};
#endif

#endif
