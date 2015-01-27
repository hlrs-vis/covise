/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef PCANLIGHT_H
#define PCANLIGHT_H

#ifdef HAVE_PCAN
#ifdef _WIN32
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x501 // This specifies WinXP or later - it is needed to access rawmouse from the user32.dll
#endif
#if (_MSC_VER >= 1300) && !(defined(MIDL_PASS) || defined(RC_INVOKED))
#define POINTER_64 __ptr64
#else
#define POINTER_64
#endif
#include <winsock2.h>
#include <io.h>
#ifndef PATH_MAX
#define PATH_MAX 512
#endif
#endif
#include <winsock2.h>
#include <windows.h>
#include <pcan_pci.h>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include "CanInterface.h"
#include "PCANLightInterface.h"

///Implementation of CanInterface interface for the Pcan PCI card (Peak Systems).

class PcanLight : public CanInterface
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
    PcanLight(HardwareType HWT, Baudrates BTR0BTR1);

    ///Disconnect Pcan PCI card
    ~PcanLight();

    ///Send Pcan can frame.
    bool sendFrame(TPCANMsg &msg);

    ///Read Pcan can frame.
    /**
		Reads Pcan can frame from the message buffer of the card.
		If there's no message in the buffer, blocks until either a new message arrives or an error occurs.
	**/
    bool readFrame(TPCANMsg &msg);

    ///Print Pcan can message to stdout.
    void printMsg(TPCANMsg &msg);

protected:
    HardwareType HWType;
};

#endif
#endif
