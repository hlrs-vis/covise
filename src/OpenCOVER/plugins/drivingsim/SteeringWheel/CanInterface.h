/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CANINTERFACE_H
#define CANINTERFACE_H
#include <util/coTypes.h>
#ifdef HAVE_PCAN
#ifdef WIN32
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
#include <winsock2.h>
#include <windows.h>
#include <Pcan_pci.h>
#else
#include <pcan.h>
#endif

///Interface for PC-CAN-Devices.
/**Serves as an interface class for interface devices, whose purpose is to connect a can bus to a PC.
**/
class CanInterface
{

public:
    ///Destructor
    virtual ~CanInterface(){};

    ///Send can frame.
    virtual bool sendFrame(TPCANMsg &) = 0;

    ///Read can frame.
    virtual bool readFrame(TPCANMsg &) = 0;

    ///Print can message to stderr.
    virtual void printMsg(TPCANMsg &) = 0;

protected:
};

#endif
#endif
