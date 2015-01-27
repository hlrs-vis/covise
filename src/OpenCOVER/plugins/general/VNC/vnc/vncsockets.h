/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//---------------------------------------------------------------------------
// VREng (Virtual Reality Engine)	http://vreng.enst.fr/
//
// Copyright (C) 1997-2007 Ecole Nationale Superieure des Telecommunications
//
// VREng is a free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public Licence as published by
// the Free Software Foundation; either version 2, or (at your option)
// any later version.
//
// VREng is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
//---------------------------------------------------------------------------
#ifndef VNCSOCKETS_H
#define VNCSOCKETS_H

#ifdef _WIN32

#if (_MSC_VER > 1310)
/* && !(defined(MIDL_PASS) || defined(RC_INVOKED)) */
#define POINTER_64 __ptr64
#endif

#include <winsock2.h>

typedef unsigned char coUByte;
typedef int coInt32;
typedef unsigned int coUInt32;
typedef long long coInt64;
typedef unsigned long long coUInt64;
//typedef long         ssize_t;
typedef unsigned short ushort;

#if defined(__MINGW32__) || (_MSC_VER >= 1600)
#include <stdint.h>
#endif

/* define the int types defined in inttypes.h on *ix systems */
/* the ifndef is necessary to avoid conflict with OpenInventor inttypes !
   The Inventor includes should be made before coTypes.h is included!
 */
#if !defined(_INVENTOR_INTTYPES_) && !defined(__MINGW32__) && (_MSC_VER < 1600)
#define _INVENTOR_INTTYPES_
typedef char int8_t;
typedef unsigned char uint8_t;
typedef short int16_t;
typedef unsigned short uint16_t;
typedef int int32_t;
typedef unsigned int uint32_t;
#endif

#define _BOOL_

#else
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#endif

#define VNC_BUF_SIZE 8192

// insecure ???
#ifndef MAXHOSTNAMELEN
#define MAXHOSTNAMELEN 64
#endif

/**
 * VNCSocket class
 */
class VNCSockets
{

private:
    char ServerName[MAXHOSTNAMELEN];
    ///< name of the server

    uint16_t port;
    ///< port

    uint32_t ipaddr;
    ///< IP address of the server

    int rfbsock;
    ///< the used socket

    uint32_t buffered;
    uint8_t buf[VNC_BUF_SIZE];
    uint8_t *bufoutptr;
    ///< buffer

    bool StringToIPAddr();
    ///< stores the IP address of the server in ipaddr, returns false if unknown

public:
    VNCSockets(){};
    VNCSockets(const char *_servername, uint16_t _port);
    VNCSockets(uint32_t IPAddr, uint16_t _port);
///< constructors

#if 0 //unused
  void PrintInHex(char *buf, int len);
  ///< Print out the contents of a packet for debugging.
#endif

    signed int ConnectToTcpAddr();
    /**<
   * Connects to the given TCP port.
   * Returns the socket if connected, -1 if connection failed */

    bool SameMachine();
    ///< Test if the other end of a socket is on the same machine.

    bool SetNonBlocking();
    ///< sets a socket into non-blocking mode.

    bool ReadFromRFBServer(char *out, uint32_t n);
    ///< Read bytes from the sever and stores it in the buffer

    bool WriteExact(char *buf, int n);
    ///< Write an exact number of bytes, and don't return until you've sent them.

    int GetSock();
    ///< get the socket used
};

#endif // VNCSOCKETS_H
