/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VNC_SOCKET_H
#define VNC_SOCKET_H

#ifdef _WIN32

#if (_MSC_VER > 1310)
/* && !(defined(MIDL_PASS) || defined(RC_INVOKED)) */
#define POINTER_64 __ptr64
#endif

#include <winsock2.h>

typedef unsigned char coUByte;
typedef int coInt32;
typedef unsigned int coUInt32;
#ifdef _MSC_VER
/* Microsoft VisualStudio */
typedef signed __int64 coInt64;
typedef unsigned __int64 coUInt64;
#else
/* MINGW (gcc) */
#if defined(__LP64__)
typedef signed long coInt64;
typedef unsigned long coUInt64;
#else
typedef signed long long coInt64;
typedef unsigned long long coUInt64;
#endif
#endif

#ifndef MINGW
typedef long ssize_t;
typedef unsigned short ushort;
#endif

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
#include <stdint.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#endif

int socketDatagram();
int socketStream();
int socketConnect(int sock, const sockaddr_in *sa);
int setReuseAddr(int sock);
int setTcpNoDelay(int sock);
uint16_t getSrcPort(int sock);
int handleBlocking(int sock, bool block);
int setBlocking(int sock);
int setNoBlocking(int sock);
int handleLoopback(int sock, uint8_t loop);
int setLoopback(int sock);
int setNoLoopback(int sock);
int setScope(int sock, uint8_t ttl);
int addMembership(int sock, const void *pmreq);
int dropMembership(int sock, const void *pmreq);
int createUcastSocket(uint32_t uni_addr, uint16_t port);
int createSendSocket(uint8_t ttl);
bool isMulticastAddress(uint32_t address);

#endif // VNC_SOCKET_H
