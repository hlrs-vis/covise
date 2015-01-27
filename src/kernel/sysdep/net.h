/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef SYSDEP_NET_H
#define SYSDEP_NET_H

#include <sys/types.h>

#ifdef _WIN32

#include <winsock2.h>
#include <ws2tcpip.h>
//#define socklen_t int

#else

#include <sys/socket.h>

#if defined(__APPLE__) && defined(__GNUC__) && __GNUC__ < 4
#define socklen_t int
#endif

#if defined(__sgi) || defined(__hpux)
#define socklen_t int
#endif

#endif /* _WIN32 */

#endif
