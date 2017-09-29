/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                          (C)2007 HLRS  **
 **                                                                        **
 ** Description: VNC Client                                                **
 **                                                                        **
 **                                                                        **
 ** Author: Lukas Pinkowski                                                **
 **                                                                        **
 ** based on vncsocket.cpp from VREng                                      **
 **                                                                        **
 ** History:                                                               **
 ** Oct 12, 2007                                                           **
 **                                                                        **
 ** License: GPL v2 or later                                               **
 **                                                                        **
\****************************************************************************/

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

#include <errno.h>
#include <string.h>
#include <stdio.h>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <unistd.h>
#include <netdb.h>
#endif

#include "vncsocket.h"
//#include "socket.h"

//#include "global.hh"
//#include "net.hh"
#ifndef WIN32
#define TCP_NODELAY 1
#endif

/** Open a datagram socket */
SOCKET socketDatagram()
{
	SOCKET sock = -1;

    if ((sock = socket(PF_INET, SOCK_DGRAM, 0)) < 0)
    {
        fprintf(stderr, "socketDatagram() err: %s\n", strerror(errno));
    }
    return sock;
}

/** Open a stream socket */
SOCKET socketStream()
{
	SOCKET sock = -1;

    if ((sock = socket(PF_INET, SOCK_STREAM, 0)) < 0)
    {
        fprintf(stderr, "socketStream() err: %s\n", strerror(errno));
    }
    return sock;
}

SOCKET socketConnect(int sock, const sockaddr_in *sa)
{
    if (connect(sock, (struct sockaddr *)sa, sizeof(struct sockaddr_in)) < 0)
    {
        fprintf(stderr, "socketConnect() err: %s\n", strerror(errno));
#ifdef _WIN32
        closesocket(sock);
#else
        close(sock);
#endif
        return -1;
    }
    return sock;
}

/**
 * Do REUSEADDR on the socket
 * return sock if OK, else -1
 */
SOCKET setReuseAddr(SOCKET sock)
{
    const int one = 1;

    if (setsockopt(sock, SOL_SOCKET, SO_REUSEADDR,
                   (char *)&one, sizeof(one)) < 0)
    {
        fprintf(stderr, "SO_REUSEADDR: %s (%d)\n", strerror(errno), errno);
        return -1;
    }
    return sock;
}

SOCKET setTcpNoDelay(SOCKET sock)
{
    const int one = 1;

    // there is a problem with setting this socket option
    struct protoent *p;
    p = getprotobyname("tcp");
    if (p && setsockopt(sock, p->p_proto, TCP_NODELAY, (char *)&one, sizeof(one)) < 0)
    {
        fprintf(stderr, "TCP_NODELAY: %s (%d) sock=%d\n", strerror(errno), errno, (int)sock);
        return -1;
    }

    //if (setsockopt(sock, SOL_SOCKET, TCP_NODELAY,
    //		 (char *) &one, sizeof(one)) < 0) {
    //    fprintf(stderr, "TCP_NODELAY: %s (%d) sock=%d\n", strerror(errno), errno,sock);
    //    return -1;
    //  }
    return sock;
}

uint16_t getSrcPort(SOCKET sock)
{
    struct sockaddr_in sa;
#ifdef WIN32
    int slen = sizeof(struct sockaddr_in);
#else
    socklen_t slen = sizeof(struct sockaddr_in);
#endif

    sa.sin_family = AF_INET;
    sa.sin_port = 0;
    sa.sin_addr.s_addr = htonl(INADDR_ANY);
    if (getsockname(sock, (struct sockaddr *)&sa, &slen) < 0)
    {
        fprintf(stderr, "getSrcPort() err: %s (%d)\n", strerror(errno), errno);
        return 0;
    }
    return ntohs(sa.sin_port);
}

/**
 * Control blocking(1)/non blocking(0)
 * return sock if OK, else -1
 */
SOCKET handleBlocking(SOCKET sock, bool block)
{
#if HAVE_FCNTL
    int flags;

    if ((flags = fcntl(sock, F_GETFL)) < 0)
    {
        fprintf(stderr, "handleBlocking(): F_GETFL: %s (%d)\n", strerror(errno), errno);
        return -1;
    }
    if (block)
        flags &= ~O_NONBLOCK;
    else
        flags |= O_NONBLOCK;
    if (fcntl(sock, F_SETFL, flags) < 0)
    {
        fprintf(stderr, "handleBlocking(): F_SETFL: %s (%d)\n", strerror(errno), errno);
        return -1;
    }
#else
    (void)block;
#endif // HAVE_FCNTL
    return sock;
}

SOCKET setBlocking(SOCKET sock)
{
    return handleBlocking(sock, true);
}

SOCKET setNoBlocking(SOCKET sock)
{
    return handleBlocking(sock, false);
}

/**
 * Set the ttl
 * return sock if OK, else -1
 */
SOCKET setScope(SOCKET sock, uint8_t _ttl)
{
#if defined(__WIN32__) || defined(_WIN32)
#define TTL_TYPE int
#else
#define TTL_TYPE uint8_t
#endif
    if (_ttl)
    {
        TTL_TYPE ttl = (TTL_TYPE)_ttl;
        if (setsockopt(sock, IPPROTO_IP, IP_MULTICAST_TTL, (const char *)&ttl, sizeof(ttl)) < 0)
        {
            fprintf(stderr, "setScope(): IP_MULTICAST_TTL: %s (%d)\n", strerror(errno), errno);
            return -1;
        }
    }
    return sock;
}

/**
 * Set loopback: active (1) either inactive (0)
 * return sock if OK, else -1
 */
SOCKET handleLoopback(SOCKET sock, uint8_t loop)
{
#ifdef IP_MULTICAST_LOOP // Windoze doesn't handle IP_MULTICAST_LOOP
    if (setsockopt(sock, IPPROTO_IP, IP_MULTICAST_LOOP,
#ifdef WIN32
                   (char *)&loop, sizeof(loop)) < 0)
    {
#else
                   &loop, sizeof(loop)) < 0)
    {
#endif
#if IPMC_ENABLED
        fprintf(stderr, "handleLoopback(): IP_MULTICAST_LOOP: %s (%d)\n", strerror(errno), errno);
#endif
        return -1;
    }
#endif
    return sock;
}

SOCKET setLoopback(SOCKET sock)
{
    return handleLoopback(sock, 1);
}

SOCKET setNoLoopback(SOCKET sock)
{
    return handleLoopback(sock, 0);
}

SOCKET addMembership(SOCKET sock, const void *pmreq)
{
    if (setsockopt(sock, IPPROTO_IP, IP_ADD_MEMBERSHIP,
#ifdef WIN32
                   (char *)pmreq, sizeof(struct ip_mreq)) < 0)
    {
#else
                   pmreq, sizeof(struct ip_mreq)) < 0)
    {
#endif
#if IPMC_ENABLED
        fprintf(stderr, "addMembership(): IP_ADD_MEMBERSHIP: %s (%d)\n", strerror(errno), errno);
#endif
        return -1;
    }
    return 0;
}

SOCKET dropMembership(SOCKET sock, const void *pmreq)
{
    if (setsockopt(sock, IPPROTO_IP, IP_DROP_MEMBERSHIP,
#ifdef WIN32
                   (char *)pmreq, sizeof(struct ip_mreq)) < 0)
    {
#else
                   pmreq, sizeof(struct ip_mreq)) < 0)
    {
#endif
#if IPMC_ENABLED
        fprintf(stderr, "dropMembership(): IP_DROP_MEMBERSHIP: %s (%d)\n", strerror(errno), errno);
#endif
        return -1;
    }
    return 0;
}

/** Set a Multicast socket */
void setSendSocket(SOCKET sock, uint8_t ttl)
{
#ifdef PROXY_MULTICAST
    if (mup.active)
    {
        setNoBlocking(sock);
        return;
    }
#endif

    setScope(sock, ttl);
#if NEEDLOOPBACK
    setLoopback(sock); // loopback
#else
    setNoLoopback(sock); // no loopback (default)
#endif
    setNoBlocking(sock);
}

/**
 * Create an UDP socket
 * Prevue pour emettre (uni et mcast) et recevoir (unicast donc)
 * Do setScope (ttl) and setLoopback (off)
 * return sock if OK, else -1
 */
SOCKET createSendSocket(uint8_t ttl)
{
	SOCKET sock;

    if ((sock = socketDatagram()) >= 0)
        setSendSocket(sock, ttl);
    return sock;
}

/**
 * Create an Unicast socket
 * return fd, -1 if problem
 */
SOCKET createUcastSocket(uint32_t uni_addr, uint16_t port)
{
	SOCKET sock;
    struct sockaddr_in sa;

    if ((sock = socketDatagram()) < 0)
        return -1;
    memset(&sa, 0, sizeof(struct sockaddr_in));
    sa.sin_family = AF_INET;
    sa.sin_port = htons(port);
    sa.sin_addr.s_addr = htonl(uni_addr);

    setReuseAddr(sock);
    if (bind(sock, (struct sockaddr *)&sa, sizeof(struct sockaddr_in)) < 0)
    {
        fprintf(stderr, "createUcastSocket(): receive unicast bind: %s\n", strerror(errno));
    }
    return sock;
}

bool isMulticastAddress(uint32_t address)
{
    // Note: We return False for addresses in the range 224.0.0.0
    // through 224.0.0.255, because these are non-routable
    unsigned addressInHostOrder = ntohl(address);
    return addressInHostOrder > 0xE00000FF && addressInHostOrder <= 0xEFFFFFFF;
}
