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
 ** based on vncsockets.cpp from VREng                                     **
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
/*
 *  Copyright (C) 1999 AT&T Laboratories Cambridge.  All Rights Reserved.
 *
 *  This is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This software is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this software; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307,
 *  USA.
 */

#ifndef WIN32
#include <stdint.h>
#include <unistd.h>
#include <netdb.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

#ifndef _WIN32
#define closesocket close
#endif

#include "vncsockets.hpp"
#include "vncsocket.h" // sorry about names...

// urgh
// taken from VReng 6.9.0  /src/com/netf.cpp
struct hostent *my_gethostbyname(const char *hostname, int af)
{
    struct hostent *hptmp, *hp;

    // prevent warning about unused parameter af
    (void)af;

    if ((hptmp = gethostbyname(hostname)) == NULL)
        return NULL;
    fprintf(stderr, "my_gethostbyname(): hostname=%s hptmp=%p\n", hostname, hptmp);

    if ((hp = (struct hostent *)malloc(sizeof(struct hostent))) != NULL)
    {
        fprintf(stderr, "my_gethostbyname(): hp=%p\n", hp);
        memcpy(hp, hptmp, sizeof(struct hostent));
    }
    return hp;
}

// Constructors

VNCSockets::VNCSockets(const char *_servername, uint16_t _port = 5901)
{
    strcpy(ServerName, _servername);
    port = _port;
    buffered = 0;
    bufoutptr = buf;
    rfbsock = -1;
    if (!StringToIPAddr())
    {
        fprintf(stderr, "VNCSockets::VNCSockets(): can't resolve %s\n", _servername);
    }
}

VNCSockets::VNCSockets(uint32_t IPAddr, uint16_t _port = 5901)
{
    fprintf(stderr, "VNCSockets::VNCSockets(): 3 not used, port=%d\n", _port);
    port = _port;
    buffered = 0;
    bufoutptr = buf;
    rfbsock = -1;
    ipaddr = IPAddr;
}

/*
 * ReadFromRFBServer is called whenever we want to read some data from the RFB
 * server.  It is non-trivial for two reasons:
 *
 * 1. For efficiency it performs some intelligent buffering, avoiding invoking
 *    the read() system call too often.  For small chunks of data, it simply
 *    copies the data out of an internal buffer.  For large amounts of data it
 *    reads directly into the buffer provided by the caller.
 *
 * 2. Whenever read() would block, it invokes the Xt event dispatching
 *    mechanism to process X events.  In fact, this is the only place these
 *    events are processed, as there is no XtAppMainLoop in the program.
 */

/*
 * Read bytes from the server
 */
bool VNCSockets::ReadFromRFBServer(char *out, uint32_t n)
{
    if (n <= buffered)
    {
        memcpy(out, bufoutptr, n);
        bufoutptr += n;
        buffered -= n;
        return true;
    }
    memcpy(out, bufoutptr, buffered);

    out += buffered;
    n -= buffered;
    bufoutptr = buf;
    buffered = 0;

    if (n <= sizeof(buf))
    {
        while (buffered < n)
        {

#ifdef _WIN32
            int i = recv(rfbsock, (char *)buf + buffered, sizeof(buf) - buffered, 0);
#else
            int i = read(rfbsock, buf + buffered, sizeof(buf) - buffered);
#endif
            if (i <= 0)
            {
                if (i < 0)
                {
#ifdef _WIN32
                    if (errno == EAGAIN || (errno == EINTR))
                    {
#else
                    if (errno == EWOULDBLOCK || errno == EAGAIN || (errno == EINTR))
                    {
#endif
                        i = 0;
                    }
                    else
                    {
                        fprintf(stderr, "VNCSockets::ReadFromRFBServer(): read\n");
                        return false;
                    }
                }
                else
                {
                    fprintf(stderr, "VNCSockets::ReadFromRFBServer(): VNC server closed connection\n");
                    return false;
                }
            }
            buffered += i;
        }
        memcpy(out, bufoutptr, n);
        bufoutptr += n;
        buffered -= n;
        return true;
    }
    else
    {
        while (n > 0)
        {
#ifdef _WIN32
            int i = recv(rfbsock, out, n, 0);
#else
            int i = read(rfbsock, out, n);
#endif
            if (i <= 0)
            {
                if (i < 0)
                {
#ifdef _WIN32
                    if (errno == EAGAIN || (errno == EINTR))
                    {
#else
                    if (errno == EWOULDBLOCK || errno == EAGAIN || (errno == EINTR))
                    {
#endif
                        i = 0;
                    }
                    else
                    {
                        fprintf(stderr, "VNCSockets::ReadFromRFBServer(): read\n");
                        return false;
                    }
                }
                else
                {
                    fprintf(stderr, "VNCSockets::ReadFromRFBServer(): VNC server closed connection\n");
                    return false;
                }
            }
            out += i;
            n -= i;
        }
        return true;
    }
}

/*
 * Write an exact number of bytes, and don't return until you've sent them.
 */
bool VNCSockets::WriteExact(char *buf, int n)
{
    for (int i = 0; i < n;)
    {
#ifdef _WIN32
        int j = send(rfbsock, buf + i, (n - i), 0);
#else
        int j = write(rfbsock, buf + i, (n - i));
#endif

        if (j <= 0)
        {
            if (j < 0)
            {
#ifdef _WIN32
                if (errno == EAGAIN || (errno == EINTR))
                {
#else
                if (errno == EWOULDBLOCK || errno == EAGAIN || (errno == EINTR))
                {
#endif
                    fd_set fds;
                    FD_ZERO(&fds);
                    FD_SET(rfbsock, &fds);
                    if (select(rfbsock + 1, NULL, &fds, NULL, NULL) <= 0)
                    {
                        fprintf(stderr, "VNCSockets::WriteExact(): select\n");
                        return false;
                    }
                    j = 0;
                }
                else
                {
                    fprintf(stderr, "VNCSockets::WriteExact(): write\n");
                    return false;
                }
            }
            else
            {
                fprintf(stderr, "VNCSockets::WriteExact() err: write failed\n");
                return false;
            }
        }
        i += j;
    }
    return true;
}

/*
 * ConnectToTcpAddr connects to the given TCP port.
 */
int VNCSockets::ConnectToTcpAddr()
{
    if ((rfbsock = socket(PF_INET, SOCK_STREAM, 0)) < 0)
    {
        fprintf(stderr, "VNCSockets::ConnectToTcpAddr() err: socket\n");
        return -1;
    }

    struct sockaddr_in sa;

    sa.sin_family = AF_INET;
    sa.sin_port = htons(port);
    sa.sin_addr.s_addr = htonl(ipaddr);

    fprintf(stderr, "VNCSockets::ConnectToTcpAddr(): connecting to %s:%i %x\n",
            ServerName, ntohs(sa.sin_port), ntohl(sa.sin_addr.s_addr));

    if (connect(rfbsock, (const struct sockaddr *)&sa, sizeof(sa)) < 0)
    {
        perror("VNC: connect");
        fprintf(stderr, "VNCSockets::ConnectToTcpAddr() err: %s (%d) sock=%d\n", strerror(errno), errno, rfbsock);
        closesocket(rfbsock);
        return -1;
    }
    if (setTcpNoDelay(rfbsock) < 0)
    {
        fprintf(stderr, "VNCSockets::ConnectToTcpAddr() err: TCP_NODELAY %s (%d)\n", strerror(errno), errno);
        //pd close(rfbsock);
        //pd return -1;
    }
    return rfbsock;
}

/*
 * SetNonBlocking sets a socket into non-blocking mode.
 */
bool VNCSockets::SetNonBlocking()
{
    if (setNoBlocking(rfbsock) < 0)
        return false;
    return true;
}

/*
 * StringToIPAddr - convert a host string to an IP address.
 */
bool VNCSockets::StringToIPAddr()
{
    struct hostent *hp;

    if ((hp = my_gethostbyname(ServerName, AF_INET)) != NULL)
    {
        memcpy(&ipaddr, hp->h_addr, hp->h_length);
        ipaddr = ntohl(ipaddr);
        fprintf(stderr, "VNCSockets::StringToIPAddr(): ServerName=%s (%x)\n", ServerName, ipaddr);
        return true;
    }
    return false;
}

/*
 * Test if the other end of a socket is on the same machine.
 */
bool VNCSockets::SameMachine()
{
    struct sockaddr_in peersa, mysa;
#ifdef WIN32
    int slen = sizeof(struct sockaddr_in);
#else
    socklen_t slen = sizeof(struct sockaddr_in);
#endif

    getpeername(rfbsock, (struct sockaddr *)&peersa, &slen);
    getsockname(rfbsock, (struct sockaddr *)&mysa, &slen);

    return (peersa.sin_addr.s_addr == mysa.sin_addr.s_addr);
}

#if 0 //unused
/*
 * Print out the contents of a packet for debugging.
 */
void VNCSockets::PrintInHex(char *buf, int len)
{
  int i;
  char c, str[17];

  str[16] = 0;

  trace(DBG_VNC, "ReadExact: ");

  for (i = 0; i < len; i++) {
    if ((i % 16 == 0) && (i != 0))
      fprintf(stderr, "           ");
    c = buf[i];
    str[i % 16] = (((c > 31) && (c < 127)) ? c : '.');
    fprintf(stderr,"%02x ", (uint8_t) c);
    if ((i % 4) == 3)
      fprintf(stderr, " ");
    if ((i % 16) == 15)
      fprintf(stderr,"%s\n", str);
  }
  if ((i % 16) != 0) {
    for (int j = i % 16; j < 16; j++) {
      fprintf(stderr, "   ");
      if ((j % 4) == 3) fprintf(stderr, " ");
    }
    str[i % 16] = 0;
    fprintf(stderr,"%s\n", str);
  }
  fflush(stderr);
}
#endif //unused

/*
 * Returns the socket used
 */
int VNCSockets::GetSock()
{
    return rfbsock;
}
