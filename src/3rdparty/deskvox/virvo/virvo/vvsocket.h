// Virvo - Virtual Reality Volume Rendering
// Copyright (C) 1999-2003 University of Stuttgart, 2004-2005 Brown University
// Contact: Jurgen P. Schulze, jschulze@ucsd.edu
//
// This file is part of Virvo.
//
// Virvo is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library (see license.txt); if not, write to the
// Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

#ifndef VV_SOCKET_H
#define VV_SOCKET_H

#include <iostream>

#include "vvplatform.h"
#ifndef _WIN32
# include <netdb.h>
# include <unistd.h>
# include <arpa/inet.h>
# include <fcntl.h>
# include <netinet/in.h>
# include <netinet/tcp.h>
# include <sys/time.h>
# include <sys/errno.h>
# include <sys/param.h>
# include <sys/ioctl.h>
# include <sys/socket.h>
# include <sys/wait.h>
# include <errno.h>
#endif
#include <string.h>
#include <signal.h>
#include <stdlib.h>
#ifdef __sun
#include <sys/filio.h>
#endif

#include "vvexport.h"
#include "vvinttypes.h"

#ifdef _WIN32
typedef SOCKET vvsock_t;
#else
typedef int vvsock_t;
#endif

#ifdef _WIN32
#define VV_INVALID_SOCKET INVALID_SOCKET
#else
#define VV_INVALID_SOCKET (-1)
#endif

//----------------------------------------------------------------------------
/** This abstract class provides basic socket functionality. It is used for
    TCP and UDP sockets. For example code see documentation about vvSocket<BR>

    For timeout-support set Socket to non-blocking with
    setParameter(VV_NONBLOCKING, true) and use vvSocketMonitor (or select() by
    hand) for event and timeout-handling
*/
class VIRVOEXPORT vvSocket
{
public:
  typedef void Sigfunc(int);

  enum ErrorType            /// Error Codes
  {
    VV_OK,                  ///< no error
    VV_SOCK_ERROR,
    VV_WRITE_ERROR,
    VV_READ_ERROR,
    VV_ACCEPT_ERROR,
    VV_CONNECT_ERROR,
    VV_HOST_ERROR,
    VV_RETRY,
    VV_ALLOC_ERROR,         ///< allocation error: not enough memory
    VV_CREATE_ERROR,        ///< socket could not be opened
    VV_HEADER_ERROR,        ///< invalid header received
    VV_DATA_ERROR,          ///< volume data format error: e.g., too many voxels received
    VV_PEER_SHUTDOWN,       ///< the connection was closed by peer
    VV_SOCKFD_ERROR,
    VV_SOCKOPT_ERROR
  };

  enum EndianType           /// endianness
  {
    VV_LITTLE_END,          ///< little endian: low-order byte is stored first
    VV_BIG_END              ///< big endian: high-order byte is stored first
  };

  enum SocketOption
  {
    VV_NONBLOCKING,
    VV_NO_NAGLE,
    VV_LINGER,
    VV_BUFFSIZE
  };

  vvSocket();
  virtual ~vvSocket();

  /** Sets socket options
    \param so desired socket option to set
    \param appropriate value of socket option, always casted to float
    */
  virtual ErrorType setParameter(SocketOption so, float value);

  ErrorType readString(char* , int);
  ErrorType writeString(const char*);
  uchar     read8();
  ErrorType write8(uchar);
  ushort    read16(EndianType = VV_BIG_END);
  ErrorType write16(ushort, EndianType = VV_BIG_END);
  uint      read32(EndianType = VV_BIG_END);
  ErrorType write32(uint, EndianType = VV_BIG_END);
  float     readFloat(EndianType = VV_BIG_END);
  ErrorType writeFloat(float, EndianType = VV_BIG_END);

  virtual ErrorType readData (      uchar *dataptr, size_t size, ssize_t *ret = NULL);
  virtual ErrorType writeData(const uchar *dataptr, size_t size, ssize_t *ret = NULL);

  int isDataWaiting() const;
  void setSockfd(vvsock_t fd);
  vvsock_t  getSockfd() const;
  int getRecvBuffsize();
  int getSendBuffsize();
  int getMTU();

protected:
  Sigfunc *signal(int, Sigfunc *);
  Sigfunc *Signal(int, Sigfunc *);
  static void noNameServer(int );
  static void peerUnreachable(int );
  static void interrupter(int );

  void printErrorMessage(const char* = NULL) const;

  ErrorType read(uchar*,        size_t);
  ErrorType write(const uchar*, size_t);

  virtual ssize_t readn (char*,       size_t) = 0;
  virtual ssize_t writen(const char*, size_t) = 0;

  int measureBdpServer();
  int measureBdpClient();
  int   RttServer(int);
  float RttClient(int);
  int checkMssMtu(int, int);
  EndianType getEndianness();


  vvsock_t _sockfd;

  ushort              _port;
  const char         *_hostname;
  struct sockaddr_in  _hostAddr;
  struct hostent     *_host;
  int _clMinPort;
  int _clMaxPort;

  int _sockBuffsize;
  int _recvBuffsize;
  int _sendBuffsize;

  socklen_t _hostAddrLen;
  socklen_t _bufflen;
};

//----------------------------------------------------------------------------
/***Tcp-Sockets***<BR>

 Features:
    - socket buffer sizes can be set be user
    - automatic bandwidth delay product discovery to set the socket buffers to the
      optimal values. Not supported under Windows and when VV_BDP Flag is not set.
      For automatic banwidth delay product discovery the socket buffer size has
      to be set to 0. Optimized for networks with more than 10 Mbits/sec. Please
      don't use if you have a lower speed (would take awhile).
    - Nagle algorithm can be disabled
    - Linger time can be set<BR>

Default values:
  - socket buffer size= system default<BR>

Here is an example code fragment to generate a TCP-client which reads 10 bytes.<BR>
<PRE>

// Create a new tcp socket class instance which shall connect to a server
// with name buxdehude on port 17171. The outgoing port shall be in the
// range between 31000 and 32000:
char* servername = "buxdehude";
vvTcpSocket* sock = new vvTcpSocket();

// Parameters must be set before the connectToHost() call !!
// e.g. socket buffer size= 65535 byte, \
sock->setParameter(VV_BUFFSIZE, 65535);

// Initialize the socket with the parameters and connect to the server.
if (sock->connectToHost(servername, 17171, 31000, 32000);) != vvSocket::VV_OK)
{
  delete sock;
  return -1;
}

// Get 10 bytes of data with read_data()
uchar buffer[10];
if (sock->readData(buffer, 10) != vvSocket::VV_OK)
{
  delete sock;
  return -1;
}

// Delete the socket object
delete sock; </PRE>
*/

//----------------------------------------------------------------------------
/***For UDP Sockets***  <BR>

 features:
    - socket buffer sizes can be set be user<BR>

 default values:
   - socket buffer size= system default<BR>

Here is an example code fragment to generate a UDP server which sends 10 bytes
and a UDP client which reads 10 bytes.<BR>
<PRE>

UDP-Server:

// Create a new UDP socket class instance which shall listen on port 17171:
vvUdpSocket* sock = new vvUdpSocket();

// Parameters must be set before the bind() call !!
// e.g. socket buffer size= 65535 byte, \
sock->setParameter(VV_BUFFSIZE, 65535);

// Initialize the socket with the parameters and wait for a client
if (sock->bind(17171) != vvSocket::VV_OK)
{
  delete sock;
  return -1;
}

// Send 10 bytes of data with write_data()
uchar buffer[10];
if (sock->writeData(&buffer, 10) != vvSocket::VV_OK)
{
  delete sock;
  return -1;
}

// Delete the socket object
delete sock;

UDP-Client:

// Create a new UDP socket class instance which shall connect to a server
// with name buxdehude on port 17171. The outgoing port shall be in the
// range between 31000 and 32000:
char* servername = "buxdehude";
vvUdpSocket* sock = new vvUdpSocket();

// Parameters must be set before the bind() call !!
// e.g. socket buffer size= 65535 byte, \
sock->setParameter(VV_BUFFSIZE, 65535);

// Initialize the socket with the parameters and connect to the server.
if (sock->bind(servername, 17171, 31000, 32000) != vvSocket::VV_OK)
{
  delete sock;
  return -1;
}

// Get 10 bytes of data with read_data()
uchar buffer[10];
if (sock->readData(&buffer, 10) != vvSocket::VV_OK)
{
  delete sock;
  return -1;
}

// Delete the socket object
delete sock; </PRE>
@author Michael Poehnl
*/
#endif
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
