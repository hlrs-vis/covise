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

#include <cassert>
#include <sstream>

#include "vvtcpsocket.h"
#include "vvdebugmsg.h"

//----------------------------------------------------------------------------
/** Constructor
*/
vvTcpSocket::vvTcpSocket() : vvSocket()
{
}

//----------------------------------------------------------------------------
/** Destructor
*/
vvTcpSocket::~vvTcpSocket()
{
}



vvSocket::ErrorType vvTcpSocket::connectToHost(const std::string& host, const ushort port, const int clminport, const int clmaxport)
{
  _hostname = host.c_str();
  _port = port;
  _clMinPort = clminport;
  _clMaxPort = clmaxport;

  int cl_port;

#ifdef _WIN32
  if ((_host= gethostbyname(_hostname)) == 0)
  {
    vvDebugMsg::msg(1, "Error: gethostbyname()");
    return VV_HOST_ERROR;
  }
#else
  Sigfunc *sigfunc;

  sigfunc = Signal(SIGALRM, noNameServer);
  if (alarm(5) != 0)
    vvDebugMsg::msg(2, "init_client(): WARNING! previously set alarm was wiped out");
  if ((_host = gethostbyname(_hostname)) == 0)
  {
    vvDebugMsg::msg(1, "Error: gethostbyname()");
    alarm(0);
    signal(SIGALRM, sigfunc);
    return VV_HOST_ERROR;
  }
  alarm(0);
  signal(SIGALRM, sigfunc);
#endif

  _sockfd = socket(AF_INET, SOCK_STREAM, 0);

  if (_sockfd == VV_INVALID_SOCKET)
  {
    vvDebugMsg::msg(1, "Error: socket", true);
    return VV_SOCK_ERROR;
  }
  if (_sockBuffsize == 0)
  {
#if !defined(_WIN32) && defined(VV_BDP)
    if (measure_BDP_client())
    {
      vvDebugMsg::msg(1, "Error: measure_BDP_client()");
      return VV_SOCK_ERROR;
    }
#else
    _sockBuffsize = getSendBuffsize();
#endif
  }
  if (_sockBuffsize > 0)
  {
    if (setsockopt(_sockfd, SOL_SOCKET, SO_SNDBUF,
      (char *) &_sockBuffsize, sizeof(_sockBuffsize)))
    {
      vvDebugMsg::msg(1, "Error: setsockopt()");
      return VV_SOCK_ERROR;
    }
    if (setsockopt(_sockfd, SOL_SOCKET, SO_RCVBUF,
      (char *) &_sockBuffsize, sizeof(_sockBuffsize)))
    {
      vvDebugMsg::msg(1, "Error: setsockopt()");
      return VV_SOCK_ERROR;
    }
  }
  memset((char *) &_hostAddr, 0, sizeof(_hostAddr));
  _hostAddr.sin_family = AF_INET;
  _hostAddrLen = sizeof(_hostAddr);
  if (_clMinPort != 0 || _clMaxPort != 0)
  {
    if (_clMinPort > _clMaxPort)
    {
      vvDebugMsg::msg(1,"Wrong port range");
      return VV_SOCK_ERROR ;
    }
    _hostAddr.sin_addr.s_addr = INADDR_ANY;
    cl_port = _clMinPort;
    _hostAddr.sin_port = htons((unsigned short)cl_port);
    while (bind(_sockfd, (struct sockaddr *)&_hostAddr, _hostAddrLen) && cl_port <= _clMaxPort)
    {
#ifdef _WIN32
      if (WSAGetLastError() == WSAEADDRINUSE)
#else
        if (errno == EADDRINUSE)
#endif
      {
        cl_port ++;
        _hostAddr.sin_port = htons((unsigned short)cl_port);
      }
      else
      {
        vvDebugMsg::msg(1, "Error: bind()");
        return VV_SOCK_ERROR ;
      }
    }
    if (cl_port > _clMaxPort)
    {
      vvDebugMsg::msg(1,"No port free!");
      return VV_SOCK_ERROR ;
    }
  }
  _hostAddr.sin_addr = *((struct in_addr *)_host->h_addr);
  _hostAddr.sin_port = htons((unsigned short)port);

  if(vvDebugMsg::getDebugLevel() >= 2)
  {
    std::ostringstream errmsg;
    errmsg << "send_buffsize: " << getSendBuffsize() << " bytes, recv_buffsize: " << getRecvBuffsize() << " bytes";
    vvDebugMsg::msg(0, errmsg.str().c_str());
  }

  if (connect(_sockfd, (struct sockaddr *)&_hostAddr, _hostAddrLen))
  {
    vvDebugMsg::msg(1, "Error: connect()");
    return VV_CONNECT_ERROR;
  }
  else
  {
    return VV_OK;
  }
}

vvSocket::ErrorType vvTcpSocket::disconnectFromHost()
{
  if (_sockfd == VV_INVALID_SOCKET)
  {
    vvDebugMsg::msg(1, "vvTcpSocket::disconnectFromHost() error: called on unbound socket");
    return VV_SOCK_ERROR;
  }

#ifdef _WIN32
  if (0 == closesocket(_sockfd))
#else
  if (0 == close(_sockfd))
#endif
  {
    _sockfd = VV_INVALID_SOCKET;
    return VV_OK;
  }

#ifdef _WIN32
  if (WSAGetLastError() == WSAEWOULDBLOCK)
#else
  if (errno == EWOULDBLOCK)
#endif
  {
    vvDebugMsg::msg(1, "Linger time expires");
  }

  return VV_SOCK_ERROR;
}

//----------------------------------------------------------------------------
/** Reads data from the TCP socket.
 @param buffer  pointer to the data to write
 @param size   number of bytes to write

*/
ssize_t vvTcpSocket::readn(char* buffer, size_t size)
{
  size_t nleft;
  ssize_t nread;

  nleft = size;
  while(nleft > 0)
  {
    nread = recv(_sockfd, buffer, (int)nleft, 0);
    if (nread < 0)
    {
#ifdef _WIN32
      if (WSAGetLastError() == WSAEINTR)
#else
        if (errno == EINTR)
#endif
          nread = 0;                              // interrupted, call read again
      else
      {
        vvDebugMsg::msg(1, "Error: recv()");
        return (ssize_t)-1;
      }
    }
    else if (nread == 0)
      break;

    nleft -= nread;
    buffer += nread;
  }
  return (size - nleft);
}

//----------------------------------------------------------------------------
/** Writes data to the TCP socket.
 @param buffer  pointer to the data to write
 @param size   number of bytes to write

*/
ssize_t vvTcpSocket::writen(const char* buffer, size_t size)
{
  size_t nleft;
  ssize_t nwritten;

#ifndef _WIN32
  ::signal(SIGPIPE, peerUnreachable);
#endif

  nleft = size;
  while(nleft > 0)
  {
    nwritten = send(_sockfd, buffer, (int)nleft, 0);
    if (nwritten < 0)
    {
#ifdef _WIN32
      if (WSAGetLastError() == WSAEINTR)
#else
        if (errno == EINTR)
#endif
          nwritten = 0;                           // interrupted, call write again
      else
      {
        vvDebugMsg::msg(1, "Error: send()");
        return (ssize_t)-1;
      }
    }

    nleft -= nwritten;
    buffer += nwritten;
  }

  return size;
}

// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
