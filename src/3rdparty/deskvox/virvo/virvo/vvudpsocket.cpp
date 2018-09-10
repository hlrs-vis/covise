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

#include "vvudpsocket.h"
#include "vvdebugmsg.h"

vvUdpSocket::vvUdpSocket() : vvSocket()
{
  _mc = false;
}

vvUdpSocket::~vvUdpSocket()
{
}

vvSocket::ErrorType vvUdpSocket::bind(const std::string& hostname, const ushort port, const int clmin, const int clmax)
{
  _hostname = hostname.c_str();
  _port = port;
  _clMinPort = clmin;
  _clMaxPort = clmax;

  int cl_port;
  uchar buff;

#ifdef _WIN32
  _host= gethostbyname(hostname.c_str());
  if (_host == 0)
  {
    vvDebugMsg::msg(1, "Error gethostbyname()");
    return VV_HOST_ERROR;
  }
#else
  Sigfunc *sigfunc;

  sigfunc = Signal(SIGALRM, noNameServer);
  if (alarm(5) != 0)
  {
    vvDebugMsg::msg(2, "init_client():WARNING! previously set alarm was wiped out");
  }
  if ((_host= gethostbyname(_hostname)) == 0)
  {
    alarm(0);
    vvDebugMsg::msg(1, "Error: gethostbyname()");
    signal(SIGALRM, sigfunc);
    return VV_HOST_ERROR;
  }
  alarm(0);
  signal(SIGALRM, sigfunc);
#endif

  _sockfd = socket(AF_INET, SOCK_DGRAM, 0);
  if (_sockfd == VV_INVALID_SOCKET)
  {
    vvDebugMsg::msg(1, "Error: socket", true);
    return VV_SOCK_ERROR;
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
  getSendBuffsize();
  std::ostringstream errmsg;
  errmsg << "send_buffsize: " << _sendBuffsize << "bytes,  recv_buffsize: " << getRecvBuffsize() << "bytes";
  vvDebugMsg::msg(1, errmsg.str().c_str());
  if (_sendBuffsize < 65507)
    _maxSendSize = _sendBuffsize;
  else
    _maxSendSize = 65507;
  memset((char *) &_hostAddr, 0, sizeof(_hostAddr));
  _hostAddr.sin_family = AF_INET;
  _hostAddrLen = sizeof(_hostAddr);
  if (_clMinPort != 0 || _clMaxPort != 0)
  {
    if (_clMinPort > _clMaxPort)
    {
      vvDebugMsg::msg(1, "Wrong port range");
      return VV_SOCK_ERROR ;
    }
    _hostAddr.sin_addr.s_addr = INADDR_ANY;
    cl_port = _clMinPort;
    _hostAddr.sin_port = htons((unsigned short)cl_port);
    while (::bind(_sockfd, (struct sockaddr *)&_hostAddr, _hostAddrLen) && cl_port <= _clMaxPort)
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
      vvDebugMsg::msg(1, "No port free!");
      return VV_SOCK_ERROR ;
    }
  }
  _hostAddr.sin_addr = *((struct in_addr *)_host->h_addr);
  _hostAddr.sin_port = htons((unsigned short)_port);
  if (connect(_sockfd, (struct sockaddr *)&_hostAddr, _hostAddrLen))
  {
    vvDebugMsg::msg(1, "Error: connect()");
    return VV_CONNECT_ERROR;
  }
  if (writeData(&buff, 1) != VV_OK)
  {
    vvDebugMsg::msg(1, "Error: write_data()");
    return VV_WRITE_ERROR;
  }
  return VV_OK;
}


//----------------------------------------------------------------------------
/// Initialize a UDP server
vvSocket::ErrorType vvUdpSocket::bind(const ushort port)
{
  _port = port;

#ifdef _WIN32
  char optval=1;
#else
  int optval=1;
#endif
  ErrorType retval;
  _sockfd = socket(AF_INET, SOCK_DGRAM, 0);
  if (_sockfd == VV_INVALID_SOCKET)
  {
    vvDebugMsg::msg(1, "Error: socket()", true);
    return VV_SOCK_ERROR;
  }
  if (setsockopt(_sockfd, SOL_SOCKET, SO_REUSEADDR, &optval,sizeof(optval)))
  {
    vvDebugMsg::msg(1, "Error: setsockopt()");
    return VV_SOCK_ERROR;
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
  getSendBuffsize();
  std::ostringstream errmsg;
  errmsg << "send_buffsize: " << _sendBuffsize << " bytes, recv_buffsize: " << getRecvBuffsize() << " bytes";
  vvDebugMsg::msg(2, errmsg.str().c_str());
  if (_sendBuffsize < 65507)
    _maxSendSize = _sendBuffsize;
  else
    _maxSendSize = 65507;
  memset((char *) &_hostAddr, 0, sizeof(_hostAddr));
  _hostAddr.sin_family = AF_INET;
  _hostAddr.sin_port = htons((unsigned short)_port);
  _hostAddr.sin_addr.s_addr = INADDR_ANY;
  _hostAddrLen = sizeof(_hostAddr);

  if (::bind(_sockfd, (struct sockaddr *)&_hostAddr, _hostAddrLen))
  {
    vvDebugMsg::msg(1, "Error: bind()");
    return VV_SOCK_ERROR;
  }

  if ((retval = getClientAddr()) != VV_OK)
  {
    vvDebugMsg::msg(1, "Error: get_client_addr()");
    return VV_READ_ERROR;
  }

  if (connect(_sockfd, (struct sockaddr *)&_hostAddr, _hostAddrLen))
  {
    vvDebugMsg::msg(1, "Error: connect()");
    return VV_CONNECT_ERROR;
  }
  return VV_OK;
}

vvSocket::ErrorType vvUdpSocket::unbind()
{
  if (_sockfd == VV_INVALID_SOCKET)
  {
    vvDebugMsg::msg(1, "vvUdpSocket::unbind() error: called on unbound socket");
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

vvSocket::ErrorType vvUdpSocket::multicast(const std::string& hostname, const ushort port, const MulticastType type)
{
  _hostname = hostname.c_str();
  _port = port;

  if(VV_MC_SENDER == type)
  {
    vvDebugMsg::msg(3, "Enter vvUdpSocket::multicast() VV_MC_SENDER");
    _mc = true;

    _sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (_sockfd == VV_INVALID_SOCKET)
    {
      vvDebugMsg::msg(1, "Error: socket", true);
      return VV_SOCK_ERROR;
    }
    if (_sockBuffsize > 0)
    {
      if (setsockopt(_sockfd, SOL_SOCKET, SO_SNDBUF, (char *) &_sockBuffsize, sizeof(_sockBuffsize)) < 0)
      {
        vvDebugMsg::msg(1, "Error: setsockopt()");
        return VV_SOCK_ERROR;
      }
      if (setsockopt(_sockfd, SOL_SOCKET, SO_RCVBUF, (char *) &_sockBuffsize, sizeof(_sockBuffsize)) < 0)
      {
        vvDebugMsg::msg(1, "Error: setsockopt()");
        return VV_SOCK_ERROR;
      }
    }
    getSendBuffsize();
    std::ostringstream errmsg;
    errmsg << "send_buffsize: " << _sendBuffsize << "bytes,  recv_buffsize: " << getRecvBuffsize() << "bytes";
    vvDebugMsg::msg(1, errmsg.str().c_str());
    if (_sendBuffsize < 65507)
      _maxSendSize = _sendBuffsize;
    else
      _maxSendSize = 65507;

    memset((char *) &_groupSock, 0, sizeof(_groupSock));
    _groupSock.sin_family = AF_INET;
    _groupSock.sin_addr.s_addr = inet_addr(_hostname);
    _groupSock.sin_port = htons(_port);

    char loopch = 0;
    if(setsockopt(_sockfd, IPPROTO_IP, IP_MULTICAST_LOOP, (char *)&loopch, sizeof(loopch)) < 0)
    {
      vvDebugMsg::msg(2, "Setting IP_MULTICAST_LOOP error", true);
      return VV_SOCK_ERROR;
    }

    _localInterface.s_addr = INADDR_ANY;
    if(setsockopt(_sockfd, IPPROTO_IP, IP_MULTICAST_IF, (char *)&_localInterface, sizeof(_localInterface)) < 0)
    {
      vvDebugMsg::msg(2, "Setting local interface error", true);
      return VV_SOCK_ERROR;
    }

    return VV_OK;
  }
  else if(VV_MC_RECEIVER == type)
  {
    vvDebugMsg::msg(3, "Enter vvUdpSocket::multicast() VV_MC_RECEIVER");
    _mc = true;

    _sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (_sockfd == VV_INVALID_SOCKET)
    {
      vvDebugMsg::msg(1, "Error: socket()", true);
      return VV_SOCK_ERROR;
    }
    int reuse = 1;
    if (setsockopt(_sockfd, SOL_SOCKET, SO_REUSEADDR, (const char *)&reuse,sizeof(reuse)))
    {
      vvDebugMsg::msg(1, "Error: setsockopt()");
      return VV_SOCK_ERROR;
    }
    if (_sockBuffsize > 0)
    {
      if(setsockopt(_sockfd, SOL_SOCKET, SO_SNDBUF, (char *) &_sockBuffsize, sizeof(_sockBuffsize)) < 0)
      {
        vvDebugMsg::msg(1, "Error: setsockopt()");
        return VV_SOCK_ERROR;
      }
      if(setsockopt(_sockfd, SOL_SOCKET, SO_RCVBUF, (char *) &_sockBuffsize, sizeof(_sockBuffsize)) < 0)
      {
        vvDebugMsg::msg(1, "Error: setsockopt()");
        return VV_SOCK_ERROR;
      }
    }
    getSendBuffsize();
    std::ostringstream errmsg;
    errmsg << "send_buffsize: " << _sendBuffsize << " bytes, recv_buffsize: " << getRecvBuffsize() << " bytes";
    vvDebugMsg::msg(2, errmsg.str().c_str());
    if (_sendBuffsize < 65507)
      _maxSendSize = _sendBuffsize;
    else
      _maxSendSize = 65507;

    memset((char *) &_localSock, 0, sizeof(_localSock));
    _localSock.sin_family = AF_INET;
    _localSock.sin_port = htons(_port);
    _localSock.sin_addr.s_addr = INADDR_ANY;

    if(::bind(_sockfd, (struct sockaddr*)&_localSock, sizeof(_localSock)))
    {
      vvDebugMsg::msg(1, "Error: bind()");
      return VV_SOCK_ERROR ;
    }

    _mcGroup.imr_multiaddr.s_addr = inet_addr(_hostname);
    _mcGroup.imr_interface.s_addr = INADDR_ANY;
    if(setsockopt(_sockfd, IPPROTO_IP, IP_ADD_MEMBERSHIP, (char *)&_mcGroup, sizeof(_mcGroup)) < 0)
    {
      vvDebugMsg::msg(2, "Adding multicast group error", true);
      return VV_SOCK_ERROR;
    }

    return VV_OK;
  }
  else
  {
    return VV_SOCK_ERROR;
  }
}

vvSocket::ErrorType vvUdpSocket::readData(uchar* dataptr, size_t size, ssize_t *ret)
{
  if(_mc)
  {
#ifdef _WIN32
    ssize_t got = recv(_sockfd, (char *)dataptr, (int)size,0);
#else
    ssize_t got = recv(_sockfd, dataptr, size,0);
#endif
    if(ret) *ret = got;
    if(got >= 0)
    {
      return VV_OK;
    }
    else
    {
      vvDebugMsg::msg(2, "vvSocket::read_data()", true);
      return VV_READ_ERROR;
    }
  }
  else
  {
    return vvSocket::readData(dataptr, size, ret);
  }
}

vvSocket::ErrorType vvUdpSocket::writeData(const uchar* dataptr, size_t size, ssize_t *ret)
{
  if(_mc)
  {
#ifdef WIN32
    ssize_t written = sendto(_sockfd, (const char *)dataptr, (int)size, 0, (struct sockaddr*)&_groupSock, sizeof(_groupSock));
#else
    ssize_t written = sendto(_sockfd, dataptr, size, 0, (struct sockaddr*)&_groupSock, sizeof(_groupSock));
#endif
    if(ret) *ret = written;
    if(written == (ssize_t)size)
    {
      return VV_OK;
    }
    else
    {
      vvDebugMsg::msg(2, "vvSocket::write_data()", true);
      return VV_WRITE_ERROR;
    }
  }
  else
  {
    return vvSocket::writeData(dataptr, size, ret);
  }
}


//----------------------------------------------------------------------------
/**Reads data from the UDP socket.
 @param buffer  pointer to the data to write
 @param size   number of bytes to write

*/
ssize_t vvUdpSocket::readn(char* buffer, size_t size)
{
  size_t nleft;
  ssize_t nread;

  nleft = size;
  while(nleft > 0)
  {
    nread = recv(_sockfd, buffer, (int)nleft, 0);
    retValue = nread;
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
        vvDebugMsg::msg(1, "Error: udp recv()");
        return (ssize_t)-1;
      }
    }
    else if (nread == 0)
      break;
    std::ostringstream errmsg;
    errmsg <<  nread << " Bytes read";
    vvDebugMsg::msg(3, errmsg.str().c_str());
    nleft -= nread;
    buffer += nread;
  }
  return (size - nleft);
}

//----------------------------------------------------------------------------
/**Writes data to the UDP socket.
 @param buffer  pointer to the data to write
 @param size   number of bytes to write

*/
ssize_t vvUdpSocket::writen(const char* buffer, size_t size)
{
  size_t nleft, towrite;
  ssize_t nwritten;

  nleft = size;

  while(nleft > 0)
  {
    if (nleft > (size_t)_maxSendSize)
      towrite = _maxSendSize;
    else
      towrite = nleft;
    nwritten = send(_sockfd, buffer, (int)towrite, 0);
    retValue = nwritten;
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
        vvDebugMsg::msg(1, "Error: udp send()");
        return (ssize_t)-1;
      }
    }
    std::ostringstream errmsg;
    errmsg << nwritten << " Bytes written";
    vvDebugMsg::msg(3, errmsg.str().c_str());
    nleft -= nwritten;
    buffer += nwritten;
  }
  return size;
}

//----------------------------------------------------------------------------
/** Reads a message from the client to get his address for a connected UDP socket.
 Calls recvfrom_timeo() for reading with timeout and recvfrom_nontimeo()
 for reading without a timeout.
*/
vvSocket::ErrorType vvUdpSocket::getClientAddr()
{
  uchar buff;

  if(recvfrom(_sockfd, (char*)&buff, 1, 0,(struct sockaddr *)&_hostAddr, &_hostAddrLen) !=1)
  {
    vvDebugMsg::msg(1, "Error: recvfrom()");
    return VV_READ_ERROR;
  }
  vvDebugMsg::msg(3, "Client Address received");
  return VV_OK;
}

// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
