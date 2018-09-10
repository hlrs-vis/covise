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

#include <sstream>

#include "vvdebugmsg.h"
#include "vvsocketmonitor.h"
#include "vvtcpserver.h"
#include "vvtcpsocket.h"

#ifndef TEMP_FAILURE_RETRY
#define TEMP_FAILURE_RETRY(x) (x)
#endif

vvTcpServer::vvTcpServer(const ushort port)
  : _listener(NULL)
{
  vvsock_t sockfd;

#ifdef _WIN32
  WSADATA wsaData;
  if (WSAStartup(MAKEWORD(2,0), &wsaData) != 0)
    vvDebugMsg::msg(0, "WSAStartup failed!");
#endif

#ifdef _WIN32
  char optval=1;
#else
  int optval=1;
#endif

  sockfd = socket(AF_INET, SOCK_STREAM, 0);

  if (sockfd == VV_INVALID_SOCKET)
  {
    vvDebugMsg::msg(0, "Error: socket()", true);
    return;
  }

  if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &optval,sizeof(optval)))
  {
    vvDebugMsg::msg(0, "Error: setsockopt()");
    return;
  }

  memset((char *) &_hostAddr, 0, sizeof(_hostAddr));
  _hostAddr.sin_family = AF_INET;
  _hostAddr.sin_port = htons((ushort)port);
  _hostAddr.sin_addr.s_addr = INADDR_ANY;
  _hostAddrlen = sizeof(_hostAddr);
  if(bind(sockfd, (struct sockaddr *)&_hostAddr, _hostAddrlen))
  {
    vvDebugMsg::msg(0, "Error: bind()");
    return;
  }

  if (listen(sockfd, 1))
  {
    vvDebugMsg::msg(0, "Error: listen()");
    return;
  }

  _listener = new vvTcpSocket();
  _listener->setSockfd(sockfd);
}

vvTcpServer::~vvTcpServer()
{
  delete _listener;
}

bool vvTcpServer::initStatus() const
{
  return _listener && (_listener->getSockfd() != VV_INVALID_SOCKET);
}

vvTcpSocket* vvTcpServer::nextConnection(double timeout)
{
  if(!initStatus())
  {
    vvDebugMsg::msg(0, "vvTcpServer::nextConnection() error: server not correctly initialized");
    return NULL;
  }

  if (timeout < 0.0 ? false : true)
  {
    if(vvSocket::VV_OK != _listener->setParameter(vvSocket::VV_NONBLOCKING, 1.f))
    {
      vvDebugMsg::msg(0, "vvTcpServer::nextConnection() error: setting O_NONBLOCK on server-socket failed");
      return NULL;
    }

    std::vector<vvSocket*> socks;
    socks.push_back(_listener);

    vvSocketMonitor sm;
    sm.setReadFds(socks);

    vvSocket* ready;
    sm.wait(&ready, &timeout);

    if(ready == NULL)
      return NULL;
  }
  else
  {
    if(vvSocket::VV_OK != _listener->setParameter(vvSocket::VV_NONBLOCKING, 0.0f))
    {
      vvDebugMsg::msg(0, "vvTcpServer::nextConnection() error: removing O_NONBLOCK from server-socket failed.");
      return NULL;
    }
  }

  vvsock_t n = TEMP_FAILURE_RETRY(accept(_listener->getSockfd(), (struct sockaddr *)&_hostAddr, &_hostAddrlen));

  if (n == VV_INVALID_SOCKET)
  {
    vvDebugMsg::msg(0, "vvTcpServer::nextConnection() error: accept() failed", true);
    return NULL;
  }

  vvTcpSocket *next = new vvTcpSocket();
  next->setSockfd(n);

  std::ostringstream errmsg;
  errmsg << "Incoming connection from " << inet_ntoa(_hostAddr.sin_addr);
  vvDebugMsg::msg(2, errmsg.str().c_str());

  return next;
}

// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
