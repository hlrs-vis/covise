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

#ifndef VV_TCPSERVER_H
#define VV_TCPSERVER_H

#include "vvexport.h"
#include "vvinttypes.h"
#include "vvsocket.h"

class vvTcpSocket;

/**
  This class creates a tcp-server which can listen on a set port for
  incoming connections and provides an appropriate vvTcpSocket object
  for further socket communications
  */
class VIRVOEXPORT vvTcpServer
{
public:
  /**
    Creates a tcp-server for the given port
    @param port the desired port to listen for incoming connections
  */
  vvTcpServer(ushort port);
  ~vvTcpServer();

  /**
    Get status of server.
    @return true if socket is ready to use, false elsewise
    */
  bool initStatus() const;
  /**
    Listen on socket for incomming connections and accept the first
    one. This call will obviously block.
    @param timeout maximum time to wait in seconds
    @return Pointer to an ready to use vvTcpSocket or NULL if errer occured
    */
  vvTcpSocket* nextConnection(double timeout = -1.0);

private:
  vvTcpSocket *_listener;
  struct sockaddr_in _hostAddr;
#if !defined(__linux__) && !defined(LINUX) && !(defined(__APPLE__) && defined(__GNUC__) && GNUC__ < 4)
#define socklen_t int
#endif
  socklen_t _hostAddrlen;

};

#endif
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
