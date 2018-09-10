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

#ifndef VV_TCPSOCKET_H
#define VV_TCPSOCKET_H

#include "vvexport.h"
#include "vvsocket.h"

//----------------------------------------------------------------------------
/** This class provides basic socket functionality. It is used for TCP and UDP
    sockets. For example code see documentation about vvSocket  <BR>
*/
class VIRVOEXPORT vvTcpSocket : public vvSocket
{
public:
  vvTcpSocket();
  ~vvTcpSocket();

  /** Connects to a host. If clminport and clmaxport are given and valid, an
    outgoing port within this range is tryed to be established
    \param host hostname or IPV4-address
    \param port port
    \param clminport lower limit port range
    \param clminport upper limit port range
    \returns VV_OK on success and appropriate error value of type vvSocket::ErrorType else
    */
  ErrorType connectToHost(const std::string& host, ushort port, int clminport = 0, int clmaxport = 0);
  /** Disconnects socket if connected, else does nothing
    \returns VV_OK on success, VV_ERROR else
    */
  ErrorType disconnectFromHost();

private:
  ssize_t readn(char*, size_t);
  ssize_t writen(const char*, size_t);
};

#endif
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
