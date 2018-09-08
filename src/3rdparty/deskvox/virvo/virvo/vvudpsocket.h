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

#ifndef VV_UDPSOCKET_H
#define VV_UDPSOCKET_H

#include "vvexport.h"
#include "vvsocket.h"

//----------------------------------------------------------------------------
/** UDP Sockets
*/
class VIRVOEXPORT vvUdpSocket : public vvSocket
{
public:

  enum MulticastType
  {
    VV_MC_SENDER,
    VV_MC_RECEIVER
  };

  vvUdpSocket();
  ~vvUdpSocket();

  /** Binds to a host. If clminport and clmaxport are given and valid, an
    outgoing port within this range is tryed to be established
    \param host hostname or IPV4-address
    \param port port
    \param clminport lower limit port range
    \param clminport upper limit port range
    \returns VV_OK on success and appropriate error value of type vvSocket::ErrorType else
    */
  ErrorType bind(const std::string& hostname, ushort port, int clmin = 0, int clmax = 0);
  /** Binds to a local port for incoming udp connections
    \param port port
    \returns VV_OK on success and appropriate error value of type vvSocket::ErrorType else
    */
  ErrorType bind(ushort port);
  /** Unbinds socket if connected, else does nothing
    \returns VV_OK on success, VV_ERROR else
    */
  ErrorType unbind();

  /** Connect to multicast-address
    \param hostname multicast address. Must be in range from 224.0.0.0 to 239.255.255.255. Watch out for reserved addresses. See IPv4-Specification for details
    \param port port
    \param type type of multicast participant. Must be either VV_MC_SENDER or VV_MC_RECEIVER.
    \returns VV_OK on connection success, appropriate error value else
    */
  ErrorType multicast(const std::string& hostname, ushort port, MulticastType type);

  /** Reimplementation of vvSocket::readData() */
  ErrorType readData (      uchar*, size_t, ssize_t *ret = NULL);
  /** Reimplementation of vvSocket::writeData() */
  ErrorType writeData(const uchar*, size_t, ssize_t *ret = NULL);

private:
  ssize_t readn(char*, size_t);
  ssize_t writen(const char*, size_t);

  ErrorType getClientAddr();

  uint _maxSendSize;
  int retValue;

  // multicasting...
  bool _mc;
  sockaddr_in _localSock;
  ip_mreq _mcGroup;
  sockaddr_in _groupSock;
  in_addr _localInterface;
};

#endif
