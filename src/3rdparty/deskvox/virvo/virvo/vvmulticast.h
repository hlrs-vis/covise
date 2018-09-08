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

#ifndef VV_MULTICAST_H
#define VV_MULTICAST_H

#include "vvexport.h"
#include "vvinttypes.h"
#include "vvudpsocket.h"

#include <string>
#include <vector>

typedef const void* NormInstanceHandle;
typedef const void* NormSessionHandle;
typedef const void* NormObjectHandle;
typedef uint32_t    NormNodeId;

/** Wrapper class for multicasting.
  This class can be used for lossless multicast communication via NormAPI
  or for standard but paket-ordered multicasting with vvSocket
  @author Stavros Delisavas (stavros.delisavas@uni-koeln.de)
 */
class VIRVOEXPORT vvMulticast
{
public:
  enum
  {
    CHUNK_SIZE = 32*1024*1024, ///< maximum size of Norm-Chunks (32mb)
    DGRAM_SIZE = 32 * 1024     ///< maximum size of datagrams used with vvSocket (32kb)
  };
  enum MulticastType
  {
    VV_SENDER,
    VV_RECEIVER
  };
  enum MulticastApi
  {
    VV_NORM,
    VV_VVSOCKET
  };

  /** Constructor creating a sending or receiving multicast-unit
    \param addr Must be an adress within the range of 224.0.0.0 to 239.255.255.255.
    \note       Some addresses are reserved! (See IPv4-documentation for further informations.)
    \param port Desired port number
    \param type Defined by VV_SENDER or VV_RECEIVER
    \param api  Desired use of API, either VV_NORM or VV_VVSOCKET
    */
  vvMulticast(const MulticastType type, const MulticastApi api = VV_VVSOCKET, const char* addr = "224.1.1.1", const ushort port = 50096);
  ~vvMulticast();

  /** send bytes to multicast-adress
    \param bytes   pointer to stored data
    \param size    size of data in bytes
    \param timeout timeout in seconds or negative for no timeout
    */
  ssize_t write(const uchar* bytes, const size_t size, double timeout = -1.0);

  /** read until "size" bytes or timeout is reached
    \param size    expected size of data in bytes
    \param bytes   pointer for data to be written to
    \param timeout timeout in seconds or negative for no timeout
    \return        number of bytes actually read
    */
  ssize_t read(uchar* data, const size_t size, double timeout = -1.0);

  std::vector<NormNodeId> _nodes;

private:
  uchar* numberConsecutively(const uchar* data, const size_t size);
  MulticastType _type;
  MulticastApi  _api;

  // Variables for multicasting with Norm
  NormInstanceHandle _instance;
  NormSessionHandle  _session;
  NormObjectHandle   _object;
  vvUdpSocket*       _normSocket;

  // Variables for multicasting with vvSocket
  vvUdpSocket* _socket;
  union bytesToInt32
  {
    uchar    x[4];
    uint32_t y;
  };
};

struct VIRVOEXPORT vvMulticastParameters
{
  vvMulticast::MulticastType  type;
  vvMulticast::MulticastApi   api;
  char*                       addr;
  ushort                      port;
};

#endif

// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
