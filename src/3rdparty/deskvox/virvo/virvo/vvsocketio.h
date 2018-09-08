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

#ifndef VV_SOCKETIO_H
#define VV_SOCKETIO_H

#include <vector>

#include "math/forward.h"

#include "vvcolor.h"
#include "vvexport.h"
#include "vvremoteevents.h"
#include "vvrequestmanagement.h"
#include "vvsocket.h"
#include "vvinttypes.h"
#include "vvtransfunc.h"

struct vvMulticastParameters;
class vvImage;
class vvVolDesc;

namespace virvo
{
  class CompressedVector;
  class IbrImage;
  class Image;
}

/** This class provides specific data transfer through sockets.
  It requires a socket of type vvSocket.<BR>
  Here is an example code fragment for a TCP sever which reads
  a volume from a socket and a TCP client which writes a volume to the
  socket:<BR>
  <PRE>

  Server:

  // Create a new TCP socket class instance, which is a server listening on port 17171
  vvSocket* sock = new vvSocket(17171 , vvSocket::VV_TCP);
  vvVolDesc* vd = new vvVolDesc();

  // Initialize the socket with the parameters and wait for a server
  if (sock->init() != vvSocket::VV_OK)
  {
    delete sock;
    return -1;
  }

  // Assign the socket to socketIO which extens the socket with more functions
  vvSocketIO* sio = new vvSocketIO(sock);

  // Get a volume
  switch (sio->getVolume(vd))
  {
  case vvSocket::VV_OK:
    cerr << "Volume transferred successfully" << endl;
    break;
  case vvSocket::VV_ALLOC_ERROR:
    cerr << "Not enough memory" << endl;
    break;
  default:
    cerr << "Cannot read volume from socket" << endl;
    break;
  }
  delete sock;
  delete sio;

  Client:

  // Create a new TCP socket class instance, which is a client and connects
  // to a server listening on port 17171

  char* servername = "buxdehude";
  vvSocket* sock = new vvSocket(17171 , servername, vvSocket::VV_TCP);
  vvVolDesc* vd = new vvVolDesc();

  // Initialize the socket with the parameters and connect to the server.
  if (sio->init() != vvSocket::VV_OK)
  {
    delete sock;
    return -1;
  }

  // Assign the socket to socketIO
  vvSocketIO* sio = new vvSocketIO(sock);

  // Put a volume
  switch (sio->putVolume(vd))
  {
  case vvSocket::VV_OK:
    cerr << "Volume transferred successfully" << endl;
    break;
  case vvSocket::VV_ALLOC_ERROR:
    cerr << "Not enough memory" << endl;
    break;
  default:
    cerr << "Cannot write volume to socket" << endl;
    break;
  }
  delete sock;
  delete sio;

</PRE>
@see vvSocket
@author Michael Poehnl
*/
class VIRVOEXPORT vvSocketIO
{
  public:
    enum DataType                                 /// data type for get/putData
    {
      VV_UCHAR,
      VV_USHORT,
      VV_INT,
      VV_FLOAT
    };

    vvSocketIO(vvSocket* sock);
    ~vvSocketIO();
    bool sock_action();
    vvSocket::ErrorType getEvent(virvo::RemoteEvent& event) const;
    vvSocket::ErrorType putEvent(virvo::RemoteEvent event) const;
    vvSocket::ErrorType getVolumeAttributes(vvVolDesc* vd) const;
    vvSocket::ErrorType getVolume(vvVolDesc* vd) const;
    vvSocket::ErrorType putVolumeAttributes(const vvVolDesc*) const;
    vvSocket::ErrorType putVolume(const vvVolDesc* vd) const;
    vvSocket::ErrorType getTransferFunction(vvTransFunc& tf) const;
    vvSocket::ErrorType putTransferFunction(vvTransFunc& tf) const;
    vvSocket::ErrorType getImage(vvImage*) const;
    vvSocket::ErrorType putImage(const vvImage*) const;
    vvSocket::ErrorType getFileName(std::string& fn) const;
    vvSocket::ErrorType putFileName(const std::string& fn) const;
    vvSocket::ErrorType allocateAndGetData(uchar**, int&) const;             //  unknown number and type
    vvSocket::ErrorType putData(uchar*, int) const;
    vvSocket::ErrorType getMatrix(virvo::mat4*) const;
    vvSocket::ErrorType putMatrix(virvo::mat4 const*) const;
    vvSocket::ErrorType getBool(bool& val) const;
    vvSocket::ErrorType putBool(const bool val) const;
    vvSocket::ErrorType getInt32(int32_t& val) const;
    vvSocket::ErrorType putInt32(int32_t val) const;
    vvSocket::ErrorType getInt64(int64_t& val) const;
    vvSocket::ErrorType putInt64(int64_t val) const;
    vvSocket::ErrorType getUint32(uint32_t& val) const;
    vvSocket::ErrorType putUint32(uint32_t val) const;
    vvSocket::ErrorType getUint64(uint64_t& val) const;
    vvSocket::ErrorType putUint64(uint64_t val) const;
    vvSocket::ErrorType getFloat(float& val) const;
    vvSocket::ErrorType putFloat(const float val) const;
    vvSocket::ErrorType getVector3(virvo::vec3f& val) const;
    vvSocket::ErrorType putVector3(const virvo::vec3f& val) const;
    vvSocket::ErrorType getVector4(virvo::vec4f& val) const;
    vvSocket::ErrorType putVector4(const virvo::vec4f& val) const;
    vvSocket::ErrorType getColor(vvColor& val) const;
    vvSocket::ErrorType putColor(const vvColor& val) const;
    vvSocket::ErrorType getViewport(virvo::recti& val) const;
    vvSocket::ErrorType putViewport(virvo::recti const& val) const;
    vvSocket::ErrorType getWinDims(int& w, int& h) const;
    vvSocket::ErrorType putWinDims(int w, int h) const;
    vvSocket::ErrorType getData(void*, int, DataType) const;      // known number and type
    vvSocket::ErrorType putData(void*, int, DataType) const;
    vvSocket::ErrorType getRendererType(vvRenderer::RendererType& type) const;
    vvSocket::ErrorType putRendererType(vvRenderer::RendererType type) const;
    vvSocket::ErrorType getServerInfo(vvServerInfo& info) const;
    vvSocket::ErrorType putServerInfo(vvServerInfo info) const;
    vvSocket::ErrorType getGpuInfo(vvGpu::vvGpuInfo& ginfo) const;
    vvSocket::ErrorType putGpuInfo(const vvGpu::vvGpuInfo& ginfo) const;
    vvSocket::ErrorType getGpuInfos(std::vector<vvGpu::vvGpuInfo>& ginfos) const;
    vvSocket::ErrorType putGpuInfos(const std::vector<vvGpu::vvGpuInfo>& ginfos) const;
    vvSocket::ErrorType getRequest(vvRequest& req) const;
    vvSocket::ErrorType putRequest(const vvRequest& req) const;
    vvSocket::ErrorType getStdVector(std::vector<unsigned char>& vec) const;
    vvSocket::ErrorType putStdVector(std::vector<unsigned char> const& vec) const;
    vvSocket::ErrorType getImage(virvo::Image& image) const;
    vvSocket::ErrorType putImage(virvo::Image const& image) const;
    vvSocket::ErrorType getIbrImage(virvo::IbrImage& image) const;
    vvSocket::ErrorType putIbrImage(virvo::IbrImage const& image) const;
    vvSocket::ErrorType getCompressedVector(virvo::CompressedVector& vec) const;
    vvSocket::ErrorType putCompressedVector(virvo::CompressedVector const& vec) const;

    vvSocket* getSocket() const;

    vvSocket *_socket;
};
#endif
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
