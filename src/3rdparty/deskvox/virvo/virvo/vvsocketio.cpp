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

#include "vvsocketio.h"
#include "vvimage.h"
#include "vvinttypes.h"
#include "vvvoldesc.h"
#include "vvdebugmsg.h"
#include "vvmulticast.h"
#include "vvtoolshed.h"

#include "math/math.h"

#include "private/vvimage.h"
#include "private/vvibrimage.h"
#include "private/vvcompressedvector.h"

//#ifdef VV_DEBUG_MEMORY
//#include <crtdbg.h>
//#define new new(_NORMAL_BLOCK,__FILE__, __LINE__)
//#endif

#include <sstream>
#include <string>

using virvo::mat4;
using virvo::vec3f;
using virvo::vec4f;


//----------------------------------------------------------------------------
/** Constructor
 @param sock ready to use socket of type vvSocket
*/
vvSocketIO::vvSocketIO(vvSocket *sock)
: _socket(sock)
{
}

//----------------------------------------------------------------------------
/// Destructor
vvSocketIO::~vvSocketIO()
{
}

//----------------------------------------------------------------------------
/** Checks if there is data in the socket receive buffer.
 @return  true for data in the socket receive buffer, false for not.
*/
bool vvSocketIO::sock_action()
{
  if(_socket)
  {
    if (_socket->isDataWaiting() > 0)
      return true;
    else
      return false;
  }
  else
    return false;
}

//----------------------------------------------------------------------------
/** Get remote event from sockets.
  @param event  @see virvo::RemoteEvents
*/
vvSocket::ErrorType vvSocketIO::getEvent(virvo::RemoteEvent& event) const
{
  int32_t val;
  vvSocket::ErrorType result = getInt32(val);
  event = static_cast<virvo::RemoteEvent>(val);
  return result;
}

//----------------------------------------------------------------------------
/** Put remote events to socket.
  @param event  @see virvo::RemoteEvents
*/
vvSocket::ErrorType vvSocketIO::putEvent(const virvo::RemoteEvent event) const
{
  return putInt32((int32_t)event);
}

//----------------------------------------------------------------------------
/** Get volume attributes from socket.
  @param vd  empty volume description which is to be filled with the volume attributes
*/
vvSocket::ErrorType vvSocketIO::getVolumeAttributes(vvVolDesc* vd) const
{
  if(_socket)
  {
    vvSocket::ErrorType retval;

    size_t size = vd->serializeAttributes();

    std::vector<uint8_t> buffer(size+4);
    if ((retval =_socket->readData(&buffer[0], size+4)) != vvSocket::VV_OK)
    {
      return retval;
    }
    vvDebugMsg::msg(3, "Header received");
    vd->deserializeAttributesOLD(&buffer[0]);
    vd->_scale = virvo::serialization::readFloat(&buffer[size]);

    return vvSocket::VV_OK;
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}


//----------------------------------------------------------------------------
/** Get volume data from socket.
  @param vd  empty volume description which is to be filled with the volume data
*/
vvSocket::ErrorType vvSocketIO::getVolume(vvVolDesc* vd) const
{
  if (_socket)
  {
    vvSocket::ErrorType retval = getVolumeAttributes(vd);
    if(retval != vvSocket::VV_OK)
      return retval;

    size_t size = vd->getFrameBytes();

    for(size_t k =0; k< vd->frames; k++)
    {
      uint8_t *buffer = new uint8_t[size];
      if (!buffer)
        return vvSocket::VV_ALLOC_ERROR;
      if ((retval =_socket->readData(buffer, size)) != vvSocket::VV_OK)
      {
        delete[] buffer;
        return retval;
      }
      vd->addFrame(buffer, vvVolDesc::ARRAY_DELETE);
    }
    vvDebugMsg::msg(3, "Data received");
    return vvSocket::VV_OK;
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

//----------------------------------------------------------------------------
/** Write volume attributes to socket.
  @param vd  volume description of volume to be send.
*/
vvSocket::ErrorType vvSocketIO::putVolumeAttributes(const vvVolDesc* vd) const
{
  if(_socket)
  {
    size_t size = vd->serializeAttributes();
    std::vector<uint8_t> buffer(size+4);
    vd->serializeAttributes(&buffer[0]);
    virvo::serialization::writeFloat(&buffer[size], vd->_scale);
    vvDebugMsg::msg(3, "Sending header ...");
    return _socket->writeData(&buffer[0], size+4);
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

//----------------------------------------------------------------------------
/** Write volume data to socket.
  @param vd  volume description of volume to be send.
*/
vvSocket::ErrorType vvSocketIO::putVolume(const vvVolDesc* vd) const
{
  if(_socket)
  {
    vvSocket::ErrorType retval = putVolumeAttributes(vd);
    if(retval != vvSocket::VV_OK)
      return retval;

    size_t frames = vd->frames;

    size_t size = vd->getFrameBytes();
    vvDebugMsg::msg(3, "Sending data ...");

    for(size_t k=0; k < frames; k++)
    {
      const uint8_t *buffer = vd->getRaw(k);
      if ((retval =_socket->writeData(buffer, size)) != vvSocket::VV_OK)
      {
        return retval;
      }
    }
    return vvSocket::VV_OK;
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

//----------------------------------------------------------------------------
/** Get a transfer function from the socket.
  @param tf  pointer to a vvTransFunc.
*/
vvSocket::ErrorType vvSocketIO::getTransferFunction(vvTransFunc& tf) const
{
  if(_socket)
  {
    uchar* buffer = NULL;
    vvSocket::ErrorType retval;
    int len;

    if ((retval = getInt32(len)) != vvSocket::VV_OK)
    {
      delete[] buffer;
      return retval;
    }

    buffer = new uchar[len+1];
    if ((retval =_socket->readData(buffer, len)) != vvSocket::VV_OK)
    {
      delete[] buffer;
      return retval;
    }
    buffer[len] = '\0';

    std::istringstream in;
    in.str((const char*)buffer);
  #ifdef WIN32
    char cline[65535];
  #else
    char cline[vvTFWidget::MAX_STR_LEN];
  #endif
    while (in.getline(cline, vvTFWidget::MAX_STR_LEN))
    {
      std::string line = std::string(cline);

      // Skip over erroneous lines.
      if (line.length() < 3)
      {
        continue;
      }

      std::vector<std::string> tokens = vvToolshed::split(line, " ");

      // At least widget type and name.
      if (tokens.size() < 2)
      {
        continue;
      }
      const char* name = tokens[0].c_str();

      vvTFWidget* widget = vvTFWidget::produce(vvTFWidget::getWidgetType(name));

      if (widget)
      {
        widget->fromString(line);
        tf._widgets.push_back(widget);
      }
    }

    delete[] buffer;
    return vvSocket::VV_OK;
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

//----------------------------------------------------------------------------
/** Write a transfer function to the socket.
  @param tf  pointer to a vvTransFunc.
*/
vvSocket::ErrorType vvSocketIO::putTransferFunction(vvTransFunc& tf) const
{
  if(_socket)
  {
    uchar* buffer = NULL;
    vvSocket::ErrorType retval;

    std::ostringstream out;

    for (std::vector<vvTFWidget*>::const_iterator it = tf._widgets.begin();
         it != tf._widgets.end(); ++it)
    {
      out << (*it)->toString();
    }

    const size_t len = strlen(out.str().c_str());
    buffer = new uchar[len+1];
    strcpy((char*)buffer, out.str().c_str());

    putInt32((int)len);

    if ((retval =_socket->writeData(buffer, len)) != vvSocket::VV_OK)
    {
      delete[] buffer;
      return retval;
    }

    delete[] buffer;
    return vvSocket::VV_OK;
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

//----------------------------------------------------------------------------
/** Get an image from the socket.
 @param im  pointer to a vvImage object.
*/
vvSocket::ErrorType vvSocketIO::getImage(vvImage* im) const
{
  if(_socket)
  {
    const size_t BUFSIZE = 13;
    uchar buffer[BUFSIZE];
    vvSocket::ErrorType retval;
    short w, h;
    int imagesize;
    int videosize;

    if ((retval =_socket->readData(&buffer[0], BUFSIZE)) != vvSocket::VV_OK)
    {
      return retval;
    }
    vvDebugMsg::msg(3, "Header received");
    w = virvo::serialization::read16(&buffer[2]);
    h = virvo::serialization::read16(&buffer[0]);

    vvImage::CodeType ct = (vvImage::CodeType)virvo::serialization::read8(&buffer[4]);
    if (h != im->getHeight() || w  != im->getWidth() || ct != im->getCodeType() )
    {
      im->setCodeType(ct);
      im->setHeight(h);
      im->setWidth(w);
      if(im->alloc_mem())
        return vvSocket::VV_ALLOC_ERROR;
    }
    imagesize = (int)virvo::serialization::read32(&buffer[5]);
    videosize = (int)virvo::serialization::read32(&buffer[9]);
    im->setSize(imagesize);
    im->setVideoSize(videosize);
    if (vvDebugMsg::isActive(3))
      fprintf(stderr, "imgsize=%d, videosize=%d\n", imagesize, videosize);
    if ((retval =_socket->readData(im->getCodedImage(), imagesize)) != vvSocket::VV_OK)
    {
      return retval;
    }
    vvDebugMsg::msg(3, "Image data received");
    if (ct == vvImage::VV_VIDEO)
    {
      if ((retval =_socket->readData(im->getVideoCodedImage(), videosize)) != vvSocket::VV_OK)
      {
        return retval;
      }
      vvDebugMsg::msg(3, "Video Image data received");
    }
    return vvSocket::VV_OK;
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

//----------------------------------------------------------------------------
/** Write an image to the socket.
 @param im  pointer to an vvImage object.
*/
vvSocket::ErrorType vvSocketIO::putImage(const vvImage* im) const
{
  if(_socket)
  {
    const int BUFSIZE = 13;
    uchar buffer[BUFSIZE];
    vvSocket::ErrorType retval;
    int imagesize;
    int videosize;
    int ct;
    imagesize = im->getSize();
    videosize = im->getVideoSize();
    ct = im->getCodeType();
    virvo::serialization::write16(&buffer[0], im->getHeight());
    virvo::serialization::write16(&buffer[2], im->getWidth());
    virvo::serialization::write8(&buffer[4], (uchar)ct);
    virvo::serialization::write32(&buffer[5], (ulong)imagesize);
    virvo::serialization::write32(&buffer[9], (ulong)videosize);

    vvDebugMsg::msg(3, "Sending header ...");
    if ((retval =_socket->writeData(&buffer[0], BUFSIZE)) != vvSocket::VV_OK)
    {
      return retval;
    }
    vvDebugMsg::msg(3, "Sending image data ...");
    if ((retval =_socket->writeData(im->getImagePtr(), imagesize)) != vvSocket::VV_OK)
    {
      return retval;
    }
    if (ct == vvImage::VV_VIDEO)
    {
      vvDebugMsg::msg(3, "Sending video image data ...");
      if ((retval =_socket->writeData(im->getVideoCodedImage(), videosize)) != vvSocket::VV_OK)
      {
        return retval;
      }
    }
    return vvSocket::VV_OK;
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

//----------------------------------------------------------------------------
/** Get a file name from the socket.
 @param fn  the file name.
*/
vvSocket::ErrorType vvSocketIO::getFileName(std::string& fn) const
{
  if (_socket != NULL)
  {
    vvSocket::ErrorType retval;

    uchar sizebuf[4];
    if ((retval =_socket->readData(sizebuf, 4)) != vvSocket::VV_OK)
    {
      return retval;
    }
    size_t len = virvo::serialization::read32(sizebuf);

    std::vector<uchar> buf(len);
    if ((retval =_socket->readData(&buf[0], len)) != vvSocket::VV_OK)
    {
      return retval;
    }

    fn = std::string((char*)&buf[0], len);

    return vvSocket::VV_OK;
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

//----------------------------------------------------------------------------
/** Write a file name to the socket.
 @param fn  the file name.
*/
vvSocket::ErrorType vvSocketIO::putFileName(const std::string& fn) const
{
  if (_socket != NULL)
  {
    vvSocket::ErrorType retval;

    uchar sizebuf[4];
    virvo::serialization::write32(sizebuf, fn.length());

    if ((retval = _socket->writeData(sizebuf, 4)) != vvSocket::VV_OK)
    {
      return retval;
    }

    uchar* buf = (uchar*)fn.data();
    return _socket->writeData(buf, fn.length());
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

//----------------------------------------------------------------------------
/** Gets arbitrary data of arbitrary size from the socket.
 @param data  pointer to the pointer where data shall be written. Memory is
 allocated which has to be deallocated outside this function.
 @param size  reference of an integer which includes the number of read bytes.
*/
vvSocket::ErrorType vvSocketIO::allocateAndGetData(uchar** data, int& size) const
{
  if(_socket)
  {
    uchar buffer[4];
    vvSocket::ErrorType retval;

    *data = NULL; // make it safe to delete[] *data

    if ((retval =_socket->readData(&buffer[0], 4)) != vvSocket::VV_OK)
    {
      return retval;
    }
    vvDebugMsg::msg(3, "Header received");
    size = (int)virvo::serialization::read32(&buffer[0]);
    *data = new uchar[size];                        // delete buffer outside!!!
    if ((retval =_socket->readData(*data, size)) != vvSocket::VV_OK)
    {
      return retval;
    }
    vvDebugMsg::msg(3, "Data received");
    return vvSocket::VV_OK;
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

//----------------------------------------------------------------------------
/** Writes data to a socket.
 @param data  pointer to the data which has to be written.
 @param size  number of bytes to write.
*/
vvSocket::ErrorType vvSocketIO::putData(uchar* data, int size) const
{
  if(_socket)
  {
    uchar buffer[4];
    vvSocket::ErrorType retval;

    virvo::serialization::write32(&buffer[0], (ulong)size);
    vvDebugMsg::msg(3, "Sending header ...");
    if ((retval =_socket->writeData(&buffer[0], 4)) != vvSocket::VV_OK)
    {
      return retval;
    }
    vvDebugMsg::msg(3, "Sending data ...");
    if ((retval =_socket->writeData(data, size)) != vvSocket::VV_OK)
    {
      return retval;
    }
    return vvSocket::VV_OK;
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

//----------------------------------------------------------------------------
/** Gets a fixed number of elements of a fixed type from the socket.
 @param data  pointer to where data shall be written.
 @param number  number of elements to read.
 @param type  data type to read. vvSocketIO::UCHAR for unsigned char,
 vvSocketIO::INT for integer and vvSocketIO::FLOAT for float.

*/
vvSocket::ErrorType vvSocketIO::getData(void* data, int number, DataType type) const
{
  if(_socket)
  {
    vvSocket::ErrorType retval;
    size_t size;
    uint8_t* buffer;

    switch(type)
    {
      case VV_UCHAR:
      {
        size = number;
        if ((retval =_socket->readData((uint8_t*)data, size)) != vvSocket::VV_OK)
        {
          return retval;
        }
        vvDebugMsg::msg(3, "uint8_t received");
      }break;
      case VV_USHORT:
      {
        int tmp;
        size = number*2;
        buffer = new uint8_t[size];
        if ((retval =_socket->readData(buffer, size)) != vvSocket::VV_OK)
        {
          delete[] buffer;
          return retval;
        }
        for (int i=0; i<number; i++)
        {
          tmp = virvo::serialization::read16(&buffer[i*2]);
          memcpy((uint8_t*)data+i*2, &tmp, 2);
        }
        vvDebugMsg::msg(3, "ushort received");
        delete[] buffer;
      }break;
      case VV_INT:
      {
        int tmp;
        size = number*4;
        buffer = new uint8_t[size];
        if ((retval =_socket->readData(buffer, size)) != vvSocket::VV_OK)
        {
          delete[] buffer;
          return retval;
        }
        for (int i=0; i<number; i++)
        {
          tmp = virvo::serialization::read32(&buffer[i*4]);
          memcpy((uint8_t*)data+i*4, &tmp, 4);
        }
        vvDebugMsg::msg(3, "int received");
        delete[] buffer;
      }break;
      case VV_FLOAT:
      {
        float tmp;
        size = number*4;
        buffer = new uint8_t[size];
        if ((retval =_socket->readData(buffer, size)) != vvSocket::VV_OK)
        {
          delete[] buffer;
          return retval;
        }
        for (int i=0; i<number; i++)
        {
          tmp = virvo::serialization::readFloat(&buffer[i*4]);
          memcpy((uchar*)data+i*4, &tmp, 4);
        }
        vvDebugMsg::msg(3, "float received");
        delete[] buffer;
      }break;
      default:
        vvDebugMsg::msg(0, "No supported data type");
        return vvSocket::VV_DATA_ERROR;
    }
    return vvSocket::VV_OK;
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

//----------------------------------------------------------------------------
/** Write a number of fixed elements to a socket.
    @param data  pointer to the data to write.
    @param number  number of elements to write.
    @param type  data type to write. vvSocketIO::UCHAR for unsigned char,
    vvSocketIO::INT for integer and vvSocketIO::FLOAT for float.
*/
vvSocket::ErrorType vvSocketIO::putData(void* data, int number, DataType type) const
{
  if(_socket)
  {
    vvSocket::ErrorType retval;
    int size;
    uchar* buffer;

    switch(type)
    {
      case VV_UCHAR:
      {
        size = number;
        buffer = (uchar*)data;
        vvDebugMsg::msg(3, "Sending uchar ...");
      }break;
      case (VV_USHORT):
      {
        int tmp;
        size = number*2;
        buffer = new uchar[size];

        for (int i=0; i<number; i++)
        {
          memcpy(&tmp, (uchar*)data+i*2 , 2);
          virvo::serialization::write16(&buffer[i*2], (ushort)tmp);
        }
        vvDebugMsg::msg(3, "Sending ushort ...");
      }break;
      case (VV_INT):
      {
        int tmp;
        size = number*4;
        buffer = new uchar[size];

        for (int i=0; i<number; i++)
        {
          memcpy(&tmp, (uchar*)data+i*4 , 4);
          virvo::serialization::write32(&buffer[i*4], (ulong)tmp);
        }
        vvDebugMsg::msg(3, "Sending integer ...");
      }break;
      case VV_FLOAT:
      {
        float tmp;
        size = number*4;
        buffer = new uchar[size];
        for (int i=0; i<number; i++)
        {
          memcpy(&tmp, (uchar*)data+i*4 , 4);
          virvo::serialization::writeFloat(&buffer[i*4], (float)tmp);
        }
        vvDebugMsg::msg(3, "Sending float ...");
      }break;
      default:
        vvDebugMsg::msg(0, "No supported data type");
        return vvSocket::VV_DATA_ERROR;
    }
    if ((retval =_socket->writeData(buffer, size)) != vvSocket::VV_OK)
    {
      if (type != VV_UCHAR)
        delete[] buffer;
      return retval;
    }
    if (type != VV_UCHAR)
      delete[] buffer;
    return vvSocket::VV_OK;
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

//----------------------------------------------------------------------------
/** Gets a Matrix from the socket.
    @param m  pointer to an object of mat4
*/
vvSocket::ErrorType vvSocketIO::getMatrix(mat4* m) const
{
  if(_socket)
  {
    uchar* buffer = NULL;
    int s;

    switch(allocateAndGetData(&buffer, s))
    {
      case vvSocket::VV_OK: break;
      case vvSocket::VV_DATA_ERROR: delete[] buffer; return vvSocket::VV_DATA_ERROR; break;
      default: delete[] buffer; return vvSocket::VV_DATA_ERROR;
    }
    for (int i=0; i<4; i++)
      for (int j=0; j<4; j++)
        (*m)(i, j) = virvo::serialization::readFloat(buffer+4*(4*i+j));
    delete[] buffer;
    return vvSocket::VV_OK;
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

//----------------------------------------------------------------------------
/** Writes a boolean flag to the socket.
 @param val  the boolean flag.
*/
vvSocket::ErrorType vvSocketIO::putBool(const bool val) const
{
  if(_socket)
  {
    uchar buffer[] = { (uchar)val };
    return _socket->writeData(&buffer[0], 1);
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

//----------------------------------------------------------------------------
/** Reads a boolean flag from the socket.
 @param val  the boolean flag.
*/
vvSocket::ErrorType vvSocketIO::getBool(bool& val) const
{
  if(_socket)
  {
    uchar buffer[1];
    vvSocket::ErrorType retval;

    if ((retval =_socket->readData(&buffer[0], 1)) != vvSocket::VV_OK)
    {
      return retval;
    }
    val = (buffer[0] != 0);

    return vvSocket::VV_OK;
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

//----------------------------------------------------------------------------
/** Writes an int value to the socket.
 @param val  the int value.
*/
vvSocket::ErrorType vvSocketIO::putInt32(int32_t val) const
{
  if(_socket)
  {
    uchar buffer[4];
    virvo::serialization::write32(&buffer[0], val);
    return _socket->writeData(&buffer[0], 4);
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

//----------------------------------------------------------------------------
/** Reads an int value from the socket.
 @param val  the int value.
*/
vvSocket::ErrorType vvSocketIO::getInt32(int32_t& val) const
{
  if(_socket)
  {
    uchar buffer[4];
    vvSocket::ErrorType retval;

    if ((retval =_socket->readData(&buffer[0], 4)) != vvSocket::VV_OK)
    {
      return retval;
    }

    val = virvo::serialization::read32(&buffer[0]);
    return retval;
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

vvSocket::ErrorType vvSocketIO::putInt64(int64_t val) const
{
  if (_socket != NULL)
  {
    uint8_t buffer[8];
    virvo::serialization::write64(&buffer[0], val);
    return _socket->writeData(&buffer[0], 8);
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

vvSocket::ErrorType vvSocketIO::getInt64(int64_t& val) const
{
  if (_socket != NULL)
  {
    uint8_t buffer[8];
    vvSocket::ErrorType retval;

    if ((retval = _socket->readData(&buffer[0], 8)) != vvSocket::VV_OK)
    {
      return retval;
    }

    val = virvo::serialization::read64(&buffer[0]);
    return retval;
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

vvSocket::ErrorType vvSocketIO::putUint32(uint32_t val) const
{
  return putInt32(static_cast<int32_t>(val));
}

vvSocket::ErrorType vvSocketIO::getUint32(uint32_t& val) const
{
  int32_t tmp;
  vvSocket::ErrorType err = getInt32(tmp);
  if (err == vvSocket::VV_OK)
  {
    val = static_cast<uint32_t>(tmp);
  }
  return err;
}

vvSocket::ErrorType vvSocketIO::putUint64(uint64_t val) const
{
  return putInt64(static_cast<uint64_t>(val));
}

vvSocket::ErrorType vvSocketIO::getUint64(uint64_t& val) const
{
  int64_t tmp;
  vvSocket::ErrorType err = getInt64(tmp);
  if (err == vvSocket::VV_OK)
  {
    val = static_cast<uint64_t>(tmp);
  }
  return err;
}

//----------------------------------------------------------------------------
/** Writes a float value to the socket.
 @param val  the float value.
*/
vvSocket::ErrorType vvSocketIO::putFloat(const float val) const
{
  if(_socket)
  {
    uchar buffer[4];
    virvo::serialization::writeFloat(&buffer[0], val);
    return _socket->writeData(&buffer[0], 4);
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

//----------------------------------------------------------------------------
/** Reads a float value from the socket.
 @param val  the float value.
*/
vvSocket::ErrorType vvSocketIO::getFloat(float& val) const
{
  if(_socket)
  {
    uchar buffer[4];
    vvSocket::ErrorType retval;

    if ((retval =_socket->readData(&buffer[0], 4)) != vvSocket::VV_OK)
    {
      return retval;
    }

    val = virvo::serialization::readFloat(&buffer[0]);
    return retval;
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}


vvSocket::ErrorType vvSocketIO::putVector3(vec3f const& val) const
{
  if(_socket)
  {
    uchar buffer[12];
    virvo::serialization::writeFloat(&buffer[0], val[0]);
    virvo::serialization::writeFloat(&buffer[4], val[1]);
    virvo::serialization::writeFloat(&buffer[8], val[2]);
    return _socket->writeData(&buffer[0], 12);
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}


vvSocket::ErrorType vvSocketIO::getVector3(vec3f& val) const
{
  if(_socket)
  {
    uchar buffer[12];
    vvSocket::ErrorType retval;

    if ((retval =_socket->readData(&buffer[0], 12)) != vvSocket::VV_OK)
    {
      return retval;
    }

    val[0] = virvo::serialization::readFloat(&buffer[0]);
    val[1] = virvo::serialization::readFloat(&buffer[4]);
    val[2] = virvo::serialization::readFloat(&buffer[8]);
    return retval;
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}


vvSocket::ErrorType vvSocketIO::putVector4(vec4f const& val) const
{
  if(_socket)
  {
    uchar buffer[16];
    virvo::serialization::writeFloat(&buffer[0], val[0]);
    virvo::serialization::writeFloat(&buffer[4], val[1]);
    virvo::serialization::writeFloat(&buffer[8], val[2]);
    virvo::serialization::writeFloat(&buffer[12], val[3]);
    return _socket->writeData(&buffer[0], 16);
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}


vvSocket::ErrorType vvSocketIO::getVector4(vec4f& val) const
{
  if(_socket)
  {
    uchar buffer[16];
    vvSocket::ErrorType retval;

    if ((retval =_socket->readData(&buffer[0], 16)) != vvSocket::VV_OK)
    {
      return retval;
    }

    val[0] = virvo::serialization::readFloat(&buffer[0]);
    val[1] = virvo::serialization::readFloat(&buffer[4]);
    val[2] = virvo::serialization::readFloat(&buffer[8]);
    val[3] = virvo::serialization::readFloat(&buffer[12]);
    return retval;
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

vvSocket::ErrorType vvSocketIO::putColor(const vvColor& val) const
{
  return putVector3(vec3f(val[0], val[1], val[2]));
}

vvSocket::ErrorType vvSocketIO::getColor(vvColor& val) const
{
  vec3f clr;
  vvSocket::ErrorType err = getVector3(clr);

  if (err == vvSocket::VV_OK)
  {
    for (size_t i = 0; i < 3; ++i)
    {
      val[i] = clr[i];
    }
  }

  return err;
}

//----------------------------------------------------------------------------
/** Writes a virvo::recti to the socket.
 @param val  the virvo::recti.
*/
vvSocket::ErrorType vvSocketIO::putViewport(virvo::recti const& val) const
{
  if(_socket)
  {
    uchar buffer[16];
    virvo::serialization::write32(&buffer[0], val[0]);
    virvo::serialization::write32(&buffer[4], val[1]);
    virvo::serialization::write32(&buffer[8], val[2]);
    virvo::serialization::write32(&buffer[12], val[3]);
    return _socket->writeData(&buffer[0], 16);
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

//----------------------------------------------------------------------------
/** Reads a virvo::recti from the socket.
 @param val  the virvo::recti.
*/
vvSocket::ErrorType vvSocketIO::getViewport(virvo::recti& val) const
{
  if(_socket)
  {
    uchar buffer[16];
    vvSocket::ErrorType retval;

    if ((retval =_socket->readData(&buffer[0], 16)) != vvSocket::VV_OK)
    {
      return retval;
    }

    val[0] = virvo::serialization::read32(&buffer[0]);
    val[1] = virvo::serialization::read32(&buffer[4]);
    val[2] = virvo::serialization::read32(&buffer[8]);
    val[3] = virvo::serialization::read32(&buffer[12]);
    return retval;
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

vvSocket::ErrorType vvSocketIO::putWinDims(const int w, const int h) const
{
  if(_socket)
  {
    uchar buffer[8];

    virvo::serialization::write32(&buffer[0], w);
    virvo::serialization::write32(&buffer[4], h);

    return _socket->writeData(&buffer[0], 8);
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

vvSocket::ErrorType vvSocketIO::getWinDims(int& w, int& h) const
{
  if(_socket)
  {
    uchar buffer[8];
    vvSocket::ErrorType retval;

    if ((retval =_socket->readData(&buffer[0], 8)) != vvSocket::VV_OK)
    {
      return  retval;
    }
    w = virvo::serialization::read32(&buffer[0]);
    h = virvo::serialization::read32(&buffer[4]);

    return vvSocket::VV_OK;
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

//----------------------------------------------------------------------------
/** Writes a Matrix to the socket.
 @param m  pointer to the matrix to write, has to be an object of mat4.
*/
vvSocket::ErrorType vvSocketIO::putMatrix(mat4 const* m) const
{
  if(_socket)
  {
    uchar buffer[64];

    for (int i=0; i<4; i++)
      for (int j=0; j<4; j++)
        virvo::serialization::writeFloat(&buffer[4*(4*i+j)], (*m)(i, j));
    return putData(buffer, 64);
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

vvSocket::ErrorType vvSocketIO::getRendererType(vvRenderer::RendererType& type) const
{
  int32_t val;
  vvSocket::ErrorType result = getInt32(val);
  type = static_cast<vvRenderer::RendererType>(val);
  return result;
}

vvSocket::ErrorType vvSocketIO::putRendererType(const vvRenderer::RendererType type) const
{
  return putInt32((int32_t)type);
}

vvSocket::ErrorType vvSocketIO::getServerInfo(vvServerInfo& info) const
{
  if (_socket != NULL)
  {
    vvSocket::ErrorType retval;

    uchar sizebuf[4];
    if ((retval =_socket->readData(sizebuf, 4)) != vvSocket::VV_OK)
    {
      return retval;
    }
    size_t len = virvo::serialization::read32(sizebuf);

    std::vector<uchar> buf(len);
    if ((retval =_socket->readData(&buf[0], len)) != vvSocket::VV_OK)
    {
      return retval;
    }

    info.renderers = std::string((char*)&buf[0], len);

    return vvSocket::VV_OK;
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

vvSocket::ErrorType vvSocketIO::putServerInfo(vvServerInfo info) const
{
  if (_socket != NULL)
  {
    vvSocket::ErrorType retval;

    uchar sizebuf[4];
    virvo::serialization::write32(sizebuf, info.renderers.length());

    if ((retval = _socket->writeData(sizebuf, 4)) != vvSocket::VV_OK)
    {
      return retval;
    }

    uchar* buf = (uchar*)info.renderers.data();
    return _socket->writeData(buf, info.renderers.length());
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

//----------------------------------------------------------------------------
/** Reads a vvGpuInfo from the socket.
  @param ginfo object in which read data will be saved
*/
vvSocket::ErrorType vvSocketIO::getGpuInfo(vvGpu::vvGpuInfo& ginfo) const
{
  if(_socket)
  {
    uchar buffer[8];
    vvSocket::ErrorType retval;

    if ((retval =_socket->readData(&buffer[0], 8)) == vvSocket::VV_OK)
    {
      ginfo.freeMem  = virvo::serialization::read32(&buffer[0]);
      ginfo.totalMem = virvo::serialization::read32(&buffer[4]);

      return vvSocket::VV_OK;
    }
    else
    {
      return retval;
    }
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

//----------------------------------------------------------------------------
/** Writes a vvGpuInfo to the socket.
  @param ginfo object which will written to socket
*/
vvSocket::ErrorType vvSocketIO::putGpuInfo(const vvGpu::vvGpuInfo& ginfo) const
{
  if(_socket)
  {
    uchar buffer[8];
    virvo::serialization::write32(&buffer[0], ginfo.freeMem);
    virvo::serialization::write32(&buffer[4], ginfo.totalMem);
    return _socket->writeData(&buffer[0], 8);
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
} 
//----------------------------------------------------------------------------
/** Reads a vector list of vvGpuInfos from the socket.
  @param ginfos vector in which read data will be saved
*/
vvSocket::ErrorType vvSocketIO::getGpuInfos(std::vector<vvGpu::vvGpuInfo>& ginfos) const
{
  if(_socket)
  {
    vvSocket::ErrorType retval;

    int size = 0;
    retval = getInt32(size);
    if(retval != vvSocket::VV_OK) return retval;
    else if(size < 0)
    {
      vvDebugMsg::msg(2, "vvSocketIO::getGpuInfos() error - received negative vector size: ", size);
      return vvSocket::VV_DATA_ERROR;
    }

    for(int i=0; i<size; i++)
    {
      vvGpu::vvGpuInfo ginfo;
      retval = getGpuInfo(ginfo);
      ginfos.push_back(ginfo);
      if(retval != vvSocket::VV_OK) return retval;
    }
    return vvSocket::VV_OK;
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

//----------------------------------------------------------------------------
/** Writes a vector list of vvGpuInfos to the socket.
  @param ginfos vector of vvGpuInfos which will be written to socket
*/
vvSocket::ErrorType vvSocketIO::putGpuInfos(const std::vector<vvGpu::vvGpuInfo>& ginfos) const
{
  if(_socket)
  {
    vvSocket::ErrorType retval;

    retval = putInt32(ginfos.size());
    if(retval != vvSocket::VV_OK) return retval;

    for(std::vector<vvGpu::vvGpuInfo>::const_iterator ginfo = ginfos.begin();ginfo != ginfos.end(); ginfo++)
    {
      retval = putGpuInfo(*ginfo);
      if(retval != vvSocket::VV_OK) return retval;
    }
    return vvSocket::VV_OK;
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

//----------------------------------------------------------------------------
/** Reads a vvRequest from the socket.
  @param req request objecto to which read data will be saved
*/
vvSocket::ErrorType vvSocketIO::getRequest(vvRequest& req) const
{
  if(_socket)
  {
    vvSocket::ErrorType retval;

    retval = getInt32(req.niceness);
    if(retval != vvSocket::VV_OK) return retval;

    int type;
    retval = getInt32(type);
    if(retval != vvSocket::VV_OK) return retval;
    req.type = (vvRenderer::RendererType)type;

    int numnodes = 0;
    retval = getInt32(numnodes);
    if(retval != vvSocket::VV_OK) return retval;

    req.nodes.clear();
    for(int i=0;i<numnodes;i++)
    {
      int numgpus = 0;
      retval = getInt32(numgpus);
      if(retval != vvSocket::VV_OK) return retval;

      req.nodes.push_back(numgpus);
    }
    return vvSocket::VV_OK;
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

//----------------------------------------------------------------------------
/** Writes a vvRequest to the socket.
  @param req vvRequest which will be written to socket
*/
vvSocket::ErrorType vvSocketIO::putRequest(const vvRequest& req) const
{
  if(_socket)
  {
    vvSocket::ErrorType retval;

    retval = putInt32((int32_t)req.niceness);
    if(retval != vvSocket::VV_OK) return retval;

    retval = putInt32((int32_t)req.type);
    if(retval != vvSocket::VV_OK) return retval;

    retval = putInt32((int32_t)req.nodes.size());
    if(retval != vvSocket::VV_OK) return retval;

    for(unsigned int i=0; i<req.nodes.size(); i++)
    {
      retval = putInt32((int32_t)req.nodes[i]);
      if(retval != vvSocket::VV_OK) return retval;
    }
    return vvSocket::VV_OK;
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

vvSocket::ErrorType vvSocketIO::getStdVector(std::vector<unsigned char>& vec) const
{
  if (!getSocket())
    return vvSocket::VV_SOCK_ERROR;

  int size = 0;

  vvSocket::ErrorType err = getInt32(size);

  if (err == vvSocket::VV_OK)
  {
    vec.resize(static_cast<size_t>(size));
    err = getSocket()->readData(&vec[0], vec.size());
  }

  return err;
}

vvSocket::ErrorType vvSocketIO::putStdVector(std::vector<unsigned char> const& vec) const
{
  if (!getSocket())
    return vvSocket::VV_SOCK_ERROR;

  if (vec.size() > static_cast<size_t>(0x7FFFFFFF))
    return vvSocket::VV_DATA_ERROR;

  vvSocket::ErrorType err = putInt32(static_cast<int>(vec.size()));

  if (err == vvSocket::VV_OK)
    err = getSocket()->writeData(&vec[0], vec.size());

  return err;
}

vvSocket::ErrorType vvSocketIO::getImage(virvo::Image& image) const
{
  if (!getSocket())
    return vvSocket::VV_SOCK_ERROR;

  unsigned char header[16]; // 4 int's

  vvSocket::ErrorType err = getSocket()->readData(header, sizeof(header));

  if (err == vvSocket::VV_OK)
  {
    int w      = virvo::serialization::read32(&header[ 0]);
    int h      = virvo::serialization::read32(&header[ 4]);
    int format = virvo::serialization::read32(&header[ 8]);
    int stride = virvo::serialization::read32(&header[12]);

    image.init(w, h, static_cast<virvo::PixelFormat>(format), stride);

    err = getCompressedVector(image.data_);
  }

  return err;
}

vvSocket::ErrorType vvSocketIO::putImage(virvo::Image const& image) const
{
  if (!getSocket())
    return vvSocket::VV_SOCK_ERROR;

  unsigned char header[16]; // 4 int's

  virvo::serialization::write32(&header[ 0], image.width());
  virvo::serialization::write32(&header[ 4], image.height());
  virvo::serialization::write32(&header[ 8], image.format());
  virvo::serialization::write32(&header[12], image.stride());

  vvSocket::ErrorType err = getSocket()->writeData(header, sizeof(header));

  if (err == vvSocket::VV_OK)
    err = putCompressedVector(image.data_);

  return err;
}

vvSocket::ErrorType vvSocketIO::getIbrImage(virvo::IbrImage& image) const
{
  if (!getSocket())
    return vvSocket::VV_SOCK_ERROR;

  vvSocket::ErrorType err = vvSocket::VV_OK;

  if (err == vvSocket::VV_OK) err = getImage(image.color_);
  if (err == vvSocket::VV_OK) err = getImage(image.depth_);
  if (err == vvSocket::VV_OK) err = getFloat(image.depthMin_);
  if (err == vvSocket::VV_OK) err = getFloat(image.depthMax_);
  if (err == vvSocket::VV_OK) err = getMatrix(&image.viewMatrix_);
  if (err == vvSocket::VV_OK) err = getMatrix(&image.projMatrix_);
  if (err == vvSocket::VV_OK) err = getViewport(image.viewport_);

  return err;
}

vvSocket::ErrorType vvSocketIO::putIbrImage(virvo::IbrImage const& image) const
{
  if (!getSocket())
    return vvSocket::VV_SOCK_ERROR;

  vvSocket::ErrorType err = vvSocket::VV_OK;

  if (err == vvSocket::VV_OK) err = putImage(image.color_);
  if (err == vvSocket::VV_OK) err = putImage(image.depth_);
  if (err == vvSocket::VV_OK) err = putFloat(image.depthMin_);
  if (err == vvSocket::VV_OK) err = putFloat(image.depthMax_);
  if (err == vvSocket::VV_OK) err = putMatrix(&image.viewMatrix_);
  if (err == vvSocket::VV_OK) err = putMatrix(&image.projMatrix_);
  if (err == vvSocket::VV_OK) err = putViewport(image.viewport_);

  return err;
}

vvSocket::ErrorType vvSocketIO::getCompressedVector(virvo::CompressedVector& vec) const
{
  if (!getSocket())
    return vvSocket::VV_SOCK_ERROR;

  vvSocket::ErrorType err = vvSocket::VV_OK;

  int type = 0;

  if (err == vvSocket::VV_OK) err = getStdVector(vec.vector());
  if (err == vvSocket::VV_OK) err = getInt32(type);

  vec.setCompressionType(static_cast<virvo::CompressionType>(type));

  return err;
}

vvSocket::ErrorType vvSocketIO::putCompressedVector(virvo::CompressedVector const& vec) const
{
  if (!getSocket())
    return vvSocket::VV_SOCK_ERROR;

  vvSocket::ErrorType err = vvSocket::VV_OK;

  if (err == vvSocket::VV_OK) err = putStdVector(vec.vector());
  if (err == vvSocket::VV_OK) err = putInt32(static_cast<int>(vec.getCompressionType()));

  return err;
}

//----------------------------------------------------------------------------
/** get assigned vvSocket
*/
vvSocket* vvSocketIO::getSocket() const
{
  return _socket;
}

// EOF
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
