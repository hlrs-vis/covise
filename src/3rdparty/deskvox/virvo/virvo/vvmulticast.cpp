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

#ifdef HAVE_CONFIG_H
#include "vvconfig.h"
#endif

#include "vvdebugmsg.h"
#include "vvinttypes.h"
#include "vvmulticast.h"
#include "vvsocketmonitor.h"
#include "vvinttypes.h"

#include <algorithm>
#include <cmath>

#ifdef HAVE_NORM
#include <normApi.h>
#include <stdlib.h>
#endif

vvMulticast::vvMulticast(const MulticastType type, const MulticastApi api, const char* addr, const ushort port)
: _type(type), _api(api), _socket(NULL)
{
  if(VV_NORM == _api)
  {
#ifdef HAVE_NORM
    _instance = NormCreateInstance();
    _session = NormCreateSession(_instance, addr, port, NORM_NODE_ANY);

    NormSetCongestionControl(_session, true);
    if(VV_SENDER == type)
    {
      NormSessionId sessionId = (NormSessionId)rand();
      // TODO: Adjust these numbers depending on the used network topology
      NormSetTransmitRate(_session, 8e10);
      NormSetTransmitCacheBounds(_session, CHUNK_SIZE, 1, 128);
      //NormSetGroupSize(_session, 10);
      NormStartSender(_session, sessionId, 1024*1024, 1400, 64, 2);
      NormSetTxSocketBuffer(_session, 1024*1024*32);
    }
    else if(VV_RECEIVER == type)
    {
      NormStartReceiver(_session, 1024*1024);
      NormSetRxSocketBuffer(_session, 1024*1024*64);
      NormDescriptor normDesc = NormGetDescriptor(_instance);
      _nodes.push_back(NormGetLocalNodeId(_session));
      _normSocket = new vvUdpSocket();
      _normSocket->setSockfd(int(normDesc));
    }
#else
    (void)addr;
    (void)port;
#endif
  }
  else if(VV_VVSOCKET == _api)
  {
    _socket = new vvUdpSocket();
    vvSocket::ErrorType retVal;
    if(VV_SENDER == _type)
    {
      retVal = _socket->multicast(addr, port, vvUdpSocket::VV_MC_SENDER);
    }
    else
    {
      retVal = _socket->multicast(addr, port, vvUdpSocket::VV_MC_RECEIVER);
    }
    if(retVal != vvSocket::VV_OK)
    {
      vvDebugMsg::msg(2, "vvMulticast() error creating multicast socket");
      return;
    }
    if(_socket->setParameter(vvSocket::VV_NONBLOCKING, true) != vvSocket::VV_OK)
    {
      vvDebugMsg::msg(2, "vvMulticast() error while calling vvSocket::setParameter()");
    }
  }
}

vvMulticast::~vvMulticast()
{
  if(VV_NORM == _api)
  {
#ifdef HAVE_NORM
    if(VV_SENDER == _type)
    {
      NormStopSender(_session);
    }
    else if(VV_RECEIVER == _type)
    {
      NormStopReceiver(_session);
      delete _normSocket;
    }
    NormDestroySession(_session);
    NormDestroyInstance(_instance);
#endif
  }
  else if(VV_VVSOCKET == _api)
  {
    delete _socket;
  }
}

ssize_t vvMulticast::write(const uchar* bytes, const size_t size, double timeout)
{
  vvDebugMsg::msg(3, "vvMulticast::write()");

  if(VV_NORM == _api)
  {
#ifdef HAVE_NORM
    for(std::vector<NormNodeId>::const_iterator it = _nodes.begin(); it != _nodes.end(); ++it)
    {
      NormAddAckingNode(_session, *it);
    }

    for(unsigned int i=0; i<size; i+=CHUNK_SIZE)
    {
      size_t frameSize = std::min(size_t(CHUNK_SIZE), size);
      _object = NormDataEnqueue(_session, (char*)&bytes[i*CHUNK_SIZE], frameSize);
      NormSetWatermark(_session, _object);

      if(NORM_OBJECT_INVALID ==_object)
      {
        vvDebugMsg::msg(2, "vvMulticast::write(): Norm Object is invalid!");
        return -2;
      }

      NormDescriptor normDesc = NormGetDescriptor(_instance);

      vvSocketMonitor* monitor = new vvSocketMonitor;
      std::vector<vvSocket*> sock;
      vvUdpSocket *udpSock = new vvUdpSocket();
      udpSock->setSockfd(int(normDesc));
      sock.push_back(reinterpret_cast<vvSocket*>(udpSock));
      monitor->setReadFds(sock);

      NormEvent theEvent;
      size_t bytesSent = 0;
      bool keepGoing = true;
      while(keepGoing)
      {
        vvSocket* ready = NULL;
        vvSocketMonitor::ErrorType err = monitor->wait(&ready, &timeout);
        if(vvSocketMonitor::VV_TIMEOUT == err)
        {
          vvDebugMsg::msg(2, "vvMulticast::write() timeout reached.");
          return bytesSent;
        }
        else if(vvSocketMonitor::VV_ERROR == err)
        {
          vvDebugMsg::msg(2, "vvMulticast::write() error.");
          return -1;
        }
        else
        {
          NormGetNextEvent(_instance, &theEvent);
          switch(theEvent.type)
          {
          case NORM_CC_ACTIVE:
            vvDebugMsg::msg(3, "vvMulticast::write() NORM_CC_ACTIVE: transmission still active");
            break;
          case NORM_TX_FLUSH_COMPLETED:
          case NORM_LOCAL_SENDER_CLOSED:
          case NORM_TX_OBJECT_SENT:
            vvDebugMsg::msg(3, "vvMulticast::write(): chunk-transfer completed.");
            bytesSent += size_t(NormObjectGetSize(theEvent.object));
            keepGoing = false;
            break;
          default:
            {
              std::string eventmsg = std::string("vvMulticast::write() Norm-Event: ");
              eventmsg += theEvent.type;
              vvDebugMsg::msg(3, eventmsg.c_str());
              break;
            }
          }
        }
      }
    }
    return size;
#else
    (void)bytes;
    (void)size;
    (void)timeout;
    return -1;
#endif
  }
  else
  {
    // number datagrams
    uchar *ndata = numberConsecutively(bytes, size);
    size_t nsize = size+size_t(ceil(float(size)/float((DGRAM_SIZE-4)))*4.0);

    size_t nleft = nsize;

    vvSocketMonitor monitor = vvSocketMonitor();
    std::vector<vvSocket*> sock;
    vvUdpSocket *udpSock = new vvUdpSocket;
    udpSock->setSockfd(_socket->getSockfd());
    sock.push_back(reinterpret_cast<vvSocket*>(udpSock));
    monitor.setWriteFds(sock);

    while(nleft > 0)
    {
      vvSocket* ready = NULL;
      vvSocketMonitor::ErrorType smErr = monitor.wait(&ready, &timeout);
      if(vvSocketMonitor::VV_TIMEOUT == smErr)
      {
        vvDebugMsg::msg(2, "vvMulticast::write() timeout reached.");
        return size-nleft;
      }
      else if(vvSocketMonitor::VV_ERROR == smErr)
      {
        vvDebugMsg::msg(2, "vvMulticast::write() error.");
        return -1;
      }
      else
      {
        size_t towrite = std::min(size_t(DGRAM_SIZE), nleft);
        vvSocket::ErrorType err = _socket->writeData((uchar*)&ndata[nsize-nleft], towrite);
        if(vvSocket::VV_OK != err)
        {
          vvDebugMsg::msg(0, "vvMulticast::write() error", true);
          return -1;
        }
        else
        {
          nleft -= towrite;
        }
      }
    }
    return size;
  }
}

ssize_t vvMulticast::read(uchar* data, const size_t size, double timeout)
{
  vvDebugMsg::msg(3, "vvMulticast::read()");

  if(VV_NORM == _api)
  {
#ifdef HAVE_NORM
    vvSocketMonitor monitor;

    std::vector<vvSocket*> sock;
    sock.push_back(_normSocket);
    monitor.setReadFds(sock);

    NormEvent theEvent;
    size_t chunk = 0;
    size_t bytesReceived = 0;
    bool keepGoing = true;
    do
    {
      vvSocket* ready = NULL;
      vvSocketMonitor::ErrorType err = monitor.wait(&ready, &timeout);
      if(vvSocketMonitor::VV_TIMEOUT == err)
      {
        vvDebugMsg::msg(2, "vvMulticast::read() timeout reached.");
        return bytesReceived;
      }
      else if(vvSocketMonitor::VV_ERROR == err)
      {
        vvDebugMsg::msg(2, "vvMulticast::read() error.");
        return -1;
      }
      else
      {
        NormGetNextEvent(_instance, &theEvent);
        switch(theEvent.type)
        {
        case NORM_RX_OBJECT_UPDATED:
          vvDebugMsg::msg(3, "vvMulticast::read() NORM_RX_OBJECT_UPDATED: the identified receive object has newly received data content.");
          break;
        case NORM_RX_OBJECT_COMPLETED:
          {
            vvDebugMsg::msg(3, "vvMulticast::read() NORM_RX_OBJECT_COMPLETED: transfer completed.");
            bytesReceived += size_t(NormObjectGetSize(theEvent.object));
            // copy data into array
            uchar *t_data = (uchar*)NormDataDetachData(theEvent.object);
            for(int i=0;i<NormObjectGetSize(theEvent.object);i++)
            {
              data[i+chunk*CHUNK_SIZE] = t_data[i];
            }
            chunk++;
            break;
          }
        case NORM_RX_OBJECT_ABORTED:
          vvDebugMsg::msg(2, "vvMulticast::read() NORM_RX_OBJECT_ABORTED: transfer incomplete!");
          return -1;
          break;
        default:
          {
            std::string eventmsg = std::string("vvMulticast::read() Norm-Event: ");
            eventmsg += theEvent.type;
            vvDebugMsg::msg(3, eventmsg.c_str());
            break;
          }
        }
      }
      if(bytesReceived >= size) keepGoing = false;
    }
    while((0.0 < timeout || -1.0 == timeout) && keepGoing);
    return bytesReceived;
#else
    (void)size;
    (void)data;
    (void)timeout;
    return -1;
#endif
  }
  else
  {
    size_t nsize = size+size_t(ceil(float(size)/float((DGRAM_SIZE-4)))*4.0);
    size_t nleft = nsize;

    vvSocketMonitor monitor = vvSocketMonitor();
    std::vector<vvSocket*> sock;
    sock.push_back(_socket);
    monitor.setReadFds(sock);

    while(nleft > 0)
    {
      vvSocket* ready = NULL;
      vvSocketMonitor::ErrorType smErr = monitor.wait(&ready, &timeout);
      if(vvSocketMonitor::VV_TIMEOUT == smErr)
      {
        vvDebugMsg::msg(2, "vvMulticast::read() timeout reached.");
        return -1;
      }
      else if(vvSocketMonitor::VV_ERROR == smErr)
      {
        vvDebugMsg::msg(2, "vvMulticast::read() error.");
        return -1;
      }
      else
      {
        uchar dgram[DGRAM_SIZE];
        ssize_t ret;
        vvSocket::ErrorType err;

        err = _socket->readData(dgram, DGRAM_SIZE, &ret);
        if(vvSocket::VV_OK != err)
        {
          vvDebugMsg::msg(2, "vvMulticast::read() error", true);
          return -1;
        }
        else
        {
          bytesToInt32 t;
          t.x[0] = dgram[ret-4];
          t.x[1] = dgram[ret-3];
          t.x[2] = dgram[ret-2];
          t.x[3] = dgram[ret-1];

          uint32_t c = ntohl(t.y);

          size_t pos = c*(DGRAM_SIZE-4);
          for(int i=0;i<ret-4;i++)
          {
            data[pos+i] = dgram[i];
          }
          nleft -= ret;
        }
      }
    }
    return size;
  }
}

uchar* vvMulticast::numberConsecutively(const uchar* data, const size_t size)
{
  uchar *numbered = new uchar[size_t(size+(ceil(float(size)/float((DGRAM_SIZE-4)))*4))];

  size_t i = 0;
  size_t n = 0;
  uint32_t c = 0;
  while(i < size)
  {
    numbered[n++] = data[i++];
    if((n+4)%DGRAM_SIZE == 0)
    {
      *((uint32_t*)(&numbered[n])) = htonl(c);
      n += 4;
      c++;
    }
  }
  // add last number if dgram not full
  if(n % DGRAM_SIZE != 0)
  {
    *((uint32_t*)(&numbered[n])) = htonl(c);
  }
  return numbered;
}

// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
