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

#if VV_HAVE_BONJOUR

#include "vvbonjourresolver.h"

#include "../vvdebugmsg.h"
#include "../vvsocket.h"

#ifdef HAVE_BONJOUR
#include <dns_sd.h>
#endif

#include <iostream>
#include <sstream>
#include <vector>

namespace
{
#ifdef HAVE_BONJOUR
void ResolveCallBack(DNSServiceRef,
          DNSServiceFlags flags,
          uint32_t ,      // interface
          DNSServiceErrorType errorCode,
          const char *fullname,
          const char *hosttarget,
          uint16_t port,
          uint16_t ,      // txt length
          const uchar *,  // txt record
          void *context)
{
  vvDebugMsg::msg(3, "vvBonjourBrowser::ResolveCallBack() Enter");

  vvBonjourResolver *instance = reinterpret_cast<vvBonjourResolver*>(context);

  if (errorCode != kDNSServiceErr_NoError)
  {
    vvDebugMsg::msg(3, "vvBonjourBrowser::ResolveCallBack() error");
  }
  else
  {
    if(vvDebugMsg::getDebugLevel() >= 0)
    {
      std::ostringstream errmsg;
      errmsg << "vvBonjourResolver::ResolveCallBack() Entry resolved: " << fullname << " is at " << hosttarget << ":" << ntohs(port);
      vvDebugMsg::msg(2, errmsg.str().c_str());
    }

    instance->_hostname = hosttarget;
    instance->_port = ntohs(port);
  }

  if (!(flags & kDNSServiceFlagsMoreComing))
  {
    instance->_eventLoop->stop();
  }
}
#endif
}

vvBonjourResolver::vvBonjourResolver()
  : _eventLoop(NULL), _hostname(""), _port(0)
{
}

vvBonjourResolver::~vvBonjourResolver()
{
  delete _eventLoop;
}

vvBonjour::ErrorType vvBonjourResolver::resolveBonjourEntry(const vvBonjourEntry& entry)
{
#ifdef HAVE_BONJOUR
  vvDebugMsg::msg(3, "vvBonjourResolver::resolveBonjourEntry()");
  DNSServiceErrorType error;
  DNSServiceRef  serviceRef;

  error = DNSServiceResolve(&serviceRef,
                 0,                 // no flags
                 0,                 // all network interfaces
                 entry.getServiceName().c_str(),    //name
                 entry.getRegisteredType().c_str(), // service type
                 entry.getReplyDomain().c_str(),    //domain
                 ResolveCallBack,
                 this);             // no context

  if (error == kDNSServiceErr_NoError)
  {
    _eventLoop = new vvBonjourEventLoop(serviceRef);
    _eventLoop->run();
    return vvBonjour::VV_OK;
  }
  else
  {
    vvDebugMsg::msg(2, "vvBonjourResolver::resolveBonjourEntry(): DNSServiceResolve failed with error code ", error);
    return vvBonjour::VV_ERROR;
  }
#else
  (void)entry;
  return vvBonjour::VV_ERROR;
#endif
}

vvTcpSocket* vvBonjourResolver::getBonjourSocket() const
{
  vvDebugMsg::msg(3, "vvBonjourResolver::getBonjourSocket() enter");

  if(_hostname.length() > 0 && 0 != _port)
  {
    vvTcpSocket* sock = new vvTcpSocket;
    sock->connectToHost(_hostname.c_str(), _port);
    return sock;
  }
  else
  {
    vvDebugMsg::msg(2, "vvBonjourResolver::getBonjourSocket() hostname and/or port not resolved");
    return NULL;
  }
}

#endif // VV_HAVE_BONJOUR

// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
