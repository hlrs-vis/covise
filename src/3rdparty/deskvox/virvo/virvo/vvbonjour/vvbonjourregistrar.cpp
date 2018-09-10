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

#include "vvbonjourregistrar.h"

#include "../vvdebugmsg.h"
#include "../vvsocket.h"
#include "vvtoolshed.h"

#ifdef HAVE_BONJOUR
#include <dns_sd.h>
#endif
#include <iostream>
#include <sstream>

namespace
{
#ifdef HAVE_BONJOUR
_DNSServiceRef_t   *serviceRef;

void RegisterCallBack(DNSServiceRef service,
           DNSServiceFlags flags,
           DNSServiceErrorType errorCode,
           const char * name,
           const char * type,
           const char * domain,
           void * context)
{
  vvDebugMsg::msg(3, "vvBonjourRegistrar::RegisterCallBack() Enter");
  (void)service;
  (void)flags;

  vvBonjourRegistrar* instance = reinterpret_cast<vvBonjourRegistrar*>(context);

  if (errorCode != kDNSServiceErr_NoError)
  {
    instance->_registeredService = vvBonjourEntry(name, type, domain);
    vvDebugMsg::msg(3, "vvBonjourRegistrar::RegisterCallBack() error");
  }
  else
  {
    if(vvDebugMsg::getDebugLevel() >= 3)
    {
      std::ostringstream errmsg;
      errmsg << "vvBonjourRegistrar::RegisterCallBack() Entry registered: " << name << "." << type << domain;
      vvDebugMsg::msg(0, errmsg.str().c_str());
    }
  }

  if (!(flags & kDNSServiceFlagsMoreComing))
  {
    // registering done
    instance->_eventLoop->_noMoreFlags = true;
    instance->_eventLoop->stop();
  }
}
#endif
}

vvBonjourRegistrar::vvBonjourRegistrar()
{
#ifdef HAVE_BONJOUR
  ::serviceRef = NULL;
#endif
  _eventLoop = NULL;
}

vvBonjourRegistrar::~vvBonjourRegistrar()
{
#ifdef HAVE_BONJOUR
  if (::serviceRef != NULL)
  {
    unregisterService();
  }
#endif
}

bool vvBonjourRegistrar::registerService(const vvBonjourEntry& entry, const ushort port)
{
#ifdef HAVE_BONJOUR
  vvDebugMsg::msg(3, "vvBonjourRegistrar::registerService() Enter");

  DNSServiceErrorType error = DNSServiceRegister(&::serviceRef,
                0,                // no flags
                0,                // all network interfaces
                entry.getServiceName().c_str(),
                entry.getRegisteredType().c_str(),
                entry.getReplyDomain().c_str(),
                NULL,             // use default host name
                htons(port),      // port number
                0,                // length of TXT record
                NULL,             // no TXT record
                RegisterCallBack, // call back function
                this);            // no context

  if (error == kDNSServiceErr_NoError)
  {
    _eventLoop = new vvBonjourEventLoop(::serviceRef);
    _eventLoop->run(false, -1.0);
    return true;
  }
  else
  {
    vvDebugMsg::msg(2, "vvBonjourRegistrar::registerService(): DNSServiceResolve failed with error code ", error);
    return false;
  }
#else
  (void)entry;
  (void)port;
  return false;
#endif
}

void vvBonjourRegistrar::unregisterService()
{
#ifdef HAVE_BONJOUR
  if(!::serviceRef)
  {
    vvDebugMsg::msg(2, "vvBonjourRegistrar::unregisterService() no service registered");
    return;
  }

  if(_eventLoop)
  {
    _eventLoop->stop();
    delete _eventLoop;
    _eventLoop = NULL;
  }

  DNSServiceRefDeallocate(::serviceRef);
  ::serviceRef = NULL;
#endif
}

#endif // VV_HAVE_BONJOUR

// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
