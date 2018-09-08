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

#include "vvbonjourbrowser.h"

#include "../vvdebugmsg.h"

#include <algorithm>
#ifdef HAVE_BONJOUR
#include <dns_sd.h>
#endif
#include <iostream>
#include <ostream>
#include <sstream>

namespace
{
void (*externCallBack)(void *); 
void *callBackParam;

#ifdef HAVE_BONJOUR
void BrowseCallBack(DNSServiceRef serviceRef, DNSServiceFlags flags, uint32_t interfaceIndex,
                                      DNSServiceErrorType errorCode,
                                      const char * name, const char * type, const char * domain,
                                      void * context)
{
  vvDebugMsg::msg(3, "vvBonjourBrowser::BrowseCallBack() Enter");

  vvBonjourBrowser *instance = reinterpret_cast<vvBonjourBrowser*>(context);

  if (errorCode != kDNSServiceErr_NoError)
    vvDebugMsg::msg(3, "vvBonjourBrowser::BrowseCallBack() Leave");
  else
  {
    if(vvDebugMsg::getDebugLevel() >= 3)
    {
      std::string addString  = (flags & kDNSServiceFlagsAdd) ? "ADD" : "REMOVE";
      std::string moreString = (flags & kDNSServiceFlagsMoreComing) ? "MORE" : "    ";

      std::ostringstream msg;
      msg << addString << " " << moreString << " " << interfaceIndex << " " << name << "." << type << domain;
      vvDebugMsg::msg(0, msg.str().c_str());
    }
    vvBonjourEntry entry = vvBonjourEntry(name, type, domain);
    if(flags & kDNSServiceFlagsAdd)
    {
      instance->_bonjourEntries.push_back(entry);
    }
    else
    {
      std::vector<vvBonjourEntry>::iterator it;
      it = std::find(instance->_bonjourEntries.begin(), instance->_bonjourEntries.end(), entry);
      instance->_bonjourEntries.erase(it);
    }
  }

  if (!(flags & kDNSServiceFlagsMoreComing) && instance->_timeout!=-1.0)
  {
    instance->_eventLoop->stop();
    DNSServiceRefDeallocate(serviceRef);
  }
  if(::externCallBack)
  {
    ::externCallBack(::callBackParam);
  }
}
#endif
}

vvBonjourBrowser::vvBonjourBrowser(void (*externCallBack)(void *), void *callBackParam)
  : _eventLoop(NULL)
{
  _timeout = 1.0;
  ::externCallBack = externCallBack;
  ::callBackParam = callBackParam;
}

vvBonjourBrowser::~vvBonjourBrowser()
{
  delete _eventLoop;
}

vvBonjour::ErrorType vvBonjourBrowser::browseForServiceType(const std::string& serviceType, const std::string domain, const double to)
{
#ifdef HAVE_BONJOUR
  DNSServiceErrorType error;
  DNSServiceRef  serviceRef;

  _bonjourEntries.clear();

  error = DNSServiceBrowse(&serviceRef,
              0,                    // no flags
              0,                    // all network interfaces
              serviceType.c_str(),  // service type
              domain.c_str(),       // default domains
              BrowseCallBack,       // call back function
              this);                // adress of pointer to eventloop
  if (error == kDNSServiceErr_NoError)
  {
    _timeout = to;
    _eventLoop = new vvBonjourEventLoop(serviceRef);
    if(to != -1.0)
      _eventLoop->run(false, to);
    else
      _eventLoop->run(true, to);
  }
  else
  {
    std::ostringstream errmsg;
    errmsg << "vvBonjourBrowser::browseForServiceType(): DNSServiceBrowse() returned with error no " << error;
    vvDebugMsg::msg(2, errmsg.str().c_str());
    return vvBonjour::VV_ERROR;
  }

  return vvBonjour::VV_OK;
#else
  (void)serviceType;
  (void)domain;
  (void)to;
  return vvBonjour::VV_ERROR;
#endif
}

std::vector<vvBonjourEntry> vvBonjourBrowser::getBonjourEntries() const
{
  return _bonjourEntries;
}

#endif // VV_HAVE_BONJOUR

// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
