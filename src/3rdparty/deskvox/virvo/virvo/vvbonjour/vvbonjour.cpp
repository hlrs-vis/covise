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

#include "vvbonjour.h"

#ifdef HAVE_CONFIG_H
#include "vvconfig.h"
#endif

#include "vvbonjourbrowser.h"
#include "vvbonjourresolver.h"

#ifdef HAVE_BONJOUR
#include <dns_sd.h>
#endif

#include <sstream>

vvBonjour::vvBonjour()
{}

vvBonjour::~vvBonjour()
{}

std::vector<vvTcpSocket*> vvBonjour::getSocketsFor(const std::string& serviceType, const std::string& domain) const
{
  std::vector<vvTcpSocket*> sockets;
  std::vector<vvBonjourEntry> entries = getEntriesFor(serviceType, domain);

  for (std::vector<vvBonjourEntry>::const_iterator it = entries.begin(); it != entries.end(); ++it)
  {
    vvBonjourResolver resolver;
    if(vvBonjour::VV_OK == resolver.resolveBonjourEntry(*it))
    {
      vvTcpSocket *socket = new vvTcpSocket();
      if(vvSocket::VV_OK == socket->connectToHost(resolver._hostname.c_str(), resolver._port))
      {
        sockets.push_back(socket);
      }
      else
      {
        delete socket;
      }
    }
  }
  return sockets;
}

std::vector<vvBonjourEntry> vvBonjour::getEntriesFor(const std::string& serviceType, const std::string& domain) const
{
  vvBonjourBrowser browser;
  browser.browseForServiceType(serviceType, domain);

  return browser.getBonjourEntries();
}

std::vector<std::string> vvBonjour::getConnectionStringsFor(const std::string& serviceType, const std::string& domain) const
{
  std::vector<vvBonjourEntry> entries = getEntriesFor(serviceType, domain);
  std::vector<std::string> connectionStrings;

  for (std::vector<vvBonjourEntry>::const_iterator it = entries.begin(); it != entries.end(); ++it)
  {
    vvBonjourResolver resolver;
    if(vvBonjour::VV_OK == resolver.resolveBonjourEntry(*it))
    {
      std::ostringstream hostAndPort;
      hostAndPort << resolver._hostname << ":" << resolver._port;
      connectionStrings.push_back(hostAndPort.str());
    }
  }

  return connectionStrings;
}

#endif // VV_HAVE_BONJOUR

// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
