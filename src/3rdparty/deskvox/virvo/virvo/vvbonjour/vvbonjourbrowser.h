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

#ifndef _VV_BONJOURBROWSER_H_
#define _VV_BONJOURBROWSER_H_

#include "vvbonjour.h"
#include "vvbonjourentry.h"
#include "vvbonjoureventloop.h"

#include <vector>

/**
  Browser class for bonjour services.
  Found entries have to be resolved by vvBonjourResolver in order to create sockets and further useage.

  @author Stavros Delisavas (stavros.delisavas@uni-koeln.de
 */
class VIRVOEXPORT vvBonjourBrowser
{
public:
  /**
    Create a bonjour-browser and register an callback for any changes (optionally)
    @param externCallBack optional callback-function which is invoked on any change of bonjour entries
    @param callBackParam  optional parameter to pass through on externCallBack if given.
    */
  vvBonjourBrowser(void (*externCallBack)(void *) = NULL, void *callBackParam = NULL);
  ~vvBonjourBrowser();

  /**
    Initiated the bonjourbrowser to search for services. If no error occured, a list of entries will be saved.
    @param serviceType String of desired service type.
    @param domain      String of desired domain to search (optional). By default all local domains.
    @param to          Timeout in seconds. If not set, default will be used.
    @return Errorcode (0 == no error). See manpages of DNSSeriviceErrorType for further informations.
   */
  vvBonjour::ErrorType browseForServiceType(const std::string& serviceType, const std::string domain = "", const double to = 1.0);

  /**
    Returns list of found bonjourentries
    @return Vectorlist of vvBonjourEntry, which have to be resolved with vvBonjourResolver for further usage.
    */
  std::vector<vvBonjourEntry> getBonjourEntries() const;
  vvBonjourEventLoop* _eventLoop;
  std::vector<vvBonjourEntry> _bonjourEntries;

  // TODO: make these private
  double _timeout;
};

#endif
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
