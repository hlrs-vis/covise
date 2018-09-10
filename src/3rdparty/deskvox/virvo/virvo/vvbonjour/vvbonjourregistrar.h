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

#ifndef _VV_BONJOURREGISTRAR_H_
#define _VV_BONJOURREGISTRAR_H_

#include "vvbonjour.h"
#include "vvbonjourentry.h"
#include "vvbonjoureventloop.h"
#include "vvinttypes.h"

/**
  Class for registering a Bonjourservice.
  A listening socket should be prepared before the service is registered.
  Also, the service should be unregistered as soon as a connection ocurred and the port is in use.

  @author Stavros Delisavas (stavros.delisavas@uni-koeln.de)
  */
class VIRVOEXPORT vvBonjourRegistrar
{
public:
  vvBonjourRegistrar();
  ~vvBonjourRegistrar();

  /**
    Register a bonjourservice
    @param entry vvBonjourEntry with desired registration data.
    @param port Port on which the service provider is listening for incoming connections
    @return Errorcode (0 == no error). See manpages of DNSSeriviceErrorType for further informations.
    */
  bool registerService(const vvBonjourEntry& entry, const ushort port);
  /**
    Unregister the registered service
    */
  void unregisterService();

  // TODO: make these private
  vvBonjourEventLoop *_eventLoop;
  vvBonjourEntry      _registeredService;
};

#endif
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
