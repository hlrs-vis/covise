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

#ifndef _VV_BONJOURRESOLVER_H_
#define _VV_BONJOURRESOLVER_H_

#include "vvbonjour.h"
#include "vvbonjourentry.h"
#include "vvbonjoureventloop.h"
#include "vvtcpsocket.h"
#include "vvsocketmonitor.h"

/**
  Class for resolving bonjourentries and creating socket-connections to desired bonjour services.
  */
class VIRVOEXPORT vvBonjourResolver
{
public:
  vvBonjourResolver();
  ~vvBonjourResolver();

  /**
    Resolves given Bonjourentry
    @param entry Entry of type vvBonjourEntry found by vvBonjourBrowser
    @return Errorcode (0 == no error). See manpages of DNSSeriviceErrorType for further informations.
    */
  vvBonjour::ErrorType resolveBonjourEntry(const vvBonjourEntry& entry);
  /**
    Create a socket connection of vvSocket (TCP) connected to the resolved bonjourentry
    @return Pointer to ready to use vvSocket or NULL in case of error.
    */
  vvTcpSocket* getBonjourSocket() const;

  // TODO: make these private
  vvBonjourEventLoop *_eventLoop;
  std::string _hostname;
  ushort      _port;
};

#endif
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
