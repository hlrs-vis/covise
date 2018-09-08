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

#ifndef _VV_BONJOURENTRY_H_
#define _VV_BONJOURENTRY_H_

#include "../vvexport.h"
#include <string>

/*!
 * Class for saving bonjour entries. Each entry consists of a name a type and a domain.
 * The name can be any human readable name
 * The type must be in the form of _typename._protocol e.g. "_vvserver._tcp"
 * The domain can be set to any desired reply domain, else default is set to .local
 *
 * @author Stefan Zellmann (zellmans@uni-koeln.de)
 */
class VIRVOEXPORT vvBonjourEntry
{
public:
  vvBonjourEntry();
  vvBonjourEntry(const std::string& serviceName,
                 const std::string& registeredType,
                 const std::string& replyDomain);

  std::string getServiceName() const;
  std::string getRegisteredType() const;
  std::string getReplyDomain() const;

  bool operator==(const vvBonjourEntry& rhs) const;
private:
  std::string _serviceName;
  std::string _registeredType;
  std::string _replyDomain;
};

#endif
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
