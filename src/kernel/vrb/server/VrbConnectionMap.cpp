/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "VrbConnectionMap.h"
#include <cassert>
#include <algorithm>
using namespace vrb;
using namespace covise;


ConnectionState::ConnectionState(const UserInfo &from, const UserInfo &to, ConnectionCapability s)
    : from(from), to(to), state(s)
{
}

void ConnectionMap::addConn(const ConnectionState& c)
{
  assert(c.state != covise::ConnectionCapability::NotChecked);
  assert(check(c.from, c.to) == covise::ConnectionCapability::NotChecked);

  m_conns.emplace_back(c);
}

covise::ConnectionCapability ConnectionMap::check(const UserInfo &from, const UserInfo &to) const
{
  auto c = std::find_if(m_conns.begin(), m_conns.end(), [&from, &to](const ConnectionState &s) {
    return s.from.ipAdress == from.ipAdress && s.to.ipAdress == to.ipAdress &&
           s.from.hostName == from.hostName && s.to.hostName == to.hostName &&
           s.from.userName == from.userName && s.to.userName == to.userName;
  });
  if (c != m_conns.end())
  {
    return c->state;
  }
  return covise::ConnectionCapability::NotChecked;
}