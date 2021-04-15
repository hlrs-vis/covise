/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
#ifndef VRB_CONNECTION_MAP_H
#define VRB_CONNECTION_MAP_H

#include <vector>
#include <vrb/UserInfo.h>
#include <comsg/PROXY.h>
namespace vrb{

struct ConnectionState{
  const UserInfo from, to;
  const covise::ConnectionCapability state;
  ConnectionState(const UserInfo &from, const UserInfo &to, covise::ConnectionCapability s);
};

bool operator==(const ConnectionState &a, const ConnectionState &b);

struct ConnectionMap
{

  void addConn(const ConnectionState &conn);
  covise::ConnectionCapability check(const UserInfo &from, const UserInfo &to) const;
private:
  std::vector<ConnectionState> m_conns;
};
}

#endif