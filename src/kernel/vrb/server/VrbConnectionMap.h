/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
#ifndef VRB_CONNECTION_MAP_H
#define VRB_CONNECTION_MAP_H

#include <vector>
#include <vrb/UserInfo.h>

namespace vrb{

struct ConnectionState{
  const UserInfo from, to;
  enum State
  {
    NotChecked,
    DirectConnectionPossible,
    ProxyRequired
  } const state;
  ConnectionState(const UserInfo &from, const UserInfo &to, State s);

};

bool operator==(const ConnectionState &a, const ConnectionState &b);

struct ConnectionMap
{

  void addConn(const ConnectionState &conn);
  ConnectionState::State check(const UserInfo &from, const UserInfo &to) const;
private:
  std::vector<ConnectionState> m_conns;
};
}

#endif