/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "netLink.h"
#include "module.h"
#include <algorithm>
using namespace covise::controller;

netlink::netlink()
{
    mptr = NULL;
}

void netlink::set_name(const std::string &str)
{
    name = str;
}

void netlink::set_instanz(const std::string &str)
{
    instanz = str;
}

void netlink::set_host(const std::string &str)
{
    host = str;
}

void netlink::del_link(const std::string &name, const std::string &instance, const std::string &host)
{
    mptr->netLinks.erase(std::remove_if(mptr->netLinks.begin(), mptr->netLinks.end(), [&name, &instance, &host](const netlink &link) {
                             return link.get_name() == name && link.get_instanz() == instance && link.get_host() == host;
                         }),
                         mptr->netLinks.end());
}