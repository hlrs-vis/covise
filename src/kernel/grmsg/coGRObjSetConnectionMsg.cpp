/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coGRObjSetConnectionMsg.h"
#include <cstdio>
#include <cstring>

namespace grmsg
{

GRMSGEXPORT coGRObjSetConnectionMsg::coGRObjSetConnectionMsg(coGRMsg::Mtype type, const char *obj_name, const char *connPoint1, const char *connPoint2, int connected, int enabled, const char *obj_name2)
    : coGRObjMsg(type, obj_name)
{
    initClass(connPoint1, connPoint2, connected, enabled, obj_name2);
}

GRMSGEXPORT coGRObjSetConnectionMsg::coGRObjSetConnectionMsg(const char *obj_name, const char *connPoint1, const char *connPoint2, int connected, int enabled, const char *obj_name2)
    : coGRObjMsg(SET_CONNECTIONPOINT, obj_name)
{
    initClass(connPoint1, connPoint2, connected, enabled, obj_name2);
}

GRMSGEXPORT coGRObjSetConnectionMsg::coGRObjSetConnectionMsg(const char *msg)
    : coGRObjMsg(msg)
{
    vector<string> tok = getAllTokens();

    if (!tok[0].empty())
    {
        string cp = tok[0];
        connPoint1_ = new char[cp.size() + 1];
        strcpy(connPoint1_, cp.c_str());
        is_valid_ = is_valid_ & 1;
    }
    else
    {
        is_valid_ = 0;
    }

    if (!tok[1].empty())
    {
        string cp = tok[1];
        connPoint2_ = new char[cp.size() + 1];
        strcpy(connPoint2_, cp.c_str());
        is_valid_ = is_valid_ & 1;
    }
    else
    {
        is_valid_ = 0;
    }

    if (!tok[2].empty())
    {
        string conn = tok[2];
        sscanf(conn.c_str(), "%d", &connected_);
        is_valid_ = is_valid_ & 1;
    }
    else
    {
        is_valid_ = 0;
    }

    if (!tok[3].empty())
    {
        string conn = tok[3];
        sscanf(conn.c_str(), "%d", &enabled_);
        is_valid_ = is_valid_ & 1;
    }
    else
    {
        is_valid_ = 0;
    }

    if (!tok[4].empty())
    {
        string cp = tok[4];
        obj_name2_ = new char[cp.size() + 1];
        strcpy(obj_name2_, cp.c_str());
        is_valid_ = is_valid_ & 1;
    }
    else
    {
        is_valid_ = 0;
    }
}

GRMSGEXPORT coGRObjSetConnectionMsg::~coGRObjSetConnectionMsg()
{
    if (connPoint1_)
        delete[] connPoint1_;

    if (connPoint2_)
        delete[] connPoint2_;

    if (obj_name2_)
        delete[] obj_name2_;
}

GRMSGEXPORT void coGRObjSetConnectionMsg::initClass(const char *connPoint1, const char *connPoint2, int connected, int enabled, const char *obj_name2)
{
    connPoint1_ = new char[strlen(connPoint1) + 1];
    strcpy(connPoint1_, connPoint1);
    addToken(connPoint1);

    connPoint2_ = new char[strlen(connPoint2) + 1];
    strcpy(connPoint2_, connPoint2);
    addToken(connPoint2);

    connected_ = connected;
    char str[1024];
    sprintf(str, "%d", connected_);
    addToken(str);

    enabled_ = enabled;
    sprintf(str, "%d", enabled_);
    addToken(str);

    obj_name2_ = new char[strlen(obj_name2) + 1];
    strcpy(obj_name2_, obj_name2);
    addToken(obj_name2_);

    is_valid_ = 1;
}
}
