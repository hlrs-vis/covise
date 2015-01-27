/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coGRCreateDefaultViewpointMsg.h"
#include <iostream>
#include <sstream>
#include <cstring>
#include <cstdlib>
#include <cstdio>

using namespace grmsg;

GRMSGEXPORT coGRCreateDefaultViewpointMsg::coGRCreateDefaultViewpointMsg(const char *msg)
    : coGRMsg(msg)
{
    name_ = NULL;

    vector<string> tok = getAllTokens();
    if (!tok[0].empty())
    {
        name_ = strdup(tok[0].c_str());
    }
    else
    {
        name_ = NULL;
        is_valid_ = 0;
    }
    if (!tok[1].empty())
    {
        sscanf(tok[1].c_str(), "%d", &id_);
    }
    else
    {
        id_ = -1;
        is_valid_ = 0;
    }
}

GRMSGEXPORT coGRCreateDefaultViewpointMsg::coGRCreateDefaultViewpointMsg(const char *name, int id)
    : coGRMsg(CREATE_DEFAULT_VIEWPOINT)
{
    is_valid_ = 1;

    id_ = id;
    name_ = strdup(name);

    ostringstream stream;
    stream << id_;

    addToken(name_);
    addToken(stream.str().c_str());
}

GRMSGEXPORT coGRCreateDefaultViewpointMsg::~coGRCreateDefaultViewpointMsg()
{
    COGRMSG_SAFEFREE(name_);
}

GRMSGEXPORT int coGRCreateDefaultViewpointMsg::getViewpointId()
{
    return id_;
}

GRMSGEXPORT const char *coGRCreateDefaultViewpointMsg::getName()
{
    return name_;
}
