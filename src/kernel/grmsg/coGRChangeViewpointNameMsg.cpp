/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coGRChangeViewpointNameMsg.h"
#include <iostream>
#include <sstream>
#include <cstdio>
#include <cstring>

using namespace std;

namespace grmsg
{

GRMSGEXPORT coGRChangeViewpointNameMsg::coGRChangeViewpointNameMsg(const char *msg)
    : coGRMsg(msg)
{
    is_valid_ = 1;
    name_ = NULL;
    vector<string> tok = getAllTokens();

    if (!tok[0].empty())
    {
        string id = tok[0];
        sscanf(id.c_str(), "%d", &id_);
    }
    else
    {
        is_valid_ = 0;
    }

    if (!tok[1].empty())
    {
        name_ = new char[strlen(tok[1].c_str()) + 1];
        strcpy(name_, tok[1].c_str());
    }
    else
    {
        is_valid_ = 0;
    }
}

GRMSGEXPORT coGRChangeViewpointNameMsg::coGRChangeViewpointNameMsg(int id, const char *name)
    : coGRMsg(CHANGE_VIEWPOINT_NAME)
{
    is_valid_ = 1;

    id_ = id;
    ostringstream stream;
    stream << id_;
    addToken(stream.str().c_str());

    name_ = new char[strlen(name) + 1];
    strcpy(name_, name);
    addToken(name_);
}

GRMSGEXPORT int coGRChangeViewpointNameMsg::getId()
{
    return id_;
}

GRMSGEXPORT char *coGRChangeViewpointNameMsg::getName()
{
    return name_;
}
}
