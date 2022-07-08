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
    name_ = "";
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
        name_ = tok[1];
    }
    else
    {
        is_valid_ = 0;
    }
}

GRMSGEXPORT coGRChangeViewpointNameMsg::coGRChangeViewpointNameMsg(int id, const char *name)
    : coGRMsg(CHANGE_VIEWPOINT_NAME)
    , name_(name)
{
    is_valid_ = 1;

    id_ = id;
    ostringstream stream;
    stream << id_;
    addToken(stream.str().c_str());

    name_ = new char[strlen(name) + 1];
    addToken(name_.c_str());
}

GRMSGEXPORT int coGRChangeViewpointNameMsg::getId() const
{
    return id_;
}

GRMSGEXPORT const char *coGRChangeViewpointNameMsg::getName()  const
{
    return name_.c_str();
}
}
