/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coGRCreateViewpointMsg.h"
#include <iostream>
#include <sstream>
#include <cstring>
#include <cstdlib>
#include <cstdio>

using namespace std;
using namespace grmsg;

GRMSGEXPORT coGRCreateViewpointMsg::coGRCreateViewpointMsg(const char *msg)
    : coGRMsg(msg)
{
    name_ = NULL;
    view_ = NULL;
    plane_ = NULL;
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
    if (!tok[2].empty())
    {
        view_ = strdup(tok[2].c_str());
    }
    else
    {
        view_ = NULL;
        is_valid_ = 0;
    }
    if (!tok[3].empty())
    {
        plane_ = strdup(tok[3].c_str());
    }
    else
    {
        //old message set no clipplane
        plane_ = strdup("");
    }
}

GRMSGEXPORT coGRCreateViewpointMsg::coGRCreateViewpointMsg(const char *name, int id, const char *view, const char *clipplane)
    : coGRMsg(CREATE_VIEWPOINT)
{
    is_valid_ = 1;

    id_ = id;
    name_ = strdup(name);
    view_ = strdup(view);
    plane_ = strdup(clipplane);

    ostringstream stream;
    stream << id_;

    addToken(name_);
    addToken(stream.str().c_str());
    addToken(view_);
    addToken(plane_);
}

GRMSGEXPORT coGRCreateViewpointMsg::~coGRCreateViewpointMsg()
{
    COGRMSG_SAFEFREE(name_);
    COGRMSG_SAFEFREE(view_);
    COGRMSG_SAFEFREE(plane_);
} //coGRCreateViewpointMsg::~coGRCreateViewpointMsg

GRMSGEXPORT int coGRCreateViewpointMsg::getViewpointId()
{
    return id_;
}

GRMSGEXPORT const char *coGRCreateViewpointMsg::getName()
{
    return name_;
}

GRMSGEXPORT const char *coGRCreateViewpointMsg::getView()
{
    return view_;
}

GRMSGEXPORT const char *coGRCreateViewpointMsg::getClipplane()
{
    return plane_;
}
