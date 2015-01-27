/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coGRChangeViewpointIdMsg.h"
#include <iostream>
#include <sstream>
#include <cstdio>

using namespace grmsg;

GRMSGEXPORT coGRChangeViewpointIdMsg::coGRChangeViewpointIdMsg(const char *msg)
    : coGRMsg(msg)
{
    vector<string> tok = getAllTokens();
    if (!tok[0].empty())
    {
        sscanf(tok[0].c_str(), "%d", &oldId_);
    }
    else
    {
        oldId_ = -1;
        is_valid_ = 0;
    }
    if (!tok[1].empty())
    {
        sscanf(tok[1].c_str(), "%d", &newId_);
    }
    else
    {
        newId_ = -1;
        is_valid_ = 0;
    }
}

GRMSGEXPORT coGRChangeViewpointIdMsg::coGRChangeViewpointIdMsg(int oldId, int newId)
    : coGRMsg(CHANGE_VIEWPOINT_ID)
{
    is_valid_ = 1;

    oldId_ = oldId;
    newId_ = newId;

    ostringstream oldIdStream;
    oldIdStream << oldId_;

    ostringstream newIdStream;
    newIdStream << newId_;

    addToken(oldIdStream.str().c_str());
    addToken(newIdStream.str().c_str());
}

GRMSGEXPORT int coGRChangeViewpointIdMsg::getOldId()
{
    return oldId_;
}

GRMSGEXPORT int coGRChangeViewpointIdMsg::getNewId()
{
    return newId_;
}
