/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coGRActivatedViewpointMsg.h"
#include <iostream>
#include <sstream>
#include <cstdio>

using namespace grmsg;
using namespace std;

GRMSGEXPORT coGRActivatedViewpointMsg::coGRActivatedViewpointMsg(const char *msg)
    : coGRMsg(msg)
{
    sscanf(extractFirstToken().c_str(), "%d", &id_);
}

GRMSGEXPORT coGRActivatedViewpointMsg::coGRActivatedViewpointMsg(int id)
    : coGRMsg(ACTIVATED_VIEWPOINT)
{
    id_ = id;
    ostringstream stream;
    stream << id;
    addToken(stream.str().c_str());
}

GRMSGEXPORT int coGRActivatedViewpointMsg::getViewpointId()
{
    return id_;
}
