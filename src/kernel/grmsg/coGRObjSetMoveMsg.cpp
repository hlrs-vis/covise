/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coGRObjSetMoveMsg.h"
#include <iostream>
#include <sstream>
#include <cstdio>

using namespace grmsg;

GRMSGEXPORT coGRObjSetMoveMsg::coGRObjSetMoveMsg(const char *msg)
    : coGRObjMsg(msg)
{
    unsigned short dummy;
    sscanf(extractFirstToken().c_str(), "%hu", &dummy);
    isMoveable_ = dummy != 0;
}

GRMSGEXPORT coGRObjSetMoveMsg::coGRObjSetMoveMsg(Mtype type, const char *obj_name, bool moveable)
    : coGRObjMsg(type, obj_name)
{
    isMoveable_ = moveable;
    ostringstream stream;
    stream << isMoveable_;
    addToken(stream.str().c_str());
}

GRMSGEXPORT bool coGRObjSetMoveMsg::isMoveable()
{
    return isMoveable_;
}
