/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coGRObjSetMoveSelectedMsg.h"
#include <iostream>
#include <sstream>
#include <cstdio>

using namespace std;
using namespace grmsg;

GRMSGEXPORT coGRObjSetMoveSelectedMsg::coGRObjSetMoveSelectedMsg(const char *msg)
    : coGRObjMsg(msg)
{
    unsigned short dummy;
    sscanf(extractFirstToken().c_str(), "%hu", &dummy);
    isSelected_ = dummy != 0;
}

GRMSGEXPORT coGRObjSetMoveSelectedMsg::coGRObjSetMoveSelectedMsg(Mtype type, const char *obj_name, bool selected)
    : coGRObjMsg(type, obj_name)
{
    isSelected_ = selected;
    ostringstream stream;
    stream << isSelected_;
    addToken(stream.str().c_str());
}

GRMSGEXPORT bool coGRObjSetMoveSelectedMsg::isSelected()
{
    return isSelected_;
}
