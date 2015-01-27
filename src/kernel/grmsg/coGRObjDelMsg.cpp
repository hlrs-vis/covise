/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coGRObjDelMsg.h"

namespace grmsg
{

GRMSGEXPORT coGRObjDelMsg::coGRObjDelMsg(const char *obj_name, int del)
    : coGRObjMsg(DELETE_OBJECT, obj_name)
{
    if (del > 0)
        is_valid_ = 1;
}

GRMSGEXPORT coGRObjDelMsg::coGRObjDelMsg(const char *msg)
    : coGRObjMsg(msg)
{
    is_valid_ = 1;
}
}
