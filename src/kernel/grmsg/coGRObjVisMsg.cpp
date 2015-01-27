/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coGRObjVisMsg.h"

using namespace grmsg;

GRMSGEXPORT coGRObjVisMsg::coGRObjVisMsg(Mtype type, const char *obj_name, int is_visible)
    : coGRObjMsg(type, obj_name)
{
    is_visible_ = is_visible;

    if (is_visible == 1)
    {
        addToken("1");
        is_valid_ = 1;
    }
    else if (is_visible == 0)
    {
        addToken("0");
        is_valid_ = 1;
    }
    else
    {
        is_valid_ = 0;
    }
}

GRMSGEXPORT coGRObjVisMsg::coGRObjVisMsg(const char *msg)
    : coGRObjMsg(msg)
{
    string content = getAllTokens()[0];

    if (!content.empty() && content[0] == '1' && content[1] == 0)
    {
        is_visible_ = 1;
        is_valid_ = is_valid_ & 1;
    }
    else if (!content.empty() && content[0] == '0' && content[1] == 0)
    {
        is_visible_ = 0;
        is_valid_ = is_valid_ & 1;
    }
    else
    {
        is_visible_ = 0;
        is_valid_ = 0;
    }
}
