/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coGRObjSelectMsg.h"

using namespace std;

namespace grmsg
{

GRMSGEXPORT coGRObjSelectMsg::coGRObjSelectMsg(const char *obj_name, int select)
    : coGRObjMsg(SELECT_OBJECT, obj_name)
{
    is_selected_ = select;

    if (is_selected_ == 1)
    {
        addToken("1");
        is_valid_ = 1;
    }
    else if (is_selected_ == 0)
    {
        addToken("0");
        is_valid_ = 1;
    }
    else
    {
        is_valid_ = 0;
    }
}

GRMSGEXPORT coGRObjSelectMsg::coGRObjSelectMsg(const char *msg)
    : coGRObjMsg(msg)
{
    string content = getAllTokens()[0];

    if (!content.empty() && content[0] == '1' && content[1] == 0)
    {
        is_selected_ = 1;
        is_valid_ = is_valid_ & 1;
    }
    else if (!content.empty() && content[0] == '0' && content[1] == 0)
    {
        is_selected_ = 0;
        is_valid_ = is_valid_ & 1;
    }
    else
    {
        is_selected_ = 0;
        is_valid_ = 0;
    }
}
}
