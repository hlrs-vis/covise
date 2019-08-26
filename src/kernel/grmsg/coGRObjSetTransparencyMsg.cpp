/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coGRObjSetTransparencyMsg.h"
#include <cstdio>

using namespace std;
using namespace grmsg;

GRMSGEXPORT coGRObjSetTransparencyMsg::coGRObjSetTransparencyMsg(Mtype type, const char *obj_name, float trans)
    : coGRObjMsg(type, obj_name)
{
    trans_ = trans;

    char str[1024];

    sprintf(str, "%f", trans);
    addToken(str);

    is_valid_ = 1;
}

GRMSGEXPORT coGRObjSetTransparencyMsg::coGRObjSetTransparencyMsg(const char *msg)
    : coGRObjMsg(msg)
{
    vector<string> tok = getAllTokens();

    if (!tok[0].empty())
    {
        string trans = tok[0];
        sscanf(trans.c_str(), "%f", &trans_);
        is_valid_ = is_valid_ & 1;
    }
    else
    {
        is_valid_ = 0;
    }
}
