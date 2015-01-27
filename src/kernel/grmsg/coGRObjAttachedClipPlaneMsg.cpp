/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coGRObjAttachedClipPlaneMsg.h"
#include <cstdio>

using namespace grmsg;

GRMSGEXPORT coGRObjAttachedClipPlaneMsg::coGRObjAttachedClipPlaneMsg(Mtype type, const char *obj_name, int index, float offset, bool flip)
    : coGRObjMsg(type, obj_name)
{
    index_ = index;
    offset_ = offset;
    flip_ = flip;

    char token[100];
    sprintf(token, "%d", index_);
    addToken(token);
    sprintf(token, "%f", offset_);
    addToken(token);
    addToken(flip_ ? "1" : "0");

    is_valid_ = 1;
}

GRMSGEXPORT coGRObjAttachedClipPlaneMsg::coGRObjAttachedClipPlaneMsg(const char *msg)
    : coGRObjMsg(msg)
{
    vector<string> tok = getAllTokens();

    if (tok.size() == 3)
    {
        is_valid_ = 1;
        if (sscanf(tok[0].c_str(), "%d", &index_) != 1)
        {
            is_valid_ = 0;
        }
        if (sscanf(tok[1].c_str(), "%f", &offset_) != 1)
        {
            is_valid_ = 0;
        }
        flip_ = (tok[2] == "1");
    }
    else
    {
        is_valid_ = 0;
    }
}
