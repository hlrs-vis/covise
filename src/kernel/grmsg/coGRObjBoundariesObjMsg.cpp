/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coGRObjBoundariesObjMsg.h"
#include <cstdio>

using namespace grmsg;

GRMSGEXPORT coGRObjBoundariesObjMsg::coGRObjBoundariesObjMsg(Mtype type, const char *obj_name, const char *boundariesName, float front, float back, float left, float right, float top, float bottom)
    : coGRObjMsg(type, obj_name)
{
    front_ = front;
    back_ = back;
    left_ = left;
    right_ = right;
    top_ = top;
    bottom_ = bottom;
    char str[1024];

    if (boundariesName)
    {
        addToken(boundariesName);

        sprintf(str, "%f", front);
        addToken(str);

        sprintf(str, "%f", back);
        addToken(str);

        sprintf(str, "%f", left);
        addToken(str);

        sprintf(str, "%f", right);
        addToken(str);

        sprintf(str, "%f", top);
        addToken(str);

        sprintf(str, "%f", bottom);
        addToken(str);

        is_valid_ = 1;
    }
    else
    {
        is_valid_ = 0;
    }
}

GRMSGEXPORT coGRObjBoundariesObjMsg::coGRObjBoundariesObjMsg(const char *msg)
    : coGRObjMsg(msg)
{
    vector<string> tok = getAllTokens();

    if (!tok[0].empty())
    {
        boundariesName_ = tok[0].c_str();
        is_valid_ = is_valid_ & 1;
    }
    else
    {
        boundariesName_ = NULL;
        is_valid_ = 0;
    }

    if (!tok[1].empty())
    {
        string front = tok[1];
        sscanf(front.c_str(), "%f", &front_);
    }
    else
    {
        is_valid_ = 0;
    }

    if (!tok[2].empty())
    {
        string back = tok[2];
        sscanf(back.c_str(), "%f", &back_);
    }
    else
    {
        is_valid_ = 0;
    }

    if (!tok[3].empty())
    {
        string left = tok[3];
        sscanf(left.c_str(), "%f", &left_);
    }
    else
    {
        is_valid_ = 0;
    }

    if (!tok[4].empty())
    {
        string right = tok[4];
        sscanf(right.c_str(), "%f", &right_);
    }
    else
    {
        is_valid_ = 0;
    }

    if (!tok[5].empty())
    {
        string top = tok[5];
        sscanf(top.c_str(), "%f", &top_);
    }
    else
    {
        is_valid_ = 0;
    }

    if (!tok[6].empty())
    {
        string bottom = tok[6];
        sscanf(bottom.c_str(), "%f", &bottom_);
    }
    else
    {
        is_valid_ = 0;
    }
}
