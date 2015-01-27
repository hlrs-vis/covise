/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cstdio>
#include <cstring>
#include "coGRObjMoveObjMsg.h"

using namespace grmsg;

GRMSGEXPORT coGRObjMoveObjMsg::coGRObjMoveObjMsg(const char *obj_name, const char *moveName, float x, float y, float z)
    : coGRObjMsg(MOVE_OBJECT, obj_name)
{
    x_ = x;
    y_ = y;
    z_ = z;
    char str[1024];

    if (moveName)
    {
        moveName_ = new char[strlen(moveName) + 1];
        strcpy(moveName_, moveName);
        addToken(moveName);

        sprintf(str, "%f", x);
        addToken(str);

        sprintf(str, "%f", y);
        addToken(str);

        sprintf(str, "%f", z);
        addToken(str);

        is_valid_ = 1;
    }
    else
    {
        is_valid_ = 0;
    }
}

GRMSGEXPORT coGRObjMoveObjMsg::coGRObjMoveObjMsg(const char *msg)
    : coGRObjMsg(msg)
{
    vector<string> tok = getAllTokens();

    if (!tok[0].empty())
    {
        moveName_ = new char[tok[0].size() + 1];
        strcpy(moveName_, tok[0].c_str());
    }
    else
    {
        moveName_ = NULL;
        is_valid_ = 0;
    }

    if (!tok[1].empty())
    {
        string x = tok[1];
        sscanf(x.c_str(), "%f", &x_);
    }
    else
    {
        is_valid_ = 0;
    }

    if (!tok[2].empty())
    {
        string y = tok[2];
        sscanf(y.c_str(), "%f", &y_);
    }
    else
    {
        is_valid_ = 0;
    }

    if (!tok[3].empty())
    {
        string z = tok[3];
        sscanf(z.c_str(), "%f", &z_);
    }
    else
    {
        is_valid_ = 0;
    }
}
