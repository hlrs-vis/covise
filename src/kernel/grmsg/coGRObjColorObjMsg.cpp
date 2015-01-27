/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coGRObjColorObjMsg.h"
#include <cstdio>

using namespace grmsg;

GRMSGEXPORT coGRObjColorObjMsg::coGRObjColorObjMsg(Mtype type, const char *obj_name, int r, int g, int b)
    : coGRObjMsg(type, obj_name)
{
    r_ = r;
    g_ = g;
    b_ = b;

    char str[1024];

    sprintf(str, "%d", r);
    addToken(str);

    sprintf(str, "%d", g);
    addToken(str);

    sprintf(str, "%d", b);
    addToken(str);

    is_valid_ = 1;
}

GRMSGEXPORT coGRObjColorObjMsg::coGRObjColorObjMsg(const char *msg)
    : coGRObjMsg(msg)
{
    vector<string> tok = getAllTokens();

    if (!tok[0].empty())
    {
        string r = tok[0];
        sscanf(r.c_str(), "%d", &r_);
    }
    else
    {
        is_valid_ = 0;
    }

    if (!tok[1].empty())
    {
        string g = tok[1];
        sscanf(g.c_str(), "%d", &g_);
    }
    else
    {
        is_valid_ = 0;
    }

    if (!tok[2].empty())
    {
        string b = tok[2];
        sscanf(b.c_str(), "%d", &b_);
    }
    else
    {
        is_valid_ = 0;
    }
}
