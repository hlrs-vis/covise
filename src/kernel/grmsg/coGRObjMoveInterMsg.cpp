/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cstring>
#include <cstdio>
#include "coGRObjMoveInterMsg.h"

using namespace grmsg;

GRMSGEXPORT coGRObjMoveInterMsg::coGRObjMoveInterMsg(Mtype type, const char *obj_name, const char *interName, float x, float y, float z)
    : coGRObjMsg(type, obj_name)
{
    x_ = x;
    x_ = y;
    x_ = z;
    char str[1024];

    interactorName_ = NULL;

    if (interName)
    {
        interactorName_ = new char[strlen(interName) + 1];
        strcpy(interactorName_, interName);

        addToken(interName);

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

GRMSGEXPORT coGRObjMoveInterMsg::coGRObjMoveInterMsg(const char *msg)
    : coGRObjMsg(msg)
{
    interactorName_ = NULL;
    vector<string> tok = getAllTokens();

    if (!tok[0].empty())
    {
        interactorName_ = new char[strlen(tok[0].c_str()) + 1];
        strcpy(interactorName_, tok[0].c_str());
    }
    else
    {
        interactorName_ = NULL;
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

coGRObjMoveInterMsg::~coGRObjMoveInterMsg()
{
    if (interactorName_)
        delete[] interactorName_;
}
