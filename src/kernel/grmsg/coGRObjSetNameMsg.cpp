/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cstring>
#include "coGRObjSetNameMsg.h"

using namespace std;
using namespace grmsg;

GRMSGEXPORT coGRObjSetNameMsg::coGRObjSetNameMsg(Mtype type, const char *obj_name, const char *newName)
    : coGRObjMsg(type, obj_name)
{
    newName_ = NULL;

    if (newName)
    {
        newName_ = new char[strlen(newName) + 1];
        strcpy(newName_, newName);
        addToken(newName);
        is_valid_ = 1;
    }
    else
    {
        is_valid_ = 0;
    }
}

GRMSGEXPORT coGRObjSetNameMsg::coGRObjSetNameMsg(const char *msg)
    : coGRObjMsg(msg)
{
    newName_ = NULL;
    vector<string> tok = getAllTokens();

    if (!tok[0].empty())
    {
        newName_ = new char[strlen(tok[0].c_str()) + 1];
        strcpy(newName_, tok[0].c_str());
    }
    else
    {
        newName_ = NULL;
        is_valid_ = 0;
    }
}

GRMSGEXPORT coGRObjSetNameMsg::~coGRObjSetNameMsg()
{
    if (newName_)
        delete[] newName_;
}
