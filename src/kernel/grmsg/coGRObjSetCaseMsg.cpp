/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cstring>
#include "coGRObjSetCaseMsg.h"

using namespace grmsg;

GRMSGEXPORT coGRObjSetCaseMsg::coGRObjSetCaseMsg(Mtype type, const char *obj_name, const char *caseName)
    : coGRObjMsg(type, obj_name)
{
    caseName_ = NULL;

    if (caseName)
    {
        caseName_ = new char[strlen(caseName) + 1];
        strcpy(caseName_, caseName);
        addToken(caseName);
        is_valid_ = 1;
    }
    else
    {
        is_valid_ = 0;
    }
}

GRMSGEXPORT coGRObjSetCaseMsg::coGRObjSetCaseMsg(const char *msg)
    : coGRObjMsg(msg)
{
    caseName_ = NULL;
    vector<string> tok = getAllTokens();

    if (!tok[0].empty())
    {
        caseName_ = new char[strlen(tok[0].c_str()) + 1];
        strcpy(caseName_, tok[0].c_str());
    }
    else
    {
        caseName_ = NULL;
        is_valid_ = 0;
    }
}

GRMSGEXPORT coGRObjSetCaseMsg::~coGRObjSetCaseMsg()
{
    if (caseName_)
        delete[] caseName_;
}
