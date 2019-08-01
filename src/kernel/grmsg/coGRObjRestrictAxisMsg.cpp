/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cstring>
#include "coGRObjRestrictAxisMsg.h"

using namespace std;
using namespace grmsg;

GRMSGEXPORT coGRObjRestrictAxisMsg::coGRObjRestrictAxisMsg(Mtype type, const char *obj_name, const char *axisName)
    : coGRObjMsg(type, obj_name)
{
    axisName_ = NULL;

    if (axisName)
    {
        axisName_ = new char[strlen(axisName) + 1];
        strcpy(axisName_, axisName);

        addToken(axisName);

        is_valid_ = 1;
    }
    else
    {
        is_valid_ = 0;
    }
}

GRMSGEXPORT coGRObjRestrictAxisMsg::coGRObjRestrictAxisMsg(const char *msg)
    : coGRObjMsg(msg)
{
    axisName_ = NULL;
    vector<string> tok = getAllTokens();

    if (!tok[0].empty())
    {
        axisName_ = new char[strlen(tok[0].c_str()) + 1];
        strcpy(axisName_, tok[0].c_str());
    }
    else
    {
        axisName_ = NULL;
        is_valid_ = 0;
    }
}

coGRObjRestrictAxisMsg::~coGRObjRestrictAxisMsg()
{
    if (axisName_)
        delete[] axisName_;
}
