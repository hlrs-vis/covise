/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cstring>
#include "coGRObjMsg.h"

using namespace grmsg;

GRMSGEXPORT coGRObjMsg::coGRObjMsg(const char *msg)
    : coGRMsg(msg)
{
    string objName = extractFirstToken();
    obj_name_ = new char[objName.size() + 1];
    strcpy(obj_name_, objName.c_str());
}

GRMSGEXPORT coGRObjMsg::coGRObjMsg(coGRMsg::Mtype type, const char *obj_name)
    : coGRMsg(type)
{
    obj_name_ = new char[strlen(obj_name) + 1];
    strcpy(obj_name_, obj_name);
    addToken(obj_name);
}

GRMSGEXPORT coGRObjMsg::~coGRObjMsg()
{
    if (obj_name_)
        delete[] obj_name_;
}

GRMSGEXPORT const char *coGRObjMsg::getObjName()
{
    return obj_name_;
}
