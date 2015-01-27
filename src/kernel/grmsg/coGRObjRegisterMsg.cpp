/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cstring>
#include <iostream>
#include <sstream>
#include <cstdio>
#include "coGRObjRegisterMsg.h"

using namespace grmsg;

#define NO_PARENT "__NO_PARENT__"

coGRObjRegisterMsg::coGRObjRegisterMsg(const char *obj_name, const char *parent_obj_name, int unregister)
    : coGRObjMsg(coGRMsg::REGISTER, obj_name)
{
    parentObjName_ = NULL;
    unregister_ = unregister;

    if (parent_obj_name)
    {
        parentObjName_ = new char[strlen(parent_obj_name) + 1];
        strcpy(parentObjName_, parent_obj_name);
        addToken(parentObjName_);
    }
    else
    {
        addToken(NO_PARENT);
    }
    if (unregister_ == 0 || unregister_ == 1)
    {
        ostringstream stream;
        stream << unregister_;
        addToken(stream.str().c_str());
        is_valid_ = 1;
    }
    else
        is_valid_ = 0;
}

coGRObjRegisterMsg::coGRObjRegisterMsg(const char *msg)
    : coGRObjMsg(msg)
{
    parentObjName_ = NULL;
    vector<string> tok = getAllTokens();

    if (!tok[0].empty())
    {
        if (string(NO_PARENT) != tok[0])
        {
            parentObjName_ = new char[strlen(tok[0].c_str()) + 1];
            strcpy(parentObjName_, tok[0].c_str());
        }
    }
    if (!tok[1].empty())
    {
        string unregister = tok[1];
        sscanf(unregister.c_str(), "%d", &unregister_);
    }
    else
    {
        unregister_ = 0;
    }
}

const char *coGRObjRegisterMsg::getParentObjName()
{
    return parentObjName_;
}

int coGRObjRegisterMsg::getUnregister()
{
    return unregister_;
}

coGRObjRegisterMsg::~coGRObjRegisterMsg()
{
    if (parentObjName_)
        delete[] parentObjName_;
}
