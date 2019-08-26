/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cstring>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include "coGRObjAddChildMsg.h"

#define NO_CHILD "__NO_CHILD__"

using namespace std;

namespace grmsg
{

coGRObjAddChildMsg::coGRObjAddChildMsg(const char *obj_name, const char *child_obj_name, int remove)
    : coGRObjMsg(coGRMsg::ADD_CHILD_OBJECT, obj_name)
{
    childObjName_ = NULL;
    remove_ = remove;

    if (child_obj_name)
    {
        childObjName_ = new char[strlen(child_obj_name) + 1];
        strcpy(childObjName_, child_obj_name);
        addToken(childObjName_);
    }
    else
    {
        addToken(NO_CHILD);
    }
    if (remove_ == 0 || remove_ == 1)
    {
        ostringstream stream;
        stream << remove_;
        addToken(stream.str().c_str());
        is_valid_ = 1;
    }
    else
        is_valid_ = 0;
}

coGRObjAddChildMsg::coGRObjAddChildMsg(const char *msg)
    : coGRObjMsg(msg)
{
    childObjName_ = NULL;
    vector<string> tok = getAllTokens();

    if (!tok[0].empty())
    {
        if (string(NO_CHILD) != tok[0])
        {
            childObjName_ = new char[strlen(tok[0].c_str()) + 1];
            strcpy(childObjName_, tok[0].c_str());
        }
    }
    if (!tok[1].empty())
    {
        string remove = tok[1];
        sscanf(remove.c_str(), "%d", &remove_);
    }
    else
    {
        remove_ = 0;
    }
}

const char *coGRObjAddChildMsg::getChildObjName()
{
    return childObjName_;
}

int coGRObjAddChildMsg::getRemove()
{
    return remove_;
}

coGRObjAddChildMsg::~coGRObjAddChildMsg()
{
    if (childObjName_)
        delete[] childObjName_;
}
}
