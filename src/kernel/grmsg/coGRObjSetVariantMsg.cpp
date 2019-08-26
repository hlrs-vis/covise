/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cstring>
#include <iostream>
#include <sstream>
#include "coGRObjSetVariantMsg.h"

using namespace std;

namespace grmsg
{

coGRObjSetVariantMsg::coGRObjSetVariantMsg(const char *obj_name, const char *groupName, const char *variantName)
    : coGRObjMsg(coGRMsg::SET_VARIANT, obj_name)
{
    groupName_ = NULL;
    variantName_ = NULL;
    is_valid_ = 1;

    if (groupName)
    {
        groupName_ = new char[strlen(groupName) + 1];
        strcpy(groupName_, groupName);
        addToken(groupName_);
    }
    else
    {
        is_valid_ = 0;
    }

    if (variantName)
    {
        variantName_ = new char[strlen(variantName) + 1];
        strcpy(variantName_, variantName);
        addToken(variantName_);
    }
    else
    {
        is_valid_ = 0;
    }
}

coGRObjSetVariantMsg::coGRObjSetVariantMsg(const char *msg)
    : coGRObjMsg(msg)
{
    groupName_ = NULL;
    variantName_ = NULL;
    vector<string> tok = getAllTokens();

    if (!tok[0].empty())
    {
        groupName_ = new char[strlen(tok[0].c_str()) + 1];
        strcpy(groupName_, tok[0].c_str());
    }
    if (!tok[1].empty())
    {
        variantName_ = new char[strlen(tok[1].c_str()) + 1];
        strcpy(variantName_, tok[1].c_str());
    }
}

const char *coGRObjSetVariantMsg::getGroupName()
{
    return groupName_;
}

const char *coGRObjSetVariantMsg::getVariantName()
{
    return variantName_;
}

coGRObjSetVariantMsg::~coGRObjSetVariantMsg()
{
    if (groupName_)
        delete[] groupName_;
    if (variantName_)
        delete[] variantName_;
}
}
