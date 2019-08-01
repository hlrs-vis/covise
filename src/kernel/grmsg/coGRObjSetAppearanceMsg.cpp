/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cstring>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include "coGRObjSetAppearanceMsg.h"

using namespace std;

namespace grmsg
{

coGRObjSetAppearanceMsg::coGRObjSetAppearanceMsg(const char *obj_name, const char *scopeName, float r, float g, float b, float a)
    : coGRObjMsg(coGRMsg::SET_APPEARANCE, obj_name)
{
    scopeName_ = NULL;
    is_valid_ = 1;
    char str[1024];

    if (scopeName)
    {
        scopeName_ = new char[strlen(scopeName) + 1];
        strcpy(scopeName_, scopeName);
        addToken(scopeName_);
    }
    else
    {
        is_valid_ = 0;
    }

    r_ = r;
    sprintf(str, "%f", r);
    addToken(str);

    g_ = g;
    sprintf(str, "%f", g);
    addToken(str);

    b_ = b;
    sprintf(str, "%f", b);
    addToken(str);

    a_ = a;
    sprintf(str, "%f", a);
    addToken(str);
}

coGRObjSetAppearanceMsg::coGRObjSetAppearanceMsg(const char *msg)
    : coGRObjMsg(msg)
{
    scopeName_ = NULL;
    vector<string> tok = getAllTokens();

    scopeName_ = new char[strlen(tok[0].c_str()) + 1];
    strcpy(scopeName_, tok[0].c_str());

    if (!tok[1].empty())
    {
        string tmp = tok[1];
        sscanf(tmp.c_str(), "%f", &r_);
        is_valid_ = is_valid_ & 1;
    }
    else
    {
        is_valid_ = 0;
    }

    if (!tok[2].empty())
    {
        string tmp = tok[2];
        sscanf(tmp.c_str(), "%f", &g_);
        is_valid_ = is_valid_ & 1;
    }
    else
    {
        is_valid_ = 0;
    }

    if (!tok[3].empty())
    {
        string tmp = tok[3];
        sscanf(tmp.c_str(), "%f", &b_);
        is_valid_ = is_valid_ & 1;
    }
    else
    {
        is_valid_ = 0;
    }

    if (!tok[4].empty())
    {
        string tmp = tok[4];
        sscanf(tmp.c_str(), "%f", &a_);
        is_valid_ = is_valid_ & 1;
    }
    else
    {
        is_valid_ = 0;
    }
}

const char *coGRObjSetAppearanceMsg::getScopeName()
{
    return scopeName_;
}

float coGRObjSetAppearanceMsg::getR()
{
    return r_;
}

float coGRObjSetAppearanceMsg::getG()
{
    return g_;
}

float coGRObjSetAppearanceMsg::getB()
{
    return b_;
}

float coGRObjSetAppearanceMsg::getA()
{
    return a_;
}

coGRObjSetAppearanceMsg::~coGRObjSetAppearanceMsg()
{
    if (scopeName_)
        delete[] scopeName_;
}
}
