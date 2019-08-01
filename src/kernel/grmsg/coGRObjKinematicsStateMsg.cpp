/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cstring>
#include <iostream>
#include <sstream>
#include "coGRObjKinematicsStateMsg.h"

using namespace std;

namespace grmsg
{

coGRObjKinematicsStateMsg::coGRObjKinematicsStateMsg(const char *obj_name, const char *state)
    : coGRObjMsg(coGRMsg::KINEMATICS_STATE, obj_name)
{
    if (state)
    {
        state_ = new char[strlen(state) + 1];
        strcpy(state_, state);
        addToken(state_);
        is_valid_ = 1;
    }
    else
    {
        is_valid_ = 0;
    }
}

coGRObjKinematicsStateMsg::coGRObjKinematicsStateMsg(const char *msg)
    : coGRObjMsg(msg)
{
    vector<string> tok = getAllTokens();

    state_ = new char[strlen(tok[0].c_str()) + 1];
    strcpy(state_, tok[0].c_str());
}

const char *coGRObjKinematicsStateMsg::getState()
{
    return state_;
}

coGRObjKinematicsStateMsg::~coGRObjKinematicsStateMsg()
{
    if (state_)
        delete[] state_;
}
}
