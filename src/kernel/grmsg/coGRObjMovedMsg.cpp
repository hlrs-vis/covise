/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coGRObjMovedMsg.h"
#include <string>
#include <sstream>
#include <cstdio>

using namespace std;

namespace grmsg
{

GRMSGEXPORT coGRObjMovedMsg::coGRObjMovedMsg(const char *obj_name, float transX, float transY, float transZ, float rotX, float rotY, float rotZ, float rotAngle)
    : coGRObjMsg(OBJECT_TRANSFORMED, obj_name)
{
    initClass(transX, transY, transZ, rotX, rotY, rotZ, rotAngle);
}

coGRObjMovedMsg::coGRObjMovedMsg(coGRMsg::Mtype type, const char *obj_name, float transX, float transY, float transZ, float rotX, float rotY, float rotZ, float rotAngle)
    : coGRObjMsg(type, obj_name)
{
    initClass(transX, transY, transZ, rotX, rotY, rotZ, rotAngle);
}

GRMSGEXPORT coGRObjMovedMsg::coGRObjMovedMsg(const char *msg)
    : coGRObjMsg(msg)
{
    vector<string> tok = getAllTokens();

    if (!tok[0].empty())
    {
        string transx = tok[0];
        sscanf(transx.c_str(), "%f", &transX_);
        is_valid_ = is_valid_ & 1;
    }
    else
    {
        is_valid_ = 0;
    }

    if (!tok[1].empty())
    {
        string transy = tok[1];
        sscanf(transy.c_str(), "%f", &transY_);
        is_valid_ = is_valid_ & 1;
    }
    else
    {
        is_valid_ = 0;
    }

    if (!tok[2].empty())
    {
        string transz = tok[2];
        sscanf(transz.c_str(), "%f", &transZ_);
        is_valid_ = is_valid_ & 1;
    }
    else
    {
        is_valid_ = 0;
    }

    if (!tok[3].empty())
    {
        string rotx = tok[3];
        sscanf(rotx.c_str(), "%f", &rotX_);
        is_valid_ = is_valid_ & 1;
    }
    else
    {
        is_valid_ = 0;
    }

    if (!tok[4].empty())
    {
        string roty = tok[4];
        sscanf(roty.c_str(), "%f", &rotY_);
        is_valid_ = is_valid_ & 1;
    }
    else
    {
        is_valid_ = 0;
    }

    if (!tok[5].empty())
    {
        string rotz = tok[5];
        sscanf(rotz.c_str(), "%f", &rotZ_);
        is_valid_ = is_valid_ & 1;
    }
    else
    {
        is_valid_ = 0;
    }

    if (!tok[6].empty())
    {
        string rotangle = tok[6];
        sscanf(rotangle.c_str(), "%f", &rotAngle_);
        is_valid_ = is_valid_ & 1;
    }
    else
    {
        is_valid_ = 0;
    }
}

GRMSGEXPORT void coGRObjMovedMsg::initClass(float transX, float transY, float transZ, float rotX, float rotY, float rotZ, float rotAngle)
{
    transX_ = transX;
    transY_ = transY;
    transZ_ = transZ;
    rotX_ = rotX;
    rotY_ = rotY;
    rotZ_ = rotZ;
    rotAngle_ = rotAngle;

    char str[1024];

    sprintf(str, "%f", transX_);
    addToken(str);
    sprintf(str, "%f", transY_);
    addToken(str);
    sprintf(str, "%f", transZ_);
    addToken(str);
    sprintf(str, "%f", rotX_);
    addToken(str);
    sprintf(str, "%f", rotY_);
    addToken(str);
    sprintf(str, "%f", rotZ_);
    addToken(str);
    sprintf(str, "%f", rotAngle_);
    addToken(str);

    is_valid_ = 1;
}
}
