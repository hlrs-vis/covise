/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coGRObjGeometryMsg.h"
#include <string>
#include <sstream>
#include <cstdio>

using namespace std;

namespace grmsg
{

GRMSGEXPORT coGRObjGeometryMsg::coGRObjGeometryMsg(const char *obj_name, float width, float height, float length)
    : coGRObjMsg(GEOMETRY_OBJECT, obj_name)
{
    width_ = width;
    height_ = height;
    length_ = length;

    char str[1024];

    sprintf(str, "%f", width_);
    addToken(str);
    sprintf(str, "%f", height_);
    addToken(str);
    sprintf(str, "%f", length_);
    addToken(str);

    is_valid_ = 1;
}

GRMSGEXPORT coGRObjGeometryMsg::coGRObjGeometryMsg(const char *msg)
    : coGRObjMsg(msg)
{
    vector<string> tok = getAllTokens();

    if (!tok[0].empty())
    {
        string width = tok[0];
        sscanf(width.c_str(), "%f", &width_);
        is_valid_ = is_valid_ & 1;
    }
    else
    {
        is_valid_ = 0;
    }

    if (!tok[1].empty())
    {
        string height = tok[1];
        sscanf(height.c_str(), "%f", &height_);
        is_valid_ = is_valid_ & 1;
    }
    else
    {
        is_valid_ = 0;
    }

    if (!tok[2].empty())
    {
        string length = tok[2];
        sscanf(length.c_str(), "%f", &length_);
        is_valid_ = is_valid_ & 1;
    }
    else
    {
        is_valid_ = 0;
    }
}
}
