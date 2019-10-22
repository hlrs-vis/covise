/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coGRSetDocPositionMsg.h"
#include <iostream>
#include <sstream>

using namespace std;
using namespace grmsg;

GRMSGEXPORT coGRSetDocPositionMsg::coGRSetDocPositionMsg(const char *document_name, float x, float y, float z)
    : coGRDocMsg(SET_DOCUMENT_POSITION, document_name)
{
    pos_[0] = x;
    pos_[1] = y;
    pos_[2] = z;
    is_valid_ = 1;
    for (int i = 0; i < 3; i++)
    {
        ostringstream stream;
        stream << pos_[i];
        addToken(stream.str().c_str());
    }
}

GRMSGEXPORT coGRSetDocPositionMsg::coGRSetDocPositionMsg(const char *msg)
    : coGRDocMsg(msg)
{
    vector<string> tok = getAllTokens();

    if (tok.size() == 3)
    {
        for (int i = 0; i < 3; i++)
        {
            istringstream stream(tok[i]);
            stream >> pos_[i];
        }
        is_valid_ = 1;
    }
    else
    {
        is_valid_ = 0;
    }
}

GRMSGEXPORT void coGRSetDocPositionMsg::getPosition(float &x, float &y, float &z)
{
    x = pos_[0];
    y = pos_[1];
    z = pos_[2];
}
