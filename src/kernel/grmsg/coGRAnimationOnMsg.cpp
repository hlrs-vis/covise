/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coGRAnimationOnMsg.h"
#include <iostream>
#include <sstream>
#include <cstdio>

using namespace grmsg;
using namespace std;

GRMSGEXPORT coGRAnimationOnMsg::coGRAnimationOnMsg(const char *msg)
    : coGRMsg(msg)
{
    sscanf(extractFirstToken().c_str(), "%d", &mode_);
}

GRMSGEXPORT coGRAnimationOnMsg::coGRAnimationOnMsg(int mode)
    : coGRMsg(ANIMATION_ON)
{
    mode_ = mode;
    ostringstream stream;
    stream << mode;
    addToken(stream.str().c_str());
}

GRMSGEXPORT int coGRAnimationOnMsg::getMode()
{
    return mode_;
}
