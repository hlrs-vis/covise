/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coGRTurnTableAnimationMsg.h"
#include <iostream>
#include <sstream>
#include <cstdio>

namespace grmsg
{

GRMSGEXPORT coGRTurnTableAnimationMsg::coGRTurnTableAnimationMsg(const char *msg)
    : coGRMsg(msg)
{
    sscanf(extractFirstToken().c_str(), "%f", &time_);
}

GRMSGEXPORT coGRTurnTableAnimationMsg::coGRTurnTableAnimationMsg(float time)
    : coGRMsg(TURNTABLE_ANIMATION)
{
    time_ = time;
    ostringstream stream;
    stream << time_;
    addToken(stream.str().c_str());
}
}
