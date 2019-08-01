/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coGRSetAnimationSpeedMsg.h"
#include <iostream>
#include <sstream>
#include <cstdio>

using namespace std;
using namespace grmsg;

GRMSGEXPORT coGRSetAnimationSpeedMsg::coGRSetAnimationSpeedMsg(const char *msg)
    : coGRMsg(msg)
{
    sscanf(extractFirstToken().c_str(), "%f %f %f", &speed_, &speedMin_, &speedMax_);
}

GRMSGEXPORT coGRSetAnimationSpeedMsg::coGRSetAnimationSpeedMsg(float speed, float min, float max)
    : coGRMsg(ANIMATION_SPEED)
{
    speed_ = speed;
    speedMin_ = min;
    speedMax_ = max;
    ostringstream stream;
    stream << speed << " " << speedMin_ << " " << speedMax_;
    addToken(stream.str().c_str());
}

GRMSGEXPORT float coGRSetAnimationSpeedMsg::getAnimationSpeed()
{
    return speed_;
}

GRMSGEXPORT float coGRSetAnimationSpeedMsg::getAnimationSpeedMin()
{
    return speedMin_;
}

GRMSGEXPORT float coGRSetAnimationSpeedMsg::getAnimationSpeedMax()
{
    return speedMax_;
}
