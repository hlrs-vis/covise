/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coGRSetTimestepMsg.h"
#include <iostream>
#include <sstream>
#include <cstdio>

using namespace std;
using namespace grmsg;

GRMSGEXPORT coGRSetTimestepMsg::coGRSetTimestepMsg(const char *msg)
    : coGRMsg(msg)
{
    sscanf(extractFirstToken().c_str(), "%d %d", &step_, &maxSteps_);
}

GRMSGEXPORT coGRSetTimestepMsg::coGRSetTimestepMsg(int step, int maxSteps)
    : coGRMsg(ANIMATION_TIMESTEP)
{
    step_ = step;
    maxSteps_ = maxSteps;
    ostringstream stream;
    stream << step_ << " " << maxSteps_;
    addToken(stream.str().c_str());
}

GRMSGEXPORT int coGRSetTimestepMsg::getActualTimeStep()
{
    return step_;
}

GRMSGEXPORT int coGRSetTimestepMsg::getNumTimeSteps()
{
    return maxSteps_;
}
