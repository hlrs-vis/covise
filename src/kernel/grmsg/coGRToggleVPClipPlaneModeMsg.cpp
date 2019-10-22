/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coGRToggleVPClipPlaneModeMsg.h"
#include <iostream>
#include <sstream>
#include <cstdio>

using namespace std;
using namespace grmsg;

GRMSGEXPORT coGRToggleVPClipPlaneModeMsg::coGRToggleVPClipPlaneModeMsg(const char *msg)
    : coGRMsg(msg)
{
    sscanf(extractFirstToken().c_str(), "%d", &mode_);
}

GRMSGEXPORT coGRToggleVPClipPlaneModeMsg::coGRToggleVPClipPlaneModeMsg(int mode)
    : coGRMsg(VPCLIPPLANEMODE_TOGGLE)
{
    mode_ = mode;
    ostringstream stream;
    stream << mode;
    addToken(stream.str().c_str());
}

GRMSGEXPORT int coGRToggleVPClipPlaneModeMsg::getMode()
{
    return mode_;
}
