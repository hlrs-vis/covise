/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coGRToggleFlymodeMsg.h"
#include <iostream>
#include <sstream>
#include <cstdio>

using namespace grmsg;

GRMSGEXPORT coGRToggleFlymodeMsg::coGRToggleFlymodeMsg(const char *msg)
    : coGRMsg(msg)
{
    sscanf(extractFirstToken().c_str(), "%d", &mode_);
}

GRMSGEXPORT coGRToggleFlymodeMsg::coGRToggleFlymodeMsg(int mode)
    : coGRMsg(FLYMODE_TOGGLE)
{
    mode_ = mode;
    ostringstream stream;
    stream << mode;
    addToken(stream.str().c_str());
}

GRMSGEXPORT int coGRToggleFlymodeMsg::getMode()
{
    return mode_;
}
