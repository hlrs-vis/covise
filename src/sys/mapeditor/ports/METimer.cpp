/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "METimer.h"
#include "ports/MEParameterPort.h"

//======================================================================

METimer::METimer(MEParameterPort *p)
    : playMode(STOP)
    , active(false)
    , port(p)
{
}

METimer::~METimer()
{
}
