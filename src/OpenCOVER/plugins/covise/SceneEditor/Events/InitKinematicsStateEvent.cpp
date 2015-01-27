/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "InitKinematicsStateEvent.h"

InitKinematicsStateEvent::InitKinematicsStateEvent()
{
    _type = EventTypes::INIT_KINEMATICS_STATE_EVENT;
    _state = "";
}

InitKinematicsStateEvent::~InitKinematicsStateEvent()
{
}

void InitKinematicsStateEvent::setState(std::string state)
{
    _state = state;
}

std::string InitKinematicsStateEvent::getState()
{
    return _state;
}
