/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "UnmountEvent.h"

UnmountEvent::UnmountEvent()
{
    _type = EventTypes::UNMOUNT_EVENT;
    _master = 0;
}

UnmountEvent::~UnmountEvent()
{
}

void UnmountEvent::setMaster(SceneObject *so)
{
    _master = so;
}

SceneObject *UnmountEvent::getMaster()
{
    return _master;
}
