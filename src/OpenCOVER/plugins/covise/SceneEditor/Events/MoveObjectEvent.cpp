/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "MoveObjectEvent.h"

MoveObjectEvent::MoveObjectEvent()
{
    _type = EventTypes::MOVE_OBJECT_EVENT;
}

MoveObjectEvent::~MoveObjectEvent()
{
}

void MoveObjectEvent::setDirection(osg::Vec3 dir)
{
    _direction = dir;
}

osg::Vec3 MoveObjectEvent::getDirection()
{
    return _direction;
}
