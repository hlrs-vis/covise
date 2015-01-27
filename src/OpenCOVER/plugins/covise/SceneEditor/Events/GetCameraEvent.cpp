/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "GetCameraEvent.h"

GetCameraEvent::GetCameraEvent()
{
    _type = EventTypes::GET_CAMERA_EVENT;
}

GetCameraEvent::~GetCameraEvent()
{
}

void GetCameraEvent::setPosition(osg::Vec3 p)
{
    _position = p;
}

osg::Vec3 GetCameraEvent::getPosition()
{
    return _position;
}

void GetCameraEvent::setOrientation(osg::Quat o)
{
    _orientation = o;
}

osg::Quat GetCameraEvent::getOrientation()
{
    return _orientation;
}
