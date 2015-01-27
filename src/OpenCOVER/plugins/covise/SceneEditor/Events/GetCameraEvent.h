/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef GET_CAMERA_EVENT_H
#define GET_CAMERA_EVENT_H

#include "Event.h"

#include <osg/Vec3>
#include <osg/Quat>

class GetCameraEvent : public Event
{
public:
    GetCameraEvent();
    virtual ~GetCameraEvent();

    void setPosition(osg::Vec3 p);
    osg::Vec3 getPosition();

    void setOrientation(osg::Quat o);
    osg::Quat getOrientation();

private:
    osg::Vec3 _position;
    osg::Quat _orientation;
};

#endif
