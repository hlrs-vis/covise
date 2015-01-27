/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef MOVE_OBJECT_EVENT_H
#define MOVE_OBJECT_EVENT_H

#include "Event.h"

#include <osg/Vec3>

class MoveObjectEvent : public Event
{
public:
    MoveObjectEvent();
    virtual ~MoveObjectEvent();

    void setDirection(osg::Vec3 dir);
    osg::Vec3 getDirection();

private:
    osg::Vec3 _direction;
};

#endif
