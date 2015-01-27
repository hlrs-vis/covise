/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CAMERA_BEHAVIOR_H
#define CAMERA_BEHAVIOR_H

#include "Behavior.h"
#include "../Events/PreFrameEvent.h"

#include <osg/Vec3>
#include <osg/Quat>

class CameraBehavior : public Behavior
{
public:
    CameraBehavior();
    virtual ~CameraBehavior();

    virtual int attach(SceneObject *);
    virtual int detach();

    virtual EventErrors::Type receiveEvent(Event *e);

    virtual bool buildFromXML(QDomElement *behaviorElement);

private:
    osg::Vec3 _position;
    osg::Quat _orientation;
};

#endif
