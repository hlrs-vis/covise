/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef SINUS_SCALING_BEHAVIOR_H
#define SINUS_SCALING_BEHAVIOR_H

#include "Behavior.h"
#include "../Events/PreFrameEvent.h"

#include <osg/MatrixTransform>

class SinusScalingBehavior : public Behavior
{
public:
    SinusScalingBehavior();
    virtual ~SinusScalingBehavior();

    virtual int attach(SceneObject *);
    virtual int detach();

    virtual EventErrors::Type receiveEvent(Event *e);

    virtual bool buildFromXML(QDomElement *behaviorElement);

private:
    float _x;
    float _speed;
    float _amplitude;
    osg::ref_ptr<osg::MatrixTransform> _scaleNode;
};

#endif
