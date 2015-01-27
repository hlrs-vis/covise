/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "SinusScalingBehavior.h"
#include "../Asset.h"
#include "../SceneUtils.h"

#include <iostream>
#include <math.h>

SinusScalingBehavior::SinusScalingBehavior()
{
    _type = BehaviorTypes::SINUS_SCALING_BEHAVIOR;
    _x = 1.0f;
    _speed = 0.1f;
    _amplitude = 0.1f;
    _scaleNode = NULL;
}

SinusScalingBehavior::~SinusScalingBehavior()
{
}

int SinusScalingBehavior::attach(SceneObject *so)
{
    // connects this behavior to its scene object
    Behavior::attach(so);

    // add a transform node in front of osg::node of asset
    _scaleNode = new osg::MatrixTransform();
    SceneUtils::insertNode(_scaleNode.get(), so);

    return 1;
}

int SinusScalingBehavior::detach()
{
    SceneUtils::removeNode(_scaleNode.get());
    _scaleNode = NULL;

    Behavior::detach();

    return 1;
}

EventErrors::Type SinusScalingBehavior::receiveEvent(Event *e)
//int SinusScalingBehavior::receiveEvent(const PreFrameEvent * e)
{
    if (e->getType() == EventTypes::PRE_FRAME_EVENT)
    {
        // do a sinusoid scale of parental asset (scene object)
        _x += _speed;
        float scale = 1.0 + _amplitude * sin(_x);
        osg::Matrix m = osg::Matrix::scale(scale, scale, scale);
        _scaleNode->setMatrix(m);
        //std::cout << "SinusScalingBehavior received PRE_FRAME_EVENT" << std::endl;
    }

    return EventErrors::UNHANDLED;
}

bool SinusScalingBehavior::buildFromXML(QDomElement *behaviorElement)
{
    QDomElement speedElem = behaviorElement->firstChildElement("speed");
    if (!speedElem.isNull())
    {
        bool ok;
        float tmpSpeed = speedElem.attribute("value").toFloat(&ok);
        if (ok)
        {
            _speed = tmpSpeed;
        }
    }
    QDomElement amplitudeElem = behaviorElement->firstChildElement("amplitude");
    if (!amplitudeElem.isNull())
    {
        bool ok;
        float tmpAmplitude = amplitudeElem.attribute("value").toFloat(&ok);
        if (ok)
        {
            _amplitude = tmpAmplitude;
        }
    }
    return true;
}
