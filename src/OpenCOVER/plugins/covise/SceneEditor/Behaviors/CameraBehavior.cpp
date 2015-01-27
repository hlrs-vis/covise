/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "CameraBehavior.h"

#include "../ErrorCodes.h"
#include "../Events/GetCameraEvent.h"

CameraBehavior::CameraBehavior()
{
    _type = BehaviorTypes::CAMERA_BEHAVIOR;
}

CameraBehavior::~CameraBehavior()
{
}

int CameraBehavior::attach(SceneObject *so)
{
    // connects this behavior to its scene object
    Behavior::attach(so);

    return 1;
}

int CameraBehavior::detach()
{
    Behavior::detach();

    return 1;
}

EventErrors::Type CameraBehavior::receiveEvent(Event *e)
{
    if (e->getType() == EventTypes::GET_CAMERA_EVENT)
    {
        GetCameraEvent *gce = dynamic_cast<GetCameraEvent *>(e);
        gce->setPosition(_position);
        gce->setOrientation(_orientation);

        return EventErrors::SUCCESS;
    }

    return EventErrors::UNHANDLED;
}

bool CameraBehavior::buildFromXML(QDomElement *behaviorElement)
{
    QDomElement elem;

    elem = behaviorElement->firstChildElement("position");
    if (!elem.isNull())
    {
        _position[0] = elem.attribute("x").toFloat();
        _position[1] = elem.attribute("y").toFloat();
        _position[2] = elem.attribute("z").toFloat();
    }

    elem = behaviorElement->firstChildElement("orientation");
    if (!elem.isNull())
    {
        float x, y, z = 0.0f;

        x = osg::DegreesToRadians(elem.attribute("x", "0.0").toFloat());
        y = osg::DegreesToRadians(elem.attribute("y", "0.0").toFloat());
        z = osg::DegreesToRadians(elem.attribute("z", "0.0").toFloat());

        _orientation.makeRotate(x, osg::Vec3(1.0f, 0.0f, 0.0f), y, osg::Vec3(0.0f, 1.0f, 0.0f), z, osg::Vec3(0.0f, 0.0f, 1.0f));
    }

    return true;
}
