/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "CeilingConnector.h"

#include "PointConnector.h"
#include "../MountBehavior.h"
#include "../../Room.h"
#include "../../Events/SetTransformAxisEvent.h"

#define MIN(a, b) (a < b ? a : b)
#define MAX(a, b) (a > b ? a : b)

CeilingConnector::CeilingConnector(MountBehavior *mb)
    : Connector(mb)
{
    _constraint = CONSTRAINT_CEILING;
}

CeilingConnector::~CeilingConnector()
{
}

bool CeilingConnector::buildFromXML(QDomElement *connectorElement)
{
    if (!Connector::buildFromXML(connectorElement))
    {
        return false;
    }
    return true;
}

void CeilingConnector::applyRestriction(Connector *slaveConnector)
{
    Room *room = dynamic_cast<Room *>(getMasterObject());
    if (!room)
    {
        return;
    }

    PointConnector *slaveConn = dynamic_cast<PointConnector *>(slaveConnector); // slave must be PointConnector
    if (!slaveConn)
    {
        return;
    }

    SceneObject *slaveObject = slaveConn->getSceneObject();

    osg::Matrix m = slaveObject->getTranslate();
    osg::Vec3 t = m.getTrans();

    // Z
    t[2] = room->getCeiling()->getPosition()[2] - slaveConn->getPosition()[2];

    // X/Y
    osg::BoundingBox bbox = slaveConn->getRotatedBBox();
    t[0] = MIN(MAX(t[0], room->getPosition()[0] - room->getWidth() / 2.0f - bbox.xMin()), room->getPosition()[0] + room->getWidth() / 2.0f - bbox.xMax()); // NOTE: room rotation is ignored
    t[1] = MIN(MAX(t[1], room->getPosition()[1] - room->getLength() / 2.0f - bbox.yMin()), room->getPosition()[1] + room->getLength() / 2.0f - bbox.yMax()); // NOTE: room rotation is ignored

    m.makeTranslate(t);
    slaveObject->setTranslate(m, slaveConnector->getBehavior());
}

void CeilingConnector::prepareTransform(Connector *slaveConnector)
{
    // set transform axis
    SetTransformAxisEvent stae;
    stae.setSender(slaveConnector->getBehavior());
    stae.setTranslateAxis(osg::Vec3(0.0f, 0.0f, 1.0f));
    if (slaveConnector->getCombinedRotation() == ROTATION_AXIS)
    {
        stae.setRotateAxis(osg::Vec3(0.0f, 0.0f, 1.0f));
    }
    slaveConnector->getSceneObject()->receiveEvent(&stae);
}
