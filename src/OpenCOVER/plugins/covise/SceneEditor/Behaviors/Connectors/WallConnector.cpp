/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "WallConnector.h"

#include "PointConnector.h"
#include "../MountBehavior.h"
#include "../../Room.h"
#include "../../Events/SetTransformAxisEvent.h"

#define MIN(a, b) (a < b ? a : b)
#define MAX(a, b) (a > b ? a : b)

WallConnector::WallConnector(MountBehavior *mb)
    : Connector(mb)
{
    _constraint = CONSTRAINT_WALL;
    _currentWall = NULL;
}

WallConnector::~WallConnector()
{
}

bool WallConnector::buildFromXML(QDomElement *connectorElement)
{
    if (!Connector::buildFromXML(connectorElement))
    {
        return false;
    }
    return true;
}

// NOTE: The rotation of the room is ignored for now.
//       When implementing this feature, look at ShapeConnector (the same problem is already solved there).
void WallConnector::applyRestriction(Connector *slaveConnector)
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
    Wall *closestWall = room->getClosestWall(slaveConn->getTransformedPosition());

    // ROTATION
    slaveConn->matchRotation(-closestWall->getNormal(), osg::Vec3(0.0f, 0.0f, 1.0f));

    // TRANSLATION

    // move to closest wall
    // get current translation
    osg::Vec3 trans = slaveObject->getTranslate().getTrans();
    // calculate the vector from the center of the wall to the mount point of the slave object
    osg::Vec3 diff = trans + slaveConn->getRotatedPosition() - closestWall->getPosition() - room->getPosition();
    // project the difference vector onto the walls normal and subtract from the translation
    trans = trans - closestWall->getNormal() * (diff * closestWall->getNormal());

    osg::BoundingBox bbox = slaveConn->getRotatedBBox();

    // restrict (left/right) on wall according to BBox
    if (fabs(closestWall->getNormal()[1]) > 0.99f)
    {
        trans[0] = MIN(MAX(trans[0], room->getPosition()[0] - room->getWidth() / 2.0f - bbox.xMin()), room->getPosition()[0] + room->getWidth() / 2.0f - bbox.xMax()); // NOTE: room rotation is ignored
    }
    else if (fabs(closestWall->getNormal()[0]) > 0.99f)
    {
        trans[1] = MIN(MAX(trans[1], room->getPosition()[1] - room->getLength() / 2.0f - bbox.yMin()), room->getPosition()[1] + room->getLength() / 2.0f - bbox.yMax()); // NOTE: room rotation is ignored
    }

    // restrict (top/bottom) on wall according to BBox and align on floor/ceiling
    if ((slaveConn->getWallAlignment() == ALIGNMENT_BOTTOM) || (trans[2] < float(room->getFloor()->getPosition()[2]) - bbox.zMin()))
    {
        trans[2] = float(room->getFloor()->getPosition()[2] + room->getPosition()[2]) - bbox.zMin();
    }
    else if ((slaveConn->getWallAlignment() == ALIGNMENT_TOP) || (trans[2] > float(room->getCeiling()->getPosition()[2]) - bbox.zMax()))
    {
        trans[2] = float(room->getCeiling()->getPosition()[2] + room->getPosition()[2]) - bbox.zMax();
    }

    // set transform
    slaveObject->setTranslate(osg::Matrix::translate(trans), slaveConnector->getBehavior());

    if (_currentWall != closestWall)
    {
        _currentWall = closestWall;
        prepareTransform(slaveConnector);
    }
}

void WallConnector::prepareTransform(Connector *slaveConnector)
{
    if (_currentWall != NULL)
    {
        // set transform axis
        SetTransformAxisEvent stae;
        stae.setSender(slaveConnector->getBehavior());
        stae.setTranslateAxis(_currentWall->getNormal());
        if (slaveConnector->getCombinedRotation() == ROTATION_AXIS)
        {
            stae.setRotateAxis(_currentWall->getNormal());
        }
        slaveConnector->getSceneObject()->receiveEvent(&stae);
    }
}
