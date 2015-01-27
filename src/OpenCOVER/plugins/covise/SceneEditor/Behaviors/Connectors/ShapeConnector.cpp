/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "ShapeConnector.h"

#include "PointConnector.h"
#include "../MountBehavior.h"
#include "../../Shape.h"
#include "../../Events/SetTransformAxisEvent.h"

#define MIN(a, b) (a < b ? a : b)
#define MAX(a, b) (a > b ? a : b)

ShapeConnector::ShapeConnector(MountBehavior *mb)
    : Connector(mb)
{
    _constraint = CONSTRAINT_SHAPE;
    _currentSide = osg::Vec3(0.0f, 0.0f, 0.0f);
}

ShapeConnector::~ShapeConnector()
{
}

bool ShapeConnector::buildFromXML(QDomElement *connectorElement)
{
    if (!Connector::buildFromXML(connectorElement))
    {
        return false;
    }
    return true;
}

void ShapeConnector::applyRestriction(Connector *slaveConnector)
{
    Shape *shape = dynamic_cast<Shape *>(getMasterObject());
    if (!shape)
    {
        return;
    }

    PointConnector *slaveConn = dynamic_cast<PointConnector *>(slaveConnector); // slave must be PointConnector
    if (!slaveConn)
    {
        return;
    }

    SceneObject *slaveObject = slaveConn->getSceneObject();
    osg::Vec3 masterPos = shape->getTranslate().getTrans();

    if (shape->getGeometryType() == Shape::GEOMETRY_CYLINDER)
    {
        // calculate direction
        osg::Vec3 slavePos = slaveConn->getTransformedPosition();
        osg::Vec3 direction = masterPos - slavePos;
        direction[2] = 0.0f;
        direction.normalize();

        // rotation
        slaveConn->matchRotation(direction, osg::Vec3(0.0f, 0.0f, 1.0f));

        // translation
        // project along the direction
        osg::Vec3 newPos = masterPos - direction * (shape->getWidth() / 2.0f);
        newPos[2] = slavePos[2]; // keep z position
        newPos -= slaveConn->getRotatedPosition();
        // max top/bottom
        osg::BoundingBox bbox = slaveConn->getRotatedBBox();
        newPos[2] = MAX(newPos[2], masterPos[2] - shape->getHeight() / 2.0f - bbox.zMin());
        newPos[2] = MIN(newPos[2], masterPos[2] + shape->getHeight() / 2.0f - bbox.zMax());
        // set
        slaveObject->setTranslate(osg::Matrix::translate(newPos), slaveConnector->getBehavior());

        prepareTransform(slaveConnector);
    }
    else
    {
        osg::Vec3 closestSide;
        osg::Vec3 sidePosition;
        shape->getClosestSide(slaveConn->getTransformedPosition(), closestSide, sidePosition);

        // ROTATION

        slaveConn->matchRotation(-closestSide, osg::Vec3(0.0f, 0.0f, 1.0f));

        // TRANSLATION

        // move to closest wall

        // get position of mount point
        osg::Vec3 mountPos = slaveConn->getTransformedPosition();
        // transform mount point according to master position/rotation (into local coordinate system)
        mountPos = mountPos * osg::Matrix::inverse(shape->getTranslate()) * osg::Matrix::inverse(shape->getRotate());
        // get difference vector from closest side to mount point
        osg::Vec3 diff = mountPos - sidePosition;
        // project difference onto the sides normal
        diff = closestSide * (diff * closestSide);
        // subtract the projected difference (now we are on the shapes side)
        mountPos = mountPos - diff;
        // restrict mount point to extent of the side
        float w = shape->getWidth() / 2.0f;
        float l = shape->getLength() / 2.0f;
        mountPos[0] = MIN(MAX(mountPos[0], -w), w);
        mountPos[1] = MIN(MAX(mountPos[1], -l), l);
        // transform the mount point back
        mountPos = mountPos * shape->getRotate() * shape->getTranslate();
        // apply the difference between the original and new mount point to the transformation
        osg::Vec3 trans = slaveObject->getTranslate().getTrans() + (mountPos - slaveConn->getTransformedPosition());

        // max top/bottom
        osg::BoundingBox bbox = slaveConn->getRotatedBBox();
        trans[2] = MAX(trans[2], masterPos[2] - shape->getHeight() / 2.0f - bbox.zMin());
        trans[2] = MIN(trans[2], masterPos[2] + shape->getHeight() / 2.0f - bbox.zMax());

        // set transform
        slaveObject->setTranslate(osg::Matrix::translate(trans), slaveConnector->getBehavior());

        if (_currentSide != closestSide)
        {
            _currentSide = closestSide;
            prepareTransform(slaveConnector);
        }
    }
}

void ShapeConnector::prepareTransform(Connector *slaveConnector)
{
    Shape *shape = dynamic_cast<Shape *>(getMasterObject());
    if (!shape)
    {
        return;
    }

    PointConnector *slaveConn = dynamic_cast<PointConnector *>(slaveConnector); // slave must be PointConnector
    if (!slaveConn)
    {
        return;
    }

    if (shape->getGeometryType() == Shape::GEOMETRY_CYLINDER)
    {
        SetTransformAxisEvent stae;
        stae.setSender(slaveConnector->getBehavior());
        if (slaveConnector->getCombinedRotation() == ROTATION_AXIS)
        {
            osg::Vec3 direction = shape->getTranslate().getTrans() - slaveConn->getTransformedPosition();
            direction[2] = 0.0f;
            direction.normalize();
            stae.setRotateAxis(direction);
        }
        stae.resetTranslate();
        slaveConnector->getSceneObject()->receiveEvent(&stae);
    }
    else
    {
        if (_currentSide != osg::Vec3(0.0f, 0.0f, 0.0f))
        {
            // set transform axis
            osg::Vec3 axis = _currentSide * shape->getRotate();
            SetTransformAxisEvent stae;
            stae.setSender(slaveConnector->getBehavior());
            stae.setTranslateAxis(axis);
            if (slaveConnector->getCombinedRotation() == ROTATION_AXIS)
            {
                stae.setRotateAxis(axis);
            }
            slaveConnector->getSceneObject()->receiveEvent(&stae);
        }
    }
}
