/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "PointConnector.h"

#include "../MountBehavior.h"
#include "../../Events/SetTransformAxisEvent.h"
#include "../../SceneObject.h"

PointConnector::PointConnector(MountBehavior *mb)
    : Connector(mb)
{
    _constraint = CONSTRAINT_POINT;
    _position = osg::Vec3(0.0f, 0.0f, 0.0f);
    _orientation = osg::Vec3(1.0f, 0.0f, 0.0f);
    _up = osg::Vec3(0.0f, 0.0f, 1.0f);
    _zAlignment = Z_NONE;
}

PointConnector::~PointConnector()
{
}

bool PointConnector::buildFromXML(QDomElement *connectorElement)
{
    if (!Connector::buildFromXML(connectorElement))
    {
        return false;
    }

    QDomElement constraintElem = connectorElement->firstChildElement("constraint");
    if (!constraintElem.isNull())
    {
        QDomElement subElem;
        subElem = constraintElem.firstChildElement("position");
        if (!subElem.isNull())
        {
            _position = osg::Vec3(subElem.attribute("x", "0.0").toFloat(),
                                  subElem.attribute("y", "0.0").toFloat(),
                                  subElem.attribute("z", "0.0").toFloat());
            if (subElem.attribute("z", "").toStdString() == "bottom")
            {
                _zAlignment = Z_BOTTOM;
            }
            else if (subElem.attribute("z", "").toStdString() == "top")
            {
                _zAlignment = Z_TOP;
            }
        }
        subElem = constraintElem.firstChildElement("orientation");
        if (!subElem.isNull())
        {
            _orientation = osg::Vec3(subElem.attribute("x", "1.0").toFloat(),
                                     subElem.attribute("y", "0.0").toFloat(),
                                     subElem.attribute("z", "0.0").toFloat());
        }
        subElem = constraintElem.firstChildElement("up");
        if (!subElem.isNull())
        {
            _up = osg::Vec3(subElem.attribute("x", "0.0").toFloat(),
                            subElem.attribute("y", "0.0").toFloat(),
                            subElem.attribute("z", "1.0").toFloat());
        }
    }

    return true;
}

osg::Vec3 PointConnector::getPosition()
{
    if (_zAlignment != Z_NONE)
    {
        osg::BoundingBox bbox = getSceneObject()->getRotatedBBox();
        if (_zAlignment == Z_BOTTOM)
        {
            _position[2] = bbox.zMin();
        }
        else if (_zAlignment == Z_TOP)
        {
            _position[2] = bbox.zMax();
        }
    }
    return _position;
}

osg::Vec3 PointConnector::getRotatedPosition()
{
    return getPosition() * getSceneObject()->getRotate();
}

osg::Vec3 PointConnector::getTransformedPosition()
{
    return getRotatedPosition() * getSceneObject()->getTranslate();
}

osg::Vec3 PointConnector::getOrientation()
{
    return _orientation;
}

osg::Vec3 PointConnector::getRotatedOrientation()
{
    return _orientation * getSceneObject()->getRotate();
}

osg::Vec3 PointConnector::getUp()
{
    return _up;
}

osg::Vec3 PointConnector::getRotatedUp()
{
    return _up * getSceneObject()->getRotate();
}

void PointConnector::matchRotation(osg::Vec3 masterOrientation, osg::Vec3 masterUp)
{
    if (getCombinedRotation() == ROTATION_FIXED)
    {
        osg::Matrix rotMat;
        // match orientation
        rotMat.makeRotate(getOrientation(), masterOrientation);
        // match up-vectors and add rotation
        osg::Matrix rotMat2;
        rotMat2.makeRotate(getUp() * rotMat, masterUp); // apply rotation calculated so far to the slaves up-vector
        rotMat.postMult(rotMat2);
        // add rotation of master object
        rotMat.postMult(getMasterObject()->getRotate());
        // set rotation
        getSceneObject()->setRotate(rotMat, getBehavior());
    }
    else if (getCombinedRotation() == ROTATION_AXIS)
    {
        osg::Matrix rotMat;
        rotMat.makeRotate(getRotatedOrientation(), masterOrientation * getMasterObject()->getRotate());
        rotMat.preMult(getSceneObject()->getRotate());
        getSceneObject()->setRotate(rotMat, getBehavior());
    }
}

void PointConnector::applyRestriction(Connector *slaveConnector)
{
    PointConnector *slaveConn = dynamic_cast<PointConnector *>(slaveConnector); // slave must be PointConnector
    if (!slaveConn)
    {
        return;
    }

    // rotation
    slaveConn->matchRotation(getOrientation(), getUp());

    // translation
    osg::Matrix transMat;
    transMat.makeTranslate(getTransformedPosition() - slaveConn->getRotatedPosition());
    slaveConn->getSceneObject()->setTranslate(transMat, slaveConnector->getBehavior());
}

void PointConnector::prepareTransform(Connector *slaveConnector)
{
    // set transform axis
    if (slaveConnector->getCombinedRotation() == ROTATION_AXIS)
    {
        SetTransformAxisEvent stae;
        stae.setSender(slaveConnector->getBehavior());
        stae.setRotateAxis(getRotatedOrientation());
        slaveConnector->getSceneObject()->receiveEvent(&stae);
    }
}
