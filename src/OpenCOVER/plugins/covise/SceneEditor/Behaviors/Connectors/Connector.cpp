/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Connector.h"

#include "../MountBehavior.h"

#define MIN(a, b) (a < b ? a : b)

Connector::Connector(MountBehavior *mb)
{
    _mountBehavior = mb;
    _constraint = CONSTRAINT_NONE;
    _role = ROLE_NONE;
    _name = "";
    _wallAlignment = ALIGNMENT_NONE;
    _maxSlaves = 99999;
    _masterConnector = NULL;
    _rotation = ROTATION_AXIS;
    _ignoreBBox = false;
}

Connector::~Connector()
{
}

bool Connector::buildFromXML(QDomElement *connectorElement)
{
    QDomElement elem;

    elem = connectorElement->firstChildElement("role");
    if (elem.isNull())
    {
        std::cerr << "Error: role missing" << std::endl;
        return false;
    }
    std::string roleString = elem.attribute("value", "").toStdString();
    if (roleString == "master")
    {
        _role = ROLE_MASTER;
    }
    else if (roleString == "slave")
    {
        _role = ROLE_SLAVE;
    }
    else
    {
        std::cerr << "Error: Unknown role: " << roleString << std::endl;
        return false;
    }

    elem = connectorElement->firstChildElement("name");
    if (!elem.isNull())
    {
        _name = elem.attribute("value", "").toStdString();
    }

    elem = connectorElement->firstChildElement("type");
    if (!elem.isNull())
    {
        _type = elem.attribute("value", "").toStdString();
    }

    elem = connectorElement->firstChildElement("maximum_slaves");
    if (!elem.isNull())
    {
        _maxSlaves = elem.attribute("value", "").toInt();
    }

    elem = connectorElement->firstChildElement("ignoreBBox");
    _ignoreBBox = !elem.isNull();

    elem = connectorElement->firstChildElement("rotation");
    if (!elem.isNull())
    {
        std::string rotString = elem.attribute("value", "").toStdString();
        if (rotString == "free")
        {
            _rotation = ROTATION_FREE;
        }
        else if (rotString == "axis")
        {
            _rotation = ROTATION_AXIS;
        }
        else if (rotString == "fixed")
        {
            _rotation = ROTATION_FIXED;
        }
        else
        {
            std::cerr << "Error: Unknown rotation: " << rotString << std::endl;
            return false;
        }
    }

    elem = connectorElement->firstChildElement("wall_alignment");
    if (!elem.isNull())
    {
        std::string alignString = elem.attribute("value", "").toStdString();
        if (alignString == "bottom")
        {
            _wallAlignment = ALIGNMENT_BOTTOM;
        }
        else if (alignString == "top")
        {
            _wallAlignment = ALIGNMENT_TOP;
        }
        else
        {
            std::cerr << "Error: Unknown wall_alignment: " << alignString << std::endl;
            return false;
        }
    }

    return true;
}

Connector::Role Connector::getRole()
{
    return _role;
}

Connector::Constraint Connector::getConstraint()
{
    return _constraint;
}

std::string Connector::getType()
{
    return _type;
}

std::string Connector::getName()
{
    return _name;
}

Connector::Rotation Connector::getRotation()
{
    return _rotation;
}

Connector::Rotation Connector::getCombinedRotation()
{
    if (_masterConnector == NULL)
    {
        return _rotation;
    }
    return MIN(_rotation, _masterConnector->_rotation);
}

Connector::WallAlignment Connector::getWallAlignment()
{
    return _wallAlignment;
}

MountBehavior *Connector::getBehavior()
{
    return _mountBehavior;
}

void Connector::setMasterConnector(Connector *c)
{
    _masterConnector = c;
}

Connector *Connector::getMasterConnector()
{
    return _masterConnector;
}

void Connector::addSlaveConnector(Connector *c)
{
    _slaveConnectors.insert(c);
}

void Connector::removeSlaveConnector(Connector *c)
{
    _slaveConnectors.erase(c);
}

bool Connector::allowsAnotherSlave()
{
    return (_slaveConnectors.size() < _maxSlaves);
}

std::set<Connector *> Connector::getSlaveConnectors()
{
    return _slaveConnectors;
}

SceneObject *Connector::getMasterObject()
{
    if (_role == ROLE_MASTER)
    {
        return _mountBehavior->getSceneObject();
    }
    else
    {
        if (_masterConnector == NULL)
        {
            return NULL;
        }
        else
        {
            return _masterConnector->_mountBehavior->getSceneObject();
        }
    }
}

SceneObject *Connector::getSceneObject()
{
    return _mountBehavior->getSceneObject();
}

osg::BoundingBox Connector::getRotatedBBox()
{
    if (_ignoreBBox)
    {
        osg::BoundingBox bbox = osg::BoundingBox();
        bbox.set(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
        return bbox;
    }
    else
    {
        return _mountBehavior->getSceneObject()->getRotatedBBox();
    }
}

void Connector::applyRestriction()
{
    for (std::set<Connector *>::iterator it = _slaveConnectors.begin(); it != _slaveConnectors.end(); ++it)
    {
        applyRestriction((*it));
    }
}

void Connector::applyRestriction(Connector *slaveConnector)
{
    // no restriction
    (void)slaveConnector;
}

void Connector::prepareTransform(Connector *slaveConnector)
{
    // nothing to do
    (void)slaveConnector;
}
