/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CONNECTOR_H
#define CONNECTOR_H

#include <iostream>
#include <set>

#include <QDomElement>

#include <osg/BoundingBox>

#include "../../SceneObject.h"

class MountBehavior;

//        MasterConnector
//        |             |
//      Slave 1       Slave 2

class Connector
{
public:
    Connector(MountBehavior *mb);
    virtual ~Connector();

    enum Role
    {
        ROLE_NONE,
        ROLE_MASTER,
        ROLE_SLAVE
    };

    enum Constraint
    {
        CONSTRAINT_NONE,
        CONSTRAINT_POINT,
        CONSTRAINT_WALL,
        CONSTRAINT_FLOOR,
        CONSTRAINT_CEILING,
        CONSTRAINT_SHAPE
    };

    enum Rotation
    {
        ROTATION_FIXED,
        ROTATION_AXIS,
        ROTATION_FREE
    };

    enum WallAlignment
    {
        ALIGNMENT_NONE,
        ALIGNMENT_BOTTOM,
        ALIGNMENT_TOP
    };

    virtual bool buildFromXML(QDomElement *connectorElement);

    Role getRole();
    Constraint getConstraint();
    std::string getType();
    std::string getName();
    Rotation getRotation();
    MountBehavior *getBehavior();

    SceneObject *getMasterObject();
    SceneObject *getSceneObject();

    ///////////////
    // slave only

    WallAlignment getWallAlignment();
    Rotation getCombinedRotation();

    void setMasterConnector(Connector *c);
    Connector *getMasterConnector();

    osg::BoundingBox getRotatedBBox();

    // slave only
    ///////////////

    ///////////////
    // master only

    void addSlaveConnector(Connector *c);
    void removeSlaveConnector(Connector *c);
    bool allowsAnotherSlave();
    std::set<Connector *> getSlaveConnectors();

    void applyRestriction();
    virtual void applyRestriction(Connector *slaveConnector);

    virtual void prepareTransform(Connector *slaveConnector);

    // master only
    ///////////////

protected:
    Role _role;
    Constraint _constraint;
    std::string _type;
    std::string _name;
    Rotation _rotation; // {fixed, axis, free} (the more restrictive value of master and slave will be used)

    MountBehavior *_mountBehavior;

    ///////////////
    // slave only
    WallAlignment _wallAlignment;
    bool _ignoreBBox;
    Connector *_masterConnector; // NULL if not connected
    // slave only
    ///////////////

    ///////////////
    // master only
    int _maxSlaves;
    std::set<Connector *> _slaveConnectors; // not present if not connected
    // master only
    ///////////////
};

#endif
