/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef POINT_CONNECTOR_H
#define POINT_CONNECTOR_H

#include "Connector.h"

#include <osg/Vec3>

class PointConnector : public Connector
{
public:
    PointConnector(MountBehavior *mb);
    virtual ~PointConnector();

    virtual bool buildFromXML(QDomElement *connectorElement);

    virtual void applyRestriction(Connector *slaveConnector);

    virtual void prepareTransform(Connector *slaveConnector);

    osg::Vec3 getPosition();
    osg::Vec3 getRotatedPosition();
    osg::Vec3 getTransformedPosition();

    osg::Vec3 getOrientation();
    osg::Vec3 getRotatedOrientation();

    osg::Vec3 getUp();
    osg::Vec3 getRotatedUp();

    ///////////////
    // slave only
    void matchRotation(osg::Vec3 masterOrientation, osg::Vec3 masterUp);
    ///////////////

private:
    enum ZAlignment
    {
        Z_NONE,
        Z_BOTTOM,
        Z_TOP
    };

    osg::Vec3 _position;
    osg::Vec3 _orientation; // orientation points from the slave towards the master object
    osg::Vec3 _up;

    ZAlignment _zAlignment;
};

#endif
