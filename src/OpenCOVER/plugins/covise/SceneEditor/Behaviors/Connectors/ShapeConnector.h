/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef SHAPE_CONNECTOR_H
#define SHAPE_CONNECTOR_H

#include "Connector.h"

#include <osg/Vec3>

class ShapeConnector : public Connector
{
public:
    ShapeConnector(MountBehavior *mb);
    virtual ~ShapeConnector();

    virtual bool buildFromXML(QDomElement *connectorElement);

    virtual void applyRestriction(Connector *slaveConnector);

    virtual void prepareTransform(Connector *slaveConnector);

private:
    osg::Vec3 _currentSide; // (1,0,0)/(-1,0,0)/(0,1,0)/(0,-1,0)
};

#endif
