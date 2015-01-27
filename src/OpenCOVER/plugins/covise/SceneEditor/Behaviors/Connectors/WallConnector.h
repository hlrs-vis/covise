/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef WALL_CONNECTOR_H
#define WALL_CONNECTOR_H

#include "Connector.h"

class Wall;

class WallConnector : public Connector
{
public:
    WallConnector(MountBehavior *mb);
    virtual ~WallConnector();

    virtual bool buildFromXML(QDomElement *connectorElement);

    virtual void applyRestriction(Connector *slaveConnector);

    virtual void prepareTransform(Connector *slaveConnector);

private:
    Wall *_currentWall;
};

#endif
