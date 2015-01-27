/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef FLOOR_CONNECTOR_H
#define FLOOR_CONNECTOR_H

#include "Connector.h"

class FloorConnector : public Connector
{
public:
    FloorConnector(MountBehavior *mb);
    virtual ~FloorConnector();

    virtual bool buildFromXML(QDomElement *connectorElement);

    virtual void applyRestriction(Connector *slaveConnector);

    virtual void prepareTransform(Connector *slaveConnector);

private:
};

#endif
