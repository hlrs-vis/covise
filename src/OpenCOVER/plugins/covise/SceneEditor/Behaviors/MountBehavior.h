/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef MOUNT_BEHAVIOR_H
#define MOUNT_BEHAVIOR_H

#include <vector>

#include "Behavior.h"
#include "Connectors/Connector.h"

class Connector;

class MountBehavior : public Behavior
{
public:
    MountBehavior();
    virtual ~MountBehavior();

    virtual int attach(SceneObject *);
    virtual int detach();

    virtual EventErrors::Type receiveEvent(Event *e);

    virtual bool buildFromXML(QDomElement *behaviorElement);
    Connector *buildConnectorFromXML(QDomElement *connectorElement);

    Connector *getActiveSlaveConnector();

private:
    std::vector<Connector *> _masterConnectors;
    std::vector<Connector *> _slaveConnectors;
};

#endif
