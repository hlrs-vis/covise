/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef MOUNT_EVENT_H
#define MOUNT_EVENT_H

#include "Event.h"

#include <iostream>

class SceneObject;
class Connector;

class MountEvent : public Event
{
public:
    MountEvent();
    virtual ~MountEvent();

    void setMaster(SceneObject *so);
    SceneObject *getMaster();

    void setMasterConnector(Connector *mc);
    Connector *getMasterConnector();

    void setSlaveConnector(Connector *sc);
    Connector *getSlaveConnector();

    void setForce(bool f);
    bool getForce();

private:
    SceneObject *_master;
    Connector *_masterConnector;
    Connector *_slaveConnector;
    bool _force;
};

#endif
