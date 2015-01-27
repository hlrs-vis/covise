/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "MountEvent.h"

MountEvent::MountEvent()
{
    _type = EventTypes::MOUNT_EVENT;
    _master = NULL;
    _masterConnector = NULL;
    _slaveConnector = NULL;
    _force = false;
}

MountEvent::~MountEvent()
{
}

void MountEvent::setMaster(SceneObject *so)
{
    _master = so;
}

SceneObject *MountEvent::getMaster()
{
    return _master;
}

void MountEvent::setMasterConnector(Connector *mc)
{
    _masterConnector = mc;
}

Connector *MountEvent::getMasterConnector()
{
    return _masterConnector;
}

void MountEvent::setSlaveConnector(Connector *sc)
{
    _slaveConnector = sc;
}

Connector *MountEvent::getSlaveConnector()
{
    return _slaveConnector;
}

void MountEvent::setForce(bool f)
{
    _force = f;
}

bool MountEvent::getForce()
{
    return _force;
}
