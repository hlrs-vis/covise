/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "MountBehavior.h"
#include "../SceneObject.h"
#include "../SceneObjectManager.h"

#include <iostream>

#include "Connectors/PointConnector.h"
#include "Connectors/CeilingConnector.h"
#include "Connectors/FloorConnector.h"
#include "Connectors/WallConnector.h"
#include "Connectors/ShapeConnector.h"

#include "../Events/MountEvent.h"
#include "../Events/UnmountEvent.h"
#include "../Events/TransformChangedEvent.h"
#include "../Events/ApplyMountRestrictionsEvent.h"

MountBehavior::MountBehavior()
{
    _type = BehaviorTypes::MOUNT_BEHAVIOR;
}

MountBehavior::~MountBehavior()
{
}

int MountBehavior::attach(SceneObject *so)
{
    Behavior::attach(so);
    return 1;
}

int MountBehavior::detach()
{
    // from all masterConnectors (loop 1) unmount all "children" (loop 2)
    for (std::vector<Connector *>::iterator it_m = _masterConnectors.begin(); it_m != _masterConnectors.end(); ++it_m)
    {
        std::set<Connector *> slaves = (*it_m)->getSlaveConnectors();
        for (std::set<Connector *>::iterator it_s = slaves.begin(); it_s != slaves.end(); ++it_s)
        {
            UnmountEvent ue;
            ue.setSender(this);
            ue.setMaster((*it_m)->getMasterObject());
            (*it_s)->getSceneObject()->receiveEvent(&ue);
        }
    }

    // unmount from "parent"
    Connector *slaveConn = getActiveSlaveConnector();
    if (slaveConn != NULL)
    {
        UnmountEvent ue;
        ue.setSender(this);
        ue.setMaster(slaveConn->getMasterObject());
        getSceneObject()->receiveEvent(&ue);
    }

    Behavior::detach();
    return -1;
}

EventErrors::Type MountBehavior::receiveEvent(Event *e)
{
    if (e->getType() == EventTypes::MOUNT_EVENT)
    {
        MountEvent *mountEvent = dynamic_cast<MountEvent *>(e);
        if (_slaveConnectors.size() == 0)
        {
            return EventErrors::FAIL; // no slave connectors available
        }

        if (getActiveSlaveConnector())
        {
            if (mountEvent->getForce())
            {
                UnmountEvent ue;
                ue.setSender(this);
                ue.setMaster(getActiveSlaveConnector()->getMasterObject());
                _sceneObject->receiveEvent(&ue);
            }
            else
            {
                return EventErrors::FAIL; // can only mount once
            }
        }

        SceneObject *master = mountEvent->getMaster();
        if (master)
        {

            MountBehavior *masterBehavior = dynamic_cast<MountBehavior *>(master->findBehavior(BehaviorTypes::MOUNT_BEHAVIOR));
            if (!masterBehavior)
            {
                return EventErrors::FAIL; // master object has no MountBehavior
            }
            // search a matching connector pair slave/master
            Connector *masterConnector = NULL;
            Connector *slaveConnector = NULL;
            for (std::vector<Connector *>::iterator it = masterBehavior->_masterConnectors.begin(); it < masterBehavior->_masterConnectors.end(); ++it)
            {
                Connector *mc;
                if (mountEvent->getMasterConnector())
                {
                    mc = mountEvent->getMasterConnector();
                    it = masterBehavior->_masterConnectors.end();
                }
                else
                {
                    mc = (*it);
                }
                for (std::vector<Connector *>::iterator it2 = _slaveConnectors.begin(); it2 < _slaveConnectors.end(); ++it2)
                {
                    Connector *sc;
                    if (mountEvent->getSlaveConnector())
                    {
                        sc = mountEvent->getSlaveConnector();
                        it2 = _slaveConnectors.end();
                    }
                    else
                    {
                        sc = (*it2);
                    }
                    if (mc->allowsAnotherSlave() && (mc->getType().find(sc->getType()) != std::string::npos))
                    {
                        masterConnector = mc;
                        slaveConnector = sc;
                        break;
                    }
                }
            }
            if (!masterConnector || !slaveConnector)
            {
                return EventErrors::FAIL;
            }
            // mount
            master->addChild(_sceneObject);
            masterConnector->addSlaveConnector(slaveConnector);
            slaveConnector->setMasterConnector(masterConnector);
            // apply restrictions
            ApplyMountRestrictionsEvent amre;
            amre.setSender(this);
            _sceneObject->receiveEvent(&amre);
            return EventErrors::SUCCESS;
        }
        else
        {

            // remember original transformation
            osg::Matrix origTrans = _sceneObject->getTranslate();
            osg::Matrix origRot = _sceneObject->getRotate();
            // search MountEvent that requires the smallest change in position (TODO: it is not nescessary to send any position information to the GUI)
            float bestDiff = 1.0e30;
            MountEvent bestMountEvent;
            std::vector<SceneObject *> list = SceneObjectManager::instance()->getSceneObjects();
            for (std::vector<SceneObject *>::iterator it = list.begin(); it != list.end(); it++)
            {
                if ((*it) != _sceneObject)
                {
                    MountEvent me;
                    me.setSender(this);
                    me.setMaster(*it);
                    if (_sceneObject->receiveEvent(&me) == EventErrors::SUCCESS)
                    {
                        float diff = (_sceneObject->getTranslate().getTrans() - origTrans.getTrans()).length2();
                        if (diff < bestDiff)
                        {
                            bestDiff = diff;
                            bestMountEvent = me;
                        }
                        // unmount
                        UnmountEvent ue;
                        me.setSender(this);
                        me.setMaster(*it);
                        _sceneObject->receiveEvent(&ue);
                        // revert position
                        _sceneObject->setTranslate(origTrans);
                        _sceneObject->setRotate(origRot);
                    }
                }
            }
            // final mount
            if (bestMountEvent.getMaster())
            {
                return _sceneObject->receiveEvent(&bestMountEvent);
            }
        }
    }
    else if (e->getType() == EventTypes::UNMOUNT_EVENT)
    {
        Connector *conn = getActiveSlaveConnector();
        if (!conn)
        {
            return EventErrors::UNHANDLED;
        }
        conn->getMasterConnector()->removeSlaveConnector(conn);
        conn->setMasterConnector(NULL);
        if (_sceneObject->getParent())
            _sceneObject->getParent()->removeChild(_sceneObject);
        return EventErrors::SUCCESS;
    }
    else if (e->getType() == EventTypes::DOUBLE_CLICK_EVENT)
    {
        if (_sceneObject->getType() == SceneObjectTypes::WINDOW)
        {
            return EventErrors::UNHANDLED; // do not mount/unmount windows via double click
        }
        if (getActiveSlaveConnector())
        {
            // unmount
            UnmountEvent ue;
            ue.setSender(this);
            _sceneObject->receiveEvent(&ue);
        }
        else
        {
            MountEvent me;
            me.setSender(this);
            _sceneObject->receiveEvent(&me);
        }
    }
    else if (e->getType() == EventTypes::START_MOUSE_EVENT)
    {
        Connector *conn = getActiveSlaveConnector();
        if (conn)
        {
            conn->getMasterConnector()->prepareTransform(conn);
        }
    }
    else if (e->getType() == EventTypes::TRANSFORM_CHANGED_EVENT)
    {
        TransformChangedEvent *tce = dynamic_cast<TransformChangedEvent *>(e);
        if (tce->getSender().isBehavior() && (tce->getSender().getBehavior()->getType() == BehaviorTypes::MOUNT_BEHAVIOR))
        {
            // do nothing if the transform change originated from a MountBehavior to avoid loops
            return EventErrors::UNHANDLED;
        }
        ApplyMountRestrictionsEvent amre;
        amre.setSender(this);
        _sceneObject->receiveEvent(&amre);
    }
    else if (e->getType() == EventTypes::APPLY_MOUNT_RESTRICTIONS_EVENT)
    {
        // process this connection
        Connector *conn = getActiveSlaveConnector();
        if (conn)
        {
            conn->getMasterConnector()->applyRestriction(conn);
        }
        // propagate to children
        for (std::vector<Connector *>::iterator it_m = _masterConnectors.begin(); it_m != _masterConnectors.end(); ++it_m)
        {
            std::set<Connector *> slaves = (*it_m)->getSlaveConnectors();
            for (std::set<Connector *>::iterator it_s = slaves.begin(); it_s != slaves.end(); ++it_s)
            {
                (*it_s)->getSceneObject()->receiveEvent(e);
            }
        }
        return EventErrors::SUCCESS;
    }

    // the following events are just propagated to the children
    if (e->getType() == EventTypes::POST_INTERACTION_EVENT)
    {
        for (std::vector<Connector *>::iterator it_m = _masterConnectors.begin(); it_m != _masterConnectors.end(); ++it_m)
        {
            std::set<Connector *> slaves = (*it_m)->getSlaveConnectors();
            for (std::set<Connector *>::iterator it_s = slaves.begin(); it_s != slaves.end(); ++it_s)
            {
                (*it_s)->getSceneObject()->receiveEvent(e);
            }
        }
    }

    return EventErrors::UNHANDLED;
}

bool MountBehavior::buildFromXML(QDomElement *behaviorElement)
{
    QDomElement connectorElem = behaviorElement->firstChildElement("connector");
    while (!connectorElem.isNull())
    {
        Connector *conn = buildConnectorFromXML(&connectorElem);
        if (conn)
        {
            if (conn->getRole() == Connector::ROLE_MASTER)
            {
                _masterConnectors.push_back(conn);
            }
            else if (conn->getRole() == Connector::ROLE_SLAVE)
            {
                _slaveConnectors.push_back(conn);
            }
            else
            {
                delete conn;
            }
        }
        connectorElem = connectorElem.nextSiblingElement("connector");
    }

    return true;
}

Connector *MountBehavior::buildConnectorFromXML(QDomElement *connectorElement)
{
    QDomElement constraintElem = connectorElement->firstChildElement("constraint");
    if (constraintElem.isNull())
    {
        std::cerr << "Error: constraint missing" << std::endl;
        return NULL;
    }
    std::string cString = constraintElem.attribute("value", "").toStdString();

    Connector *conn = NULL;

    if (cString == "point") // slave or master
    {
        conn = new PointConnector(this);
    }
    else if (cString == "ceiling") // master
    {
        conn = new CeilingConnector(this);
    }
    else if (cString == "floor") // master
    {
        conn = new FloorConnector(this);
    }
    else if (cString == "wall") // master
    {
        conn = new WallConnector(this);
    }
    else if (cString == "shape") // master
    {
        conn = new ShapeConnector(this);
    }

    if (!conn)
    {
        std::cerr << "Error: Unknown constraint: " << cString << std::endl;
        return NULL;
    }

    if (!conn->buildFromXML(connectorElement))
    {
        delete conn;
        return NULL;
    }

    return conn;
}

Connector *MountBehavior::getActiveSlaveConnector()
{
    for (std::vector<Connector *>::iterator it = _slaveConnectors.begin(); it != _slaveConnectors.end(); ++it)
    {
        if ((*it)->getMasterConnector() != NULL)
        {
            return (*it);
        }
    }
    return NULL;
}
