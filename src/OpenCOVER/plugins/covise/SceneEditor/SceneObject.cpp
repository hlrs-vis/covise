/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "SceneObject.h"

#include <osgShadow/ShadowedScene>

#include "Events/StartMouseEvent.h"
#include "Events/TransformChangedEvent.h"

#include "SceneObjectManager.h"
#include "ErrorCodes.h"
#include <typeinfo>

#include <iostream>

#include <cover/coVRPluginSupport.h>

#include <cover/coVRMSController.h>
#include <grmsg/coGRObjSelectMsg.h>
#include <grmsg/coGRObjAddChildMsg.h>

#include <OpenVRUI/osg/OSGVruiMatrix.h>
#include <OpenVRUI/osg/OSGVruiTransformNode.h>

#include <osg/ComputeBoundsVisitor>
#include <net/message.h>
#include <net/message_types.h>

SceneObject::SceneObject()
{
    _name = "";
    _type = SceneObjectTypes::NONE;
    _parent = NULL;
    _geometryNode = NULL;
    _translateNode = NULL;
    _rotateNode = NULL;
    _covise_key = "";
}

SceneObject::~SceneObject()
{
    // remove from parent
    if (_parent)
    {
        _parent->removeChild(this);
    }

    // TODO: remove/delete/reassign all children?

    deleteAllBehaviors();

    if (_translateNode)
    {
        opencover::cover->getObjectsRoot()->removeChild(_translateNode.get());
    }
}

SceneObjectTypes::Type SceneObject::getType()
{
    return _type;
}

int SceneObject::addBehavior(Behavior *b)
{
    // search location in behavior list according to type
    std::vector<Behavior *>::iterator it = _behaviors.begin();
    while ((it < _behaviors.end()) && ((*it)->getType() <= b->getType()))
    {
        if ((*it)->getType() == b->getType())
        {
            std::cerr << "Error: Ignoring duplicate behavior" << std::endl;
            return -1;
        }
        ++it;
    }
    _behaviors.insert(it, b);

    b->attach(this);

    return 1;
}

int SceneObject::deleteBehavior(Behavior *b)
{
    if (!b)
        return 0;

    for (std::vector<Behavior *>::iterator it = _behaviors.begin(); it < _behaviors.end(); it++)
    {
        if (*it == b)
        {
            b->detach();
            _behaviors.erase(it);
            delete b;

            return 1;
        }
    }

    // Behavior not found (no child)

    return -1;
}

int SceneObject::deleteAllBehaviors()
{
    // detach and delete all behaviors
    for (std::vector<Behavior *>::iterator it = _behaviors.begin(); it < _behaviors.end(); it++)
    {
        (*it)->detach();
        delete (*it);
    }
    _behaviors.clear();

    return 1;
}

Behavior *SceneObject::findBehavior(BehaviorTypes::Type behaviorType)
{
    for (std::vector<Behavior *>::iterator it = _behaviors.begin(); it < _behaviors.end(); it++)
    {
        if ((*it)->getType() == behaviorType)
        {
            return (*it);
        }
    }
    return NULL;
}

int SceneObject::addChild(SceneObject *so)
{
    if (!so)
        return 0;

    _children.push_back(so);
    so->setParent(this);

    sendAddChildMessage(so, true);

    return 1;
}

int SceneObject::removeChild(SceneObject *so)
{
    if (!so)
        return 0;

    for (std::vector<SceneObject *>::iterator it = _children.begin(); it < _children.end(); it++)
    {
        if (*it == so)
        {
            so->setParent(NULL);
            _children.erase(it);
            sendAddChildMessage(so, false);
            return 1;
        }
    }

    // SceneObject not found (no child)

    return -1;
}

int SceneObject::setParent(SceneObject *so)
{
    if (_parent && so)
    {
        // warning: replacing parent of this scene object with another!
    }

    _parent = so;

    return 0;
}

SceneObject *SceneObject::getParent()
{
    return _parent;
}

EventErrors::Type SceneObject::receiveEvent(Event *e)
{

    // do custom event processing in derived SceneObject classes

    int numSuccess = 0;
    int numFail = 0;

    // send select / deselect event to gui (is this the best place for that?)
    if (e->getType() == EventTypes::SELECT_EVENT || e->getType() == EventTypes::DESELECT_EVENT)
    {
        int select = 0;
        if (e->getType() == EventTypes::SELECT_EVENT)
            select = 1;
        if (opencover::coVRMSController::instance()->isMaster())
        {
            grmsg::coGRObjSelectMsg selectMsg(_covise_key.c_str(), select);
            Message grmsg{ COVISE_MESSAGE_UI, DataHandle{(char*)selectMsg.c_str(),strlen(selectMsg.c_str()) + 1 , false} };
            opencover::cover->sendVrbMessage(&grmsg);
        }
    }

    for (int i = 0; i < _behaviors.size(); i++)
    {
        EventErrors::Type err;
        err = _behaviors[i]->receiveEvent(e);

        switch (err)
        {
        case EventErrors::SUCCESS:
            e->setHandled();
            numSuccess++;
            break;
        case EventErrors::FAIL:
            e->setHandled();
            numFail++;
            break;
        case EventErrors::UNHANDLED:
            break;
        default:
            break;
        }
    }

    if ((numSuccess == 0) && (numFail == 0))
    {
        return EventErrors::UNHANDLED;
    }
    else if ((numSuccess > 0) && (numFail == 0))
    {
        return EventErrors::SUCCESS;
    }
    else if ((numSuccess == 0) && (numFail > 0))
    {
        return EventErrors::FAIL;
    }
    else
    {
        std::cerr << "Warning: Behaviors returning SUCCESS and FAILURE during event handling." << std::endl;
        std::cerr << "SceneObject type: " << this->getType() << ", Event type: " << e->getType() << std::endl;
        std::cerr << "SceneObject class name: " << typeid(this).name() << ", Event class name: " << typeid(e).name() << std::endl;

        return EventErrors::EPIC_FAIL;
    }
}

void SceneObject::setName(std::string n)
{
    _name = n;
}

std::string SceneObject::getName()
{
    return _name;
}

int SceneObject::setGeometryNode(osg::ref_ptr<osg::Node> n)
{
    if (_geometryNode)
    {
        if (_geometryNode->getNumParents() == 1)
        {
            osg::Group *parent = _geometryNode->getParent(0);
            parent->removeChild(_geometryNode.get());
            if (n != NULL)
            {
                parent->addChild(n.get());
            }
        }
    }
    else
    {
        if (!_translateNode)
        {
            _translateNode = new osg::MatrixTransform();
            opencover::cover->getObjectsRoot()->addChild(_translateNode.get());
            _rotateNode = new osg::MatrixTransform();
            _translateNode->addChild(_rotateNode.get());
        }
        _rotateNode->addChild(n);
    }

    _geometryNode = n;
    _geometryNode->getOrCreateStateSet(); // we need a stateset for shadowedSceene

    return 1;
}

osg::Node *SceneObject::getGeometryNode()
{
    return _geometryNode.get();
}

osg::Node *SceneObject::getGeometryNode(std::string geoNameSpace)
{
    if (geoNameSpace == "")
    {
        return _geometryNode.get();
    }
    osg::Group *group = _geometryNode->asGroup();
    if (group != NULL)
    {
        for (int i = 0; i < group->getNumChildren(); i++)
        {
            osg::Node *child = group->getChild(i);
            if (child->getName() == "NAMESPACE:" + geoNameSpace)
            {
                return child;
            }
        }
    }
    return NULL;
}

osg::Group *SceneObject::getRootNode()
{
    return _translateNode.get();
}

void SceneObject::setTransform(osg::Matrix trans, osg::Matrix rot, EventSender sender)
{
    _translateNode->setMatrix(trans);
    _rotateNode->setMatrix(rot);

    TransformChangedEvent tce;
    tce.setSender(sender);
    receiveEvent(&tce);
}

void SceneObject::setTranslate(osg::Matrix m, EventSender sender)
{
    _translateNode->setMatrix(m);

    TransformChangedEvent tce;
    tce.setSender(sender);
    receiveEvent(&tce);
}

osg::Matrix SceneObject::getTranslate()
{
    return _translateNode->getMatrix();
}

void SceneObject::setRotate(osg::Matrix m, EventSender sender)
{
    _rotateNode->setMatrix(m);

    TransformChangedEvent tce;
    tce.setSender(sender);
    receiveEvent(&tce);
}

osg::Matrix SceneObject::getRotate()
{
    return _rotateNode->getMatrix();
}

osg::BoundingBox SceneObject::getRotatedBBox()
{
    if (_rotateNode)
    {
        osg::ComputeBoundsVisitor bb;
        _rotateNode->accept(bb);
        return bb.getBoundingBox();
    }
    else
    {
        return osg::BoundingBox();
    }
}

void SceneObject::setCoviseKey(std::string ck)
{
    _covise_key = ck;
}

std::string SceneObject::getCoviseKey()
{
    return _covise_key;
}

void SceneObject::sendAddChildMessage(SceneObject *so, bool add)
{
    if (opencover::coVRMSController::instance()->isMaster())
    {
        int remove = 0;
        if (!add)
            remove = 1;
        grmsg::coGRObjAddChildMsg childMsg(_covise_key.c_str(), so->getCoviseKey().c_str(), remove);
        Message grmsg{ COVISE_MESSAGE_UI, DataHanle{(char*)childMsg.c_str(),strlen(childMsg.c_str()) + 1 , false} };
        opencover::cover->sendVrbMessage(&grmsg);
    }
}
