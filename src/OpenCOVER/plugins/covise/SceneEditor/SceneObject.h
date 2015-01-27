/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef SCENE_OBJECT_H
#define SCENE_OBJECT_H

#include <string>
#include <vector>

#include "SceneObjectTypes.h"
#include "Behaviors/Behavior.h"
#include "Events/Event.h"

#include <cover/coVRPluginSupport.h>

#include <osg/Node>
#include <osg/Group>

// forward declaration
class Behavior;

class SceneObject
{
public:
    SceneObject();
    virtual ~SceneObject();

    int addChild(SceneObject *);
    int removeChild(SceneObject *);

    int setParent(SceneObject *);
    SceneObject *getParent();

    void setName(std::string n);
    std::string getName();

    int addBehavior(Behavior *b);
    int deleteBehavior(Behavior *b);
    int deleteAllBehaviors();
    Behavior *findBehavior(BehaviorTypes::Type behaviorType);

    int setGeometryNode(osg::ref_ptr<osg::Node> n);
    osg::Node *getGeometryNode();
    osg::Node *getGeometryNode(std::string geoNameSpace);
    osg::Group *getRootNode();

    virtual void setTransform(osg::Matrix trans, osg::Matrix rot, EventSender sender = EventSender());
    virtual void setTranslate(osg::Matrix m, EventSender sender = EventSender());
    virtual void setRotate(osg::Matrix m, EventSender sender = EventSender());
    virtual osg::Matrix getRotate();
    virtual osg::Matrix getTranslate();

    void setCoviseKey(std::string ck);
    std::string getCoviseKey();

    virtual SceneObjectTypes::Type getType();

    virtual EventErrors::Type receiveEvent(Event *e);

    virtual osg::BoundingBox getRotatedBBox();

protected:
    std::string _name;
    std::string _covise_key;
    SceneObjectTypes::Type _type;

    std::vector<Behavior *> _behaviors;

    SceneObject *_parent;
    std::vector<SceneObject *> _children;

    void sendAddChildMessage(SceneObject *so, bool add);

    // Behaviours insert their nodes between _groupNode and _geometryNode
    osg::ref_ptr<osg::Node> _geometryNode;
    osg::ref_ptr<osg::MatrixTransform> _translateNode;
    osg::ref_ptr<osg::MatrixTransform> _rotateNode;

    // userData?
    // std::vector< std::string > _userData;
};

#endif
