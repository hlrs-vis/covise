/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef KINEMATICS_BEHAVIOR_H
#define KINEMATICS_BEHAVIOR_H

#include "Behavior.h"
#include "../Events/PreFrameEvent.h"
#include "KardanikXML/MotionStateFactory.h"
#include "KardanikXML/KardanikConstructor.h"
#include <PluginUtil/coPlane.h>

#include <memory>
#include <tuple>
#include <osg/Material>
#include <osg/Vec3>
#include <osg/Matrix>
#include <osg/Node>

#include <string>
#include <map>
#include <set>

class btDynamicsWorld;
class btDefaultCollisionConfiguration;
class btCollisionDispatcher;
struct btDbvtBroadphase;
class btSequentialImpulseConstraintSolver;
class btClock;
class btMotionState;
class btPoint2PointConstraint;

class coPlane;

namespace osg
{
class Node;
class MatrixTransform;
}

namespace KardanikXML
{
class Construction;
class Body;
class ConstructionParser;
class Point;
class MotionState;
}

class KardanikConstructor;

class KinematicsBehavior : public Behavior, KardanikXML::MotionStateFactory
{
public:
    KinematicsBehavior();
    virtual ~KinematicsBehavior();

    virtual int attach(SceneObject *);
    virtual int detach();

    virtual EventErrors::Type receiveEvent(Event *e);

    virtual bool buildFromXML(QDomElement *behaviorElement);

protected:
    // from KardanikXML::MotionStateFactory
    KardanikXML::MotionState *CreateMotionStateForBody(std::shared_ptr<KardanikXML::Body> body, const btTransform &centerOfMass);

private:
    typedef std::map<std::string, osg::Node *> CollectedNodes;

    void AttachConstraintToBody(std::shared_ptr<KardanikXML::Body> body, const osg::Vec3f &worldPositionAttach);
    void MoveConstraintOpposite(const osg::Vec3f &oppositePoint);
    void RemoveConstraintFromBody();
    std::shared_ptr<KardanikXML::Body> GetBodyFromNodePath(const osg::NodePathList &nodePathList) const;
    std::tuple<std::shared_ptr<KardanikXML::Body>, osg::Vec3f> GetBodyAndPickPosition() const;

    CollectedNodes CollectAnchorNodes(osg::Node *node, std::shared_ptr<KardanikXML::ConstructionParser> parser);
    void CollectAnchorNodesSubtree(osg::Node *node, CollectedNodes &collectedNodes, std::shared_ptr<KardanikXML::ConstructionParser> parse);
    void CreateDynamicsWorld();
    void DestroyDynamicsWorld();
    float getDeltaTimeMicroseconds() const;

    osg::MatrixTransform *ReplaceNodeWithTransform(osg::Node *node);
    osg::MatrixTransform *findAnimatableParentNode(osg::Node *node);

    void setOperatingRangesVisible(bool visible);
    void initPhysicsSimulation();
    void initTransformsFromState(const std::string &state);

    std::shared_ptr<KardanikXML::Body> LookupBodyByNamespace(std::string theNamespace, std::string theID);
    std::shared_ptr<KardanikXML::Point> LookupPointByNamespace(std::string theNamespace, std::string theID);

    typedef std::map<std::string, osg::MatrixTransform *> AnchorNodesMap;

    struct ConstructionInfo
    {
        std::shared_ptr<KardanikXML::ConstructionParser> parser;
        AnchorNodesMap anchorNodesMap;
    };

    void CreateAnchorNodesMap(osg::Node *rootNode, ConstructionInfo &constructionInfo);

    void sendStateToGui();
    std::string makeStateForBodyTransforms(const KardanikConstructor::BodyTransforms &transforms) const;

    typedef std::vector<ConstructionInfo> ConstructionInfos;
    ConstructionInfos m_ConstructionInfos;

    btDynamicsWorld *m_DynamicsWorld;
    std::shared_ptr<KardanikConstructor> m_KardanikConstructor;
    btDefaultCollisionConfiguration *m_collisionConfiguration;
    btCollisionDispatcher *m_dispatcher;
    btDbvtBroadphase *m_overlappingPairCache;
    btSequentialImpulseConstraintSolver *m_constraintSolver;
    std::shared_ptr<btClock> m_Clock;
    bool m_ClockInitialized;

    osg::Vec3 m_startPickPos;
    std::shared_ptr<opencover::coPlane> m_DragPlane;

    btPoint2PointConstraint *m_PickConstraint;
    std::shared_ptr<KardanikXML::Body> m_PickedBody;

    osg::NodePath m_PickedNodePath;
    osg::Node *m_PickedNode;
    mutable osg::Matrixf m_WorldToSceneMatrix;
    bool m_AnimationRunning;

    int m_feMask;

public:
    void SetWorldTransformOfAnchor(osg::MatrixTransform *anchorNode, const osg::Matrixf &anchorWorldTransform);
    osg::Matrixf GetWorldTransformOfAnchor(osg::MatrixTransform *anchorNode);
    osg::MatrixTransform *GetAnchorNodeByName(const std::string &nodeName, std::shared_ptr<KardanikXML::Construction> construction) const;
};

#endif
