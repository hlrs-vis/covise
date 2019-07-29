/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "KinematicsBehavior.h"

#include <iostream>
#include <fstream>
#include <algorithm>
#include <sstream>
#include <string>
#include <iomanip>

#include <cover/coVRPluginSupport.h>
#include <cover/coVRNavigationManager.h>
#include <PluginUtil/coPlane.h>

#include <appl/RenderInterface.h>
#include <cover/coVRMSController.h>
#include <covise/covise_appproc.h>
#include <grmsg/coGRObjKinematicsStateMsg.h>

#include <osg/Node>
#include <osg/Group>
#include <osg/Switch>

#include <osg/MatrixTransform>
#include <osgDB/ReadFile>
#include <osg/ShapeDrawable>

#include <boost/foreach.hpp>
#include <boost/tokenizer.hpp>
#include <boost/lexical_cast.hpp>
#include <bullet/btBulletDynamicsCommon.h>

#include "KardanikXML/Anchor.h"
#include "KardanikXML/Point.h"
#include "KardanikXML/Construction.h"
#include "KardanikXML/ConstructionParser.h"

#include "KardanikXML/Body.h"
#include "KardanikXML/KardanikConstructor.h"
#include "KardanikXML/MotionStateFactory.h"
#include "KardanikXML/ConstructionParser.h"
#include "KardanikXML/OperatingRange.h"
#include "KardanikXML/MotionState.h"

#include "../Events/StartMouseEvent.h"
#include "../Events/DoMouseEvent.h"
#include "../Events/InitKinematicsStateEvent.h"
#include "../SceneObject.h"

#include "../Settings.h"

using namespace std;

// #define KARDANIK_DEBUG_FPE 1
#ifdef KARDANIK_DEBUG_FPE
#include <fenv.h>
#endif

namespace
{

ostream &operator<<(ostream &os, const btVector3 &vec)
{
    os << "[Vector3: " << vec[0] << ", " << vec[1] << ", " << vec[2] << "]";
    return os;
}

ostream &operator<<(ostream &os, const btMatrix3x3 &matrix)
{
    os << "[Matrix3x3: " << endl;
    os << matrix.getRow(0) << endl;
    os << matrix.getRow(1) << endl;
    os << matrix.getRow(2) << endl;
    os << "]" << endl;
    return os;
}

string btTransformToString(const btTransform &transform)
{
    ostringstream outstring;
    const btVector3 &origin = transform.getOrigin();
    const btMatrix3x3 &basis = transform.getBasis();

    outstring << origin[0] << ",";
    outstring << origin[1] << ",";
    outstring << origin[2] << ",";
    outstring << basis[0][0] << ",";
    outstring << basis[0][1] << ",";
    outstring << basis[0][2] << ",";
    outstring << basis[1][0] << ",";
    outstring << basis[1][1] << ",";
    outstring << basis[1][2] << ",";
    outstring << basis[2][0] << ",";
    outstring << basis[2][1] << ",";
    outstring << basis[2][2];
    return outstring.str();
}

btTransform stringToBtTransform(const string &stateString)
{
    typedef boost::tokenizer<boost::char_separator<char> >
        tokenizer;
    boost::char_separator<char> sep(",");
    tokenizer tokens(stateString, sep);
    vector<float> values;

    for (tokenizer::iterator tok_iter = tokens.begin();
         tok_iter != tokens.end(); ++tok_iter)
    {
        values.push_back(boost::lexical_cast<float>(*tok_iter));
    }
    btTransform transform;
    btVector3 origin;
    origin[0] = values[0];
    origin[1] = values[1];
    origin[2] = values[2];

    btMatrix3x3 basis;
    basis[0][0] = values[3];
    basis[0][1] = values[4];
    basis[0][2] = values[5];
    basis[1][0] = values[6];
    basis[1][1] = values[7];
    basis[1][2] = values[8];
    basis[2][0] = values[9];
    basis[2][1] = values[10];
    basis[2][2] = values[11];

    transform.setOrigin(origin);
    transform.setBasis(basis);

    return transform;
}

ostream &operator<<(ostream &os, const btTransform &transform)
{
    os << transform.getBasis() << endl;
    os << transform.getOrigin() << endl;
    return os;
}

ostream &operator<<(ostream &os, const osg::Vec3f &vec3)
{
    os << "[osg::Vec3f: " << vec3[0] << ", " << vec3[1] << ", " << vec3[2] << "] " << endl;
    return os;
}

ostream &operator<<(ostream &os, const osg::Matrixf &transform)
{
    os << "[osg::Matrixf: " << endl;
    os << "[ " << transform(0, 0) << ", " << transform(0, 1) << ", " << transform(0, 2) << ", " << transform(0, 3) << "]" << endl;
    os << "[ " << transform(1, 0) << ", " << transform(1, 1) << ", " << transform(1, 2) << ", " << transform(1, 3) << "]" << endl;
    os << "[ " << transform(2, 0) << ", " << transform(2, 1) << ", " << transform(2, 2) << ", " << transform(2, 3) << "]" << endl;
    os << "[ " << transform(3, 0) << ", " << transform(3, 1) << ", " << transform(3, 2) << ", " << transform(3, 3) << "]" << endl;
    os << "]" << endl;
    return os;
}

ostream &operator<<(ostream &os, const KardanikXML::Point &point)
{
    os << "[KardakinXML::Point: " << point.GetX() << ", " << point.GetY() << ", " << point.GetZ() << "]";
    return os;
}

osg::Matrixf BulletToOSG(const btTransform &bulletTrans)
{
    // btTransform(btQuaternion(btVector3(1,0,0),  M_PI/2.0)) *
    btTransform bulletGraphicsWorldTrans = bulletTrans;
    static float oglMatrix[16];
    bulletGraphicsWorldTrans.getOpenGLMatrix(oglMatrix);
    osg::Matrixf osgTransform;
    osgTransform.set(oglMatrix);

    // m to mm
    osg::Vec3f trans = osgTransform.getTrans();
    trans *= 1000.0f;
    osgTransform.setTrans(trans);
    return osgTransform;
}

osg::ref_ptr<osg::Drawable> CreateCircle(float radius, const osg::Vec3 & /* axis */, const osg::Vec4f &color, unsigned int segments = 50u)
{
    osg::ref_ptr<osg::Geometry> geom(new osg::Geometry());
    osg::ref_ptr<osg::Vec3Array> circleVertices = new osg::Vec3Array;
    osg::ref_ptr<osg::Vec3Array> crossVertices = new osg::Vec3Array;

    osg::ref_ptr<osg::Vec4Array> colors = new osg::Vec4Array;
    osg::ref_ptr<osg::Vec3Array> normals = new osg::Vec3Array;

    radius *= 1000;

    float delta = 2.0 * M_PI / segments;
    float alpha = 0.0f;
    while (alpha < 2 * M_PI)
    {
        circleVertices->push_back(osg::Vec3f(cosf(alpha) * radius, sinf(alpha) * radius, 0.0f));
        alpha += delta;
    }
    circleVertices->push_back(osg::Vec3f(1 * radius, 0.0f, 0.0));
    geom->setVertexArray(circleVertices.get());

    colors->push_back(color);
    geom->setColorArray(colors.get());
    geom->setColorBinding(osg::Geometry::BIND_OVERALL);

    normals->push_back(osg::Vec3f(0.0, 0.0, 1.0));
    geom->setNormalArray(normals.get());
    geom->setNormalBinding(osg::Geometry::BIND_OVERALL);

    geom->addPrimitiveSet(new osg::DrawArrays(GL_LINE_STRIP, 0, circleVertices->getNumElements()));

    osg::ref_ptr<osg::StateSet> stateSet(new osg::StateSet());
    stateSet->setMode(GL_LIGHTING, 0);

    geom->setStateSet(stateSet);

    return geom;
}

} // end unnamed namespace

KinematicsBehavior::KinematicsBehavior()
    : m_DynamicsWorld(0)
    , m_collisionConfiguration(0)
    , m_dispatcher(0)
    , m_overlappingPairCache(0)
    , m_constraintSolver(0)
    , m_ClockInitialized(false)
    , m_DragPlane(new opencover::coPlane(osg::Vec3(0.0, 1.0, 0.0), osg::Vec3(0.0, 0.0, 0.0)))
    , m_PickConstraint(0)
    , m_PickedNode(0)
    , m_AnimationRunning(false)
{
    _type = BehaviorTypes::KINEMATICS_BEHAVIOR;
    setEnabled(false);
}

KinematicsBehavior::~KinematicsBehavior()
{
}

int KinematicsBehavior::attach(SceneObject *so)
{
#ifdef KARDANIK_DEBUG_FPE
    int m_feMask = fegetexcept();
    feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
#endif

    // connects this behavior to its scene object
    Behavior::attach(so);

    CreateDynamicsWorld();
    m_KardanikConstructor.reset(new KardanikConstructor());
    m_KardanikConstructor->SetMotionStateFactory(this);
    m_KardanikConstructor->SetDynamicWorld(m_DynamicsWorld);

    return 1;
}

int KinematicsBehavior::detach()
{

    m_KardanikConstructor.reset();
    DestroyDynamicsWorld();
    m_DynamicsWorld = 0;
    m_Clock.reset();
    m_ClockInitialized = false;

    Behavior::detach();

#ifdef KARDANIK_DEBUG_FPE
    feenableexcept(m_feMask);
#endif

    return 1;
}

void KinematicsBehavior::CreateAnchorNodesMap(osg::Node *rootNode, ConstructionInfo &constructionInfo)
{
    CollectedNodes namedNodes = CollectAnchorNodes(rootNode, constructionInfo.parser);
    BOOST_FOREACH (const CollectedNodes::value_type &nodeName, namedNodes)
    {
        osg::MatrixTransform *anchorNode = ReplaceNodeWithTransform(nodeName.second);
        constructionInfo.anchorNodesMap.insert(AnchorNodesMap::value_type(nodeName.first, anchorNode));
    }
}

KinematicsBehavior::CollectedNodes KinematicsBehavior::CollectAnchorNodes(osg::Node *rootNode, std::shared_ptr<KardanikXML::ConstructionParser> parser)
{
    CollectedNodes collectedNodes;
    CollectAnchorNodesSubtree(rootNode, collectedNodes, parser);
    return collectedNodes;
}

void KinematicsBehavior::CollectAnchorNodesSubtree(osg::Node *node, CollectedNodes &collectedNodes, std::shared_ptr<KardanikXML::ConstructionParser> parser)
{
    if (!node)
    {
        return;
    }
    if (parser->hasAnchorNodeWithName(node->getName()))
    {
        collectedNodes.insert(CollectedNodes::value_type(node->getName(), node));
    }
    osg::Group *group = node->asGroup();
    if (group)
    {
        unsigned int numChildren = group->getNumChildren();
        for (unsigned int i = 0U; i < numChildren; ++i)
        {
            osg::Node *childNode = group->getChild(i);
            if (!childNode)
            {
                continue;
            }
            CollectAnchorNodesSubtree(childNode, collectedNodes, parser);
        }
    }
}

osg::MatrixTransform *KinematicsBehavior::ReplaceNodeWithTransform(osg::Node *node)
{
    if (!node)
    {
        return 0;
    }
    osg::ref_ptr<osg::MatrixTransform> transform = new osg::MatrixTransform();

    osg::Group *group = node->asGroup();
    if (group)
    {
        unsigned int numChildren = group->getNumChildren();
        for (unsigned int i = 0U; i < numChildren; ++i)
        {
            transform->addChild(group->getChild(i));
        }
        while (group->getNumChildren() > 0)
        {
            group->removeChild(0U, 1U);
        }
        group->addChild(transform);
    }
    return transform;
}

void KinematicsBehavior::CreateDynamicsWorld()
{
    m_collisionConfiguration = new btDefaultCollisionConfiguration();
    m_dispatcher = new btCollisionDispatcher(m_collisionConfiguration);
    m_overlappingPairCache = new btDbvtBroadphase();
    m_constraintSolver = new btSequentialImpulseConstraintSolver();
    m_DynamicsWorld = new btDiscreteDynamicsWorld(m_dispatcher, m_overlappingPairCache, m_constraintSolver,
                                                  m_collisionConfiguration);
    btContactSolverInfo &info = m_DynamicsWorld->getSolverInfo();
    info.m_numIterations = 200; //more iterations will allow the solver to converge to the actual solution better
    info.m_solverMode = SOLVER_RANDMIZE_ORDER;
    info.m_maxErrorReduction = 1e30f;
    info.m_erp = .8f;
    info.m_erp2 = .8f;
    info.m_damping = 1;

    m_DynamicsWorld->setGravity(btVector3(0, 0, 0));
}

void KinematicsBehavior::DestroyDynamicsWorld()
{
    delete m_DynamicsWorld;
    m_DynamicsWorld = 0;
    delete m_constraintSolver;
    m_constraintSolver = 0;
    delete m_overlappingPairCache;
    m_overlappingPairCache = 0;
    delete m_dispatcher;
    m_dispatcher = 0;
    delete m_collisionConfiguration;
    m_collisionConfiguration = 0;
}

osg::MatrixTransform *KinematicsBehavior::findAnimatableParentNode(osg::Node *baseNode)
{
    if (!baseNode)
    {
        return 0;
    }
    BOOST_FOREACH (const osg::NodePath &nodePath, baseNode->getParentalNodePaths(_sceneObject->getGeometryNode()))
    {
        BOOST_FOREACH (osg::Node *node, nodePath)
        {
            BOOST_FOREACH (const ConstructionInfo &cInfo, m_ConstructionInfos)
            {
                BOOST_FOREACH (const AnchorNodesMap::value_type &anchorNode, cInfo.anchorNodesMap)
                {
                    osg::Node *aNode = anchorNode.second;
                    if (aNode == node)
                    {
                        return anchorNode.second;
                    }
                }
            }
        }
    }
    return 0;
}

namespace
{
void TestPicking()
{
    const osg::NodePath &nodePath = opencover::cover->getIntersectedNodePath();
    osg::Vec3f hitPointWorld = opencover::cover->getIntersectionHitPointWorld();

    osg::Matrixf worldTransformNode = osg::computeLocalToWorld(nodePath);
    osg::Matrixf worldTransformNodeInv = osg::Matrixf::inverse(worldTransformNode);
    osg::Vec3f hitPointObject = hitPointWorld * worldTransformNodeInv;
}
}

EventErrors::Type KinematicsBehavior::receiveEvent(Event *e)
{
    if (!e)
    {
        return EventErrors::UNHANDLED;
    }

    if (e->getType() == EventTypes::PRE_FRAME_EVENT)
    {

        if (_isEnabled && m_AnimationRunning)
        {
            if (!m_ClockInitialized)
            {
                m_ClockInitialized = true;
                m_Clock->reset();
            }
            float dt = float(getDeltaTimeMicroseconds()) * 0.000001f;
            if (m_DynamicsWorld)
            {
                m_DynamicsWorld->stepSimulation(dt, 10, 1.0 / 240.0);
            }
        }

        return EventErrors::UNHANDLED;
    }
    else if (e->getType() == EventTypes::START_MOUSE_EVENT)
    {
        bool mouseAPressed = dynamic_cast<StartMouseEvent *>(e)->getMouseButton() == MouseEvent::TYPE_BUTTON_A;
        bool mouseCPressed = dynamic_cast<StartMouseEvent *>(e)->getMouseButton() == MouseEvent::TYPE_BUTTON_C;

        if (mouseAPressed || mouseCPressed)
        {

            setEnabled(true);

            m_PickedNode = opencover::cover->getIntersectedNode();
            if (!m_PickedNode)
            {
                return EventErrors::FAIL;
            }

            m_startPickPos = opencover::cover->getIntersectionHitPointWorld();
            m_PickedNodePath = opencover::cover->getIntersectedNodePath();

            // move planes to startPickPos (dont adjust the normal!)
            m_DragPlane->update(m_DragPlane->getNormal(), m_startPickPos);

            std::tuple<std::shared_ptr<KardanikXML::Body>, osg::Vec3f> bodyInfo = GetBodyAndPickPosition();
            std::shared_ptr<KardanikXML::Body> body = get<0>(bodyInfo);
            if (!body || body->GetMotionType() == KardanikXML::Body::MOTION_STATIC)
            {
                m_AnimationRunning = false;
                return EventErrors::UNHANDLED;
            }
            else
            {
                m_AnimationRunning = true;
            }

            AttachConstraintToBody(get<0>(bodyInfo), get<1>(bodyInfo));
            m_KardanikConstructor->MakeHigherBodiesStatic(get<0>(bodyInfo), mouseCPressed);

            return EventErrors::SUCCESS;
        }
    }
    else if (e->getType() == EventTypes::STOP_MOUSE_EVENT)
    {
        m_PickedNode = 0;
        m_PickedNodePath.clear();
        RemoveConstraintFromBody();
        m_KardanikConstructor->ResetBodyMotions();
        m_AnimationRunning = false;
        setEnabled(false);
        sendStateToGui();
        return EventErrors::SUCCESS;
    }
    else if (e->getType() == EventTypes::DO_MOUSE_EVENT && _isEnabled && m_AnimationRunning)
    {
        if (dynamic_cast<DoMouseEvent *>(e)->getMouseButton() == MouseEvent::TYPE_BUTTON_A || dynamic_cast<DoMouseEvent *>(e)->getMouseButton() == MouseEvent::TYPE_BUTTON_C)
        {
            osg::Matrix pointerMat;
            osg::Vec3 pointerPos1, pointerPos2, pointerDir; // defining a line
            osg::Vec3 planeIntersectionPoint;

            pointerMat = opencover::cover->getPointerMat();
            pointerPos1 = pointerMat.getTrans();
            pointerDir[0] = pointerMat(1, 0);
            pointerDir[1] = pointerMat(1, 1);
            pointerDir[2] = pointerMat(1, 2);

            pointerPos2 = pointerPos1 + pointerDir;

            if (m_DragPlane->getLineIntersectionPoint(pointerPos1, pointerPos2, planeIntersectionPoint))
            {
                MoveConstraintOpposite(planeIntersectionPoint);
                osg::Vec3 transVec = planeIntersectionPoint - m_startPickPos;
            }
        }
        return EventErrors::SUCCESS;
    }
    else if (e->getType() == EventTypes::SETTINGS_CHANGED_EVENT)
    {
        setOperatingRangesVisible(Settings::instance()->isOperatingRangeVisible());
        return EventErrors::SUCCESS;
    }
    else if (e->getType() == EventTypes::INIT_KINEMATICS_STATE_EVENT)
    {
        InitKinematicsStateEvent *ikse = dynamic_cast<InitKinematicsStateEvent *>(e);
        std::string state = ikse->getState();

        if (!state.empty())
        {
            initPhysicsSimulation();
            m_KardanikConstructor->SetJointLimits();
            initTransformsFromState(state);
        }
        else
        {
            initPhysicsSimulation();
            m_KardanikConstructor->MoveJointsToInitialPositions();
            m_KardanikConstructor->SetJointLimits();
        }
    }
    return EventErrors::UNHANDLED;
}

namespace
{

KardanikConstructor::BodyTransforms::value_type parseBodyTransform(const std::string &bodyString)
{
    typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
    boost::char_separator<char> sep("{");
    tokenizer tokens(bodyString, sep);
    tokenizer::iterator tok_iter = tokens.begin();

    string bodyName = *tok_iter;
    tok_iter++;
    string transform = *tok_iter;
    btTransform bodyTrans = stringToBtTransform(transform);

    return KardanikConstructor::BodyTransforms::value_type(bodyName, bodyTrans);
}
}
void KinematicsBehavior::initTransformsFromState(const std::string &state)
{
    KardanikConstructor::BodyTransforms bodyTransforms;
    typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
    boost::char_separator<char> sep("}");
    tokenizer tokens(state, sep);

    for (tokenizer::iterator tok_iter = tokens.begin();
         tok_iter != tokens.end(); ++tok_iter)
    {
        KardanikConstructor::BodyTransforms::value_type bodyTransform = parseBodyTransform(*tok_iter);
        bodyTransforms.insert(bodyTransform);
    }
    m_KardanikConstructor->SetInitialBodyTransforms(bodyTransforms);
    m_DynamicsWorld->stepSimulation(5.0 / 50.0, 10, 1.0 / 240.0);
}

float KinematicsBehavior::getDeltaTimeMicroseconds() const
{
    float dt = static_cast<float>(m_Clock->getTimeMicroseconds());
    m_Clock->reset();
    return dt;
}

void KinematicsBehavior::AttachConstraintToBody(std::shared_ptr<KardanikXML::Body> body, const osg::Vec3f &scenePositionAttach)
{
    // get rigidbody for body
    btRigidBody *rigidBody = m_KardanikConstructor->GetRigidBody(body);
    if (!rigidBody)
    {
        return;
    }

    btVector3 position(scenePositionAttach[0], scenePositionAttach[1], scenePositionAttach[2]);
    position *= .001;
    // transform world point to point relative to center of mass of rigid body
    btTransform com = rigidBody->getCenterOfMassTransform();
    //    btVector3 positionLocal = com.inverse() * btTransform(btQuaternion(btVector3(1,0,0), -1.0 * M_PI/2.0)) * position;
    btVector3 positionLocal = com.inverse() * position;

    // create p2p constraint and attach rigidbody  to it

    btPoint2PointConstraint *p2p = new btPoint2PointConstraint(*rigidBody, positionLocal);
    m_DynamicsWorld->addConstraint(p2p);
    m_PickConstraint = p2p;
    p2p->m_setting.m_impulseClamp = .9;
    //very weak constraint for picking
    p2p->m_setting.m_tau = 0.3f;
    m_PickedBody = body;
}

void KinematicsBehavior::MoveConstraintOpposite(const osg::Vec3f &oppositePoint)
{
    if (!m_PickConstraint)
        return;

    osg::Vec3 oppositePointLocal = oppositePoint * m_WorldToSceneMatrix;

    btVector3 pivotB(oppositePointLocal[0], oppositePointLocal[1], oppositePointLocal[2]);
    pivotB *= 0.001;

    //    pivotB = btTransform(btQuaternion(btVector3(1,0,0), -1.0 * M_PI/2.0)) * pivotB;
    pivotB = pivotB;

    m_PickConstraint->setPivotB(pivotB);
}

void KinematicsBehavior::RemoveConstraintFromBody()
{
    if (!m_PickConstraint || !m_DynamicsWorld)
        return;

    m_DynamicsWorld->removeConstraint(m_PickConstraint);
    delete m_PickConstraint;
    m_PickConstraint = 0;
}

std::shared_ptr<KardanikXML::Body> KinematicsBehavior::GetBodyFromNodePath(const osg::NodePathList &nodePathList) const
{
    BOOST_FOREACH (const osg::NodePath &nodePath, nodePathList)
    {
        BOOST_FOREACH (osg::Node *node, nodePath)
        {
            BOOST_FOREACH (const ConstructionInfo &cInfo, m_ConstructionInfos)
            {
                BOOST_FOREACH (const AnchorNodesMap::value_type &anchorNode, cInfo.anchorNodesMap)
                {
                    osg::Node *aNode = anchorNode.second;
                    if (aNode == node)
                    {
                        return cInfo.parser->getBodyForAnchor(anchorNode.first);
                    }
                }
            }
        }
    }
    return std::shared_ptr<KardanikXML::Body>();
}

std::tuple<std::shared_ptr<KardanikXML::Body>, osg::Vec3f> KinematicsBehavior::GetBodyAndPickPosition() const
{
    std::shared_ptr<KardanikXML::Body> body = GetBodyFromNodePath(m_PickedNode->getParentalNodePaths(_sceneObject->getGeometryNode()));
    if (!body)
    {
        return std::make_tuple(std::shared_ptr<KardanikXML::Body>(), osg::Vec3f());
    }

    osg::Matrixf worldTransformNode = osg::computeLocalToWorld(m_PickedNodePath);
    osg::Matrixf worldTransformNodeInv = osg::Matrixf::inverse(worldTransformNode);

    osg::MatrixList matrices = m_PickedNode->getWorldMatrices(_sceneObject->getGeometryNode());
    osg::Matrixf sceneMatrix = matrices[0];
    m_WorldToSceneMatrix = worldTransformNodeInv * sceneMatrix;

    osg::Vec3f pickPosScene = m_startPickPos * m_WorldToSceneMatrix;
    return std::make_tuple(body, pickPosScene);
}

bool KinematicsBehavior::buildFromXML(QDomElement *behaviorElement)
{
    QDomElement construction = behaviorElement->firstChildElement("construction");
    if (construction.isNull())
    {
        return false;
    }

    while (!construction.isNull())
    {
        ConstructionInfo cInfo;
        cInfo.parser.reset(new KardanikXML::ConstructionParser);

        cInfo.parser->setBodyNamespaceLookupCallback(std::bind(&KinematicsBehavior::LookupBodyByNamespace, this, std::placeholders::_1, std::placeholders::_2));
        cInfo.parser->setPointNamespaceLookupCallback(std::bind(&KinematicsBehavior::LookupPointByNamespace, this, std::placeholders::_1, std::placeholders::_2));

        cInfo.parser->parseConstructionElement(construction);
        m_ConstructionInfos.push_back(cInfo);
        construction = construction.nextSiblingElement("construction");
    }
    return true;
}

void KinematicsBehavior::SetWorldTransformOfAnchor(osg::MatrixTransform *anchorNode, const osg::Matrixf &anchorWorldTransform)
{
    if (!anchorNode)
    {
        return;
    }

    osg::Group *parent = anchorNode->getParent(0);
    if (!parent)
    {
        return;
    }

    osg::MatrixList matrices = parent->getWorldMatrices(_sceneObject->getGeometryNode());
    osg::Matrixf worldMat;

    if (!matrices.empty())
    {
        worldMat = matrices[0];
    }
    else
    {
        worldMat = osg::Matrixf::identity();
    }

    osg::Matrixf invWorldMat = osg::Matrixf::inverse(worldMat);
    anchorNode->setMatrix(anchorWorldTransform * invWorldMat);
}

osg::Matrixf KinematicsBehavior::GetWorldTransformOfAnchor(osg::MatrixTransform *anchorNode)
{
    osg::MatrixList matrices = anchorNode->getWorldMatrices(_sceneObject->getGeometryNode());
    osg::Matrixf worldMat;

    if (!matrices.empty())
    {
        worldMat = matrices[0];
    }
    else
    {
        worldMat = osg::Matrixf::identity();
    }
    return worldMat * anchorNode->getMatrix();
}

void KinematicsBehavior::setOperatingRangesVisible(bool visible)
{
    BOOST_FOREACH (const ConstructionInfo &cInfo, m_ConstructionInfos)
    {
        BOOST_FOREACH (const AnchorNodesMap::value_type &anchorNodePair, cInfo.anchorNodesMap)
        {
            osg::MatrixTransform *anchorNode = anchorNodePair.second;
            if (!anchorNode)
            {
                continue;
            }
            if (anchorNode->getNumChildren() < 2)
            {
                continue;
            }
            osg::Switch *rangeSwitch = anchorNode->getChild(1)->asSwitch();
            if (!rangeSwitch)
            {
                continue;
            }
            if (visible)
            {
                rangeSwitch->setAllChildrenOn();
            }
            else
            {
                rangeSwitch->setAllChildrenOff();
            }
        }
    }
}

std::shared_ptr<KardanikXML::Body> KinematicsBehavior::LookupBodyByNamespace(std::string theNamespace, std::string theID)
{
    BOOST_FOREACH (const ConstructionInfo &cInfo, m_ConstructionInfos)
    {
        if (cInfo.parser->getConstruction()->GetNamespace() != theNamespace)
        {
            continue;
        }
        return cInfo.parser->getBodyByNameLocal(theID);
    }
    return std::shared_ptr<KardanikXML::Body>();
}

std::shared_ptr<KardanikXML::Point> KinematicsBehavior::LookupPointByNamespace(std::string theNamespace, std::string theID)
{
    BOOST_FOREACH (const ConstructionInfo &cInfo, m_ConstructionInfos)
    {
        if (cInfo.parser->getConstruction()->GetNamespace() != theNamespace)
        {
            continue;
        }
        return cInfo.parser->getPointByNameLocal(theID);
    }
    return std::shared_ptr<KardanikXML::Point>();
}

osg::MatrixTransform *
KinematicsBehavior::GetAnchorNodeByName(const string &nodeName, std::shared_ptr<KardanikXML::Construction> construction) const
{
    BOOST_FOREACH (const ConstructionInfo &info, m_ConstructionInfos)
    {
        if (info.parser->getConstruction() != construction)
        {
            continue;
        }
        AnchorNodesMap::const_iterator foundAnchorNode = info.anchorNodesMap.find(nodeName);
        if (foundAnchorNode == info.anchorNodesMap.end())
        {
            std::cerr << "No anchor node with name " << nodeName << "found !" << std::endl;
            return 0;
        }
        return foundAnchorNode->second;
    }
    return 0;
}

KardanikXML::MotionState *KinematicsBehavior::CreateMotionStateForBody(std::shared_ptr<KardanikXML::Body> body, const btTransform &centerOfMass)
{
    std::shared_ptr<KardanikXML::Anchor> anchor = body->GetAnchor();
    if (!anchor)
    {
        btTransform centerOfMassOffset;
        centerOfMassOffset.setIdentity();
        return new KardanikXML::MotionState(NULL, this, centerOfMass, centerOfMassOffset, body->GetName());
    }

    std::shared_ptr<KardanikXML::Point> anchorPoint = anchor->GetAnchorPoint();

    btTransform anchorPointTransform;
    anchorPointTransform.setIdentity();
    anchorPointTransform.setOrigin(btVector3(anchorPoint->GetX(), anchorPoint->GetY(), anchorPoint->GetZ()));
    btTransform centerOfMassOffset = centerOfMass.inverse() * anchorPointTransform;

    std::shared_ptr<KardanikXML::Construction> construction = body->GetParentConstruction().lock();
    osg::MatrixTransform *anchorNode = GetAnchorNodeByName(anchor->GetAnchorNodeName(), construction);

    if (!body->GetOperatingRanges().empty())
    {
        osg::ref_ptr<osg::Switch> circleSwitch(new osg::Switch);
        circleSwitch->setAllChildrenOff();
        anchorNode->addChild(circleSwitch);

        BOOST_FOREACH (std::shared_ptr<KardanikXML::OperatingRange> range, body->GetOperatingRanges())
        {
            std::shared_ptr<KardanikXML::Point> centerPoint = range->GetCenterPoint();
            osg::ref_ptr<osg::MatrixTransform> centerPointTransform(new osg::MatrixTransform);
            osg::ref_ptr<osg::Geode> circleGeom(new osg::Geode);
            centerPointTransform->addChild(circleGeom);
            circleGeom->addDrawable(CreateCircle(range->GetRadius(), osg::Vec3f(0.0, 1.0, 0.0), range->GetColor()).get());
            circleSwitch->addChild(centerPointTransform);

            btTransform centerPointMat;
            centerPointMat.setIdentity();
            centerPointMat.setOrigin(btVector3(centerPoint->GetX(), centerPoint->GetY(), centerPoint->GetZ()));

            btTransform centerPointRelMat = anchorPointTransform.inverse() * centerPointMat;
            centerPointTransform->setMatrix(BulletToOSG(centerPointRelMat));
        }
    }

    return new KardanikXML::MotionState(anchorNode, this, centerOfMass, centerOfMassOffset, body->GetName());
}

string KinematicsBehavior::makeStateForBodyTransforms(const KardanikConstructor::BodyTransforms &transforms) const
{
    ostringstream outString;
    BOOST_FOREACH (const KardanikConstructor::BodyTransforms::value_type &body, transforms)
    {
        outString << body.first << "{" << btTransformToString(body.second) << "}";
    }
    return outString.str();
}

void KinematicsBehavior::initPhysicsSimulation()
{
    BOOST_FOREACH (ConstructionInfo &constructionInfo, m_ConstructionInfos)
    {
        std::shared_ptr<KardanikXML::ConstructionParser> parser = constructionInfo.parser;
        string geomNamespace = parser->getConstruction()->GetNamespace();
        CreateAnchorNodesMap(getSceneObject()->getGeometryNode(geomNamespace), constructionInfo);
        m_KardanikConstructor->InstanciateConstruction(parser->getConstruction());
    }

    // initialize world
    m_DynamicsWorld->stepSimulation(1.0 / 50.0, 10, 1.0 / 240.0);

    m_Clock.reset(new btClock());
    m_ClockInitialized = false;

    setOperatingRangesVisible(Settings::instance()->isOperatingRangeVisible());
}

void KinematicsBehavior::sendStateToGui()
{
    if (opencover::coVRMSController::instance()->isMaster())
    {
        KardanikConstructor::BodyTransforms transforms(m_KardanikConstructor->GetBodyTransforms());
        string transformStateMsg = makeStateForBodyTransforms(transforms);

        grmsg::coGRObjKinematicsStateMsg stateMsg(_sceneObject->getCoviseKey().c_str(), transformStateMsg.c_str());
        Message grmsg{ COVISE_MESSAGE_UI, DataHandle{(char*)stateMsg.c_str(),strlen(stateMsg.c_str()) + 1 , false} };
        opencover::cover->sendVrbMessage(&grmsg);
    }
}
