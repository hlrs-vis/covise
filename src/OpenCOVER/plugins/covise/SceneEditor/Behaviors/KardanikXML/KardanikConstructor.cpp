/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "KardanikConstructor.h"

#include <math.h>
#include <bullet/btBulletDynamicsCommon.h>
#include <assert.h>
#include <iostream>
#ifndef WIN32
#include <boost/tr1/functional.hpp>
#endif
#include <functional>
#include <algorithm>
#include <iostream>
#include <boost/foreach.hpp>

#include <cover/coVRPluginSupport.h>

#include "Construction.h"
#include "Body.h"
#include "LineStrip.h"
#include "Point.h"
#include "Line.h"
#include "Joint.h"
#include "BodyJointDesc.h"
#include "MotionStateFactory.h"
#include "MotionState.h"

using namespace KardanikXML;
using namespace std;
using namespace std::tr1;

namespace
{
const short kardanik_group = 64;
}

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

ostream &operator<<(ostream &os, const btTransform &transform)
{
    os << transform.getBasis() << endl;
    os << transform.getOrigin() << endl;
    return os;
}

ostream &operator<<(ostream &os, const KardanikXML::Point &point)
{
    os << "[KardakinXML::Point: " << point.GetX() << ", " << point.GetY() << ", " << point.GetZ() << "]";
    return os;
}
}

KardanikConstructor::KardanikConstructor()
    : m_DynamicsWorld(0)
    , m_MotionStateFactory(0)
{
}

void KardanikConstructor::SetDynamicWorld(btDynamicsWorld *dynamicsWorld)
{
    m_DynamicsWorld = dynamicsWorld;
}

void KardanikConstructor::InstanciateConstruction(std::tr1::shared_ptr<KardanikXML::Construction> construction)
{
    if (!construction)
    {
        return;
    }

    for_each(construction->GetBodies().begin(), construction->GetBodies().end(),
             bind(&KardanikConstructor::AddBody, this, std::tr1::placeholders::_1));
    for_each(construction->GetJoints().begin(), construction->GetJoints().end(),
             bind(&KardanikConstructor::AddJoint, this, std::tr1::placeholders::_1));
}

void KardanikConstructor::AddBody(std::tr1::shared_ptr<Body> body)
{
    if (!body)
    {
        return;
    }

    bool emptyObject = false;

    if (body->GetMotionType() == Body::MOTION_DYNAMIC_NOCOLLISION)
    {
        emptyObject = true;
    }

    btCompoundShape *bodyShape = new btCompoundShape();

    // import LineStrips
    BOOST_FOREACH (std::tr1::shared_ptr<LineStrip> lineStrip, body->GetLineStrips())
    {
        if (!lineStrip)
        {
            continue;
        }
        std::tr1::shared_ptr<Point> lastPoint;
        BOOST_FOREACH (std::tr1::shared_ptr<Point> point, lineStrip->GetPoints())
        {
            if (!lastPoint)
            {
                lastPoint = point;
                continue;
            }
            AddLineToCompound(bodyShape, lastPoint, point, lineStrip->GetRadius(), emptyObject);
            lastPoint = point;
        }
    }

    // import Lines
    BOOST_FOREACH (std::tr1::shared_ptr<Line> line, body->GetLines())
    {
        if (!line)
        {
            continue;
        }
        AddLineToCompound(bodyShape, line->GetPointA(), line->GetPointB(), line->GetRadius(), emptyObject);
    }

    btTransform ident;
    ident.setIdentity();

    btTransform centerOfMass = CalculateCenterOfMass(bodyShape);
    ;

    btRigidBody *rigidBody = 0;
    MotionState *motionState = m_MotionStateFactory->CreateMotionStateForBody(body, centerOfMass);

    if (body->GetMotionType() == Body::MOTION_DYNAMIC || body->GetMotionType() == Body::MOTION_DYNAMIC_NOCOLLISION)
    {
        rigidBody = CreateRigidBody(1.0f, bodyShape, motionState);
        rigidBody->setActivationState(DISABLE_DEACTIVATION);
        rigidBody->setDamping(.1, 1);
    }
    else
    {
        rigidBody = CreateRigidBody(0.0f, bodyShape, motionState);
    }

    TransformChildrenRelativeToCenterOfMass(bodyShape, centerOfMass);

    RigidBodyInfo rbInfo;
    rbInfo.m_RigidBody = rigidBody;
    rbInfo.m_CenterOfMass = centerOfMass;
    rbInfo.m_MotionState = motionState;

    m_BodyToRigidBody.insert(BodyToRigidBody::value_type(body, rbInfo));
}

btTransform KardanikConstructor::CalculateCenterOfMass(btCompoundShape *bodyShape) const
{
    vector<btScalar> masses(bodyShape->getNumChildShapes(), 1.0f);
    btVector3 inertia;
    btTransform principal;

    bodyShape->calculatePrincipalAxisTransform(&masses[0], principal, inertia);
    btTransform centerOfMass = principal;
    // centerOfMass.setRotation(btQuaternion(btVector3(1.f, 0.0f, 0.0f), 0.0f));
    return centerOfMass;
}

void KardanikConstructor::TransformChildrenRelativeToCenterOfMass(btCompoundShape *bodyShape,
                                                                  const btTransform &centerOfMass) const
{
    btTransform invCenterOfMass = centerOfMass.inverse();
    for (int i = 0; i < bodyShape->getNumChildShapes(); ++i)
    {
        btTransform newChildTransform = invCenterOfMass * bodyShape->getChildTransform(i);

        btVector3 pointInA = newChildTransform.getOrigin();

        bodyShape->updateChildTransform(i, newChildTransform, true);
    }
}

void KardanikConstructor::AddJoint(std::tr1::shared_ptr<Joint> joint)
{
    if (!joint)
    {
        return;
    }

    btVector3 hingeAxis;
    switch (joint->GetAxis())
    {
    case Joint::X_AXIS:
        hingeAxis = btVector3(1.0f, 0.0f, 0.0f);
        break;
    case Joint::Y_AXIS:
        hingeAxis = btVector3(0.0f, 1.0f, 0.0f);
        break;
    case Joint::Z_AXIS:
        hingeAxis = btVector3(0.0f, 0.0f, 1.0f);
        break;
    default:
        return;
    }

    std::tr1::shared_ptr<BodyJointDesc> bodyADesc = joint->GetBodyA();
    std::tr1::shared_ptr<BodyJointDesc> bodyBDesc = joint->GetBodyB();

    if (!bodyADesc || !bodyBDesc)
    {
        assert(false);
        return;
    }

    btRigidBody *rigidBodyA = GetRigidBody(bodyADesc->GetBody());
    btRigidBody *rigidBodyB = GetRigidBody(bodyBDesc->GetBody());

    if (!rigidBodyA || !rigidBodyB)
    {
        assert(false);
        return;
    }

    btVector3 pointInAWorld = PointToVector(bodyADesc->GetPoint());
    btVector3 pointInBWorld = PointToVector(bodyBDesc->GetPoint());

    btTransform centerOfMassA = GetCenterOfMass(bodyADesc->GetBody());
    btTransform centerOfMassB = GetCenterOfMass(bodyBDesc->GetBody());

    btTransform centerOfMassAInv = centerOfMassA.inverse();
    btTransform centerOfMassBInv = centerOfMassB.inverse();

    btVector3 pointInA = centerOfMassAInv * pointInAWorld;
    btVector3 pointInB = centerOfMassBInv * pointInBWorld;

    btVector3 hingeAxisInA = centerOfMassAInv.getBasis() * hingeAxis;
    btVector3 hingeAxisInB = centerOfMassBInv.getBasis() * hingeAxis;

    btHingeConstraint *hinge = CreateHinge(*rigidBodyA, *rigidBodyB, pointInA, pointInB, hingeAxisInA, hingeAxisInB);
    if (joint->GetUpperLimit() && joint->GetLowerLimit())
    {
        hinge->setLimit(joint->GetLowerLimit().get(), joint->GetUpperLimit().get());
    }
    m_Joints.push_back(joint);
    m_JointToHinge[joint] = hinge;

    m_DynamicsWorld->addConstraint(hinge, false);
}

namespace
{

bool AlmostEqual(float a, float b, float epsilon = 0.0001)
{
    return fabs(b - a) < epsilon;
}
}

void KardanikConstructor::AddLineToCompound(btCompoundShape *shape, std::tr1::shared_ptr<KardanikXML::Point> start,
                                            std::tr1::shared_ptr<KardanikXML::Point> end, float radius, bool objectIsEmpty)
{
    btVector3 lineStart(start->GetX(), start->GetY(), start->GetZ());
    btVector3 lineEnd(end->GetX(), end->GetY(), end->GetZ());

    btVector3 direction = lineEnd - lineStart;
    float lineLength = direction.length();
    direction.normalize();

    btVector3 yAxis(0.0, 1.0, 0.0);

    double rotAngle = ::acos(btVector3(0.0f, 1.0f, 0.0f).dot(direction));

    btVector3 rotAxis;
    if (AlmostEqual(rotAngle, 0.0, 0.01) || AlmostEqual(rotAngle, M_PI, 0.01))
    {
        rotAxis = btVector3(1.0, 0.0, 0.0);
    }
    else
    {
        rotAxis = yAxis.cross(direction);
    }

    btTransform lineShapeTransform;
    lineShapeTransform.setRotation(btQuaternion(rotAxis, rotAngle));
    lineShapeTransform.setOrigin(lineStart + direction * (lineLength / 2.0f));

    btCollisionShape *lineShape = 0;
    if (objectIsEmpty)
    {
        lineShape = new btSphereShape(0.1f);
    }
    else
    {
        lineShape = new btBoxShape(btVector3(radius, lineLength / 2.0f, radius));
    }
    shape->addChildShape(lineShapeTransform, lineShape);
}

btRigidBody *KardanikConstructor::CreateRigidBody(float mass, btCollisionShape *shape, MotionState *motionState)
{
    //rigidbody is dynamic if and only if mass is non zero, otherwise static
    bool isDynamic = (mass != 0.f);

    btVector3 localInertia(0, 0, 0);
    if (isDynamic)
        shape->calculateLocalInertia(mass, localInertia);

    btRigidBody::btRigidBodyConstructionInfo cInfo(mass, motionState, shape, localInertia);

    btRigidBody *body = new btRigidBody(cInfo);
    body->setContactProcessingThreshold(BT_LARGE_FLOAT);
    m_DynamicsWorld->addRigidBody(body, kardanik_group, 1);
    body->setDamping(1, 1);

    return body;
}

btHingeConstraint *KardanikConstructor::CreateHinge(btRigidBody &rbA, btRigidBody &rbB, const btVector3 &pivotInA,
                                                    const btVector3 &pivotInB, const btVector3 &axisInA, const btVector3 &axisInB) const
{
    btHingeConstraint *pHinge = new btHingeConstraint(rbA, rbB, pivotInA, pivotInB, axisInA, axisInB, false);

    for (int i = -1; i < 6; ++i)
    {
        pHinge->setParam(BT_CONSTRAINT_STOP_ERP, 0.8, i);
        pHinge->setParam(BT_CONSTRAINT_CFM, 0.0, i);
        pHinge->setParam(BT_CONSTRAINT_STOP_CFM, 0.0, i);
    }

    return pHinge;
}

btRigidBody *KardanikConstructor::GetRigidBody(std::tr1::shared_ptr<KardanikXML::Body> body) const
{
    BodyToRigidBody::const_iterator foundBody = m_BodyToRigidBody.find(body);
    if (foundBody == m_BodyToRigidBody.end())
    {
        return 0;
    }
    return foundBody->second.m_RigidBody;
}

btTransform KardanikConstructor::GetCenterOfMass(std::tr1::shared_ptr<KardanikXML::Body> body) const
{
    BodyToRigidBody::const_iterator foundBody = m_BodyToRigidBody.find(body);
    if (foundBody == m_BodyToRigidBody.end())
    {
        return btTransform();
    }
    return foundBody->second.m_RigidBody->getCenterOfMassTransform();
}

btVector3 KardanikConstructor::PointToVector(std::tr1::shared_ptr<KardanikXML::Point> point) const
{
    if (!point)
    {
        return btVector3(0.0f, 0.0f, 0.0f);
    }
    return btVector3(point->GetX(), point->GetY(), point->GetZ());
}

void KardanikConstructor::SetMotionStateFactory(KardanikXML::MotionStateFactory *motionStateFactory)
{
    m_MotionStateFactory = motionStateFactory;
}

void KardanikConstructor::MakeHigherBodiesStatic(std::tr1::shared_ptr<KardanikXML::Body> touchedBody, bool freeHigher)
{
    const unsigned int specialBodyID = 99;

    unsigned int bodyID = touchedBody->GetMotionID();
    BOOST_FOREACH (const BodyToRigidBody::value_type &bodyPair, m_BodyToRigidBody)
    {
        std::tr1::shared_ptr<Body> body = bodyPair.first;
        LockConnectedJoints(body);
    }

    BOOST_FOREACH (const BodyToRigidBody::value_type &bodyPair, m_BodyToRigidBody)
    {
        std::tr1::shared_ptr<Body> body = bodyPair.first;
        unsigned int thisBodyID = body->GetMotionID();
        if ((thisBodyID < bodyID) && freeHigher)
        {
            UnlockConnectedJoints(body, true);
        }
        if (thisBodyID == bodyID)
        {
            UnlockConnectedJoints(body, false);
        }
    }

    // if special body like tft or marLED ist touched, keep it locked
    BOOST_FOREACH (const BodyToRigidBody::value_type &bodyPair, m_BodyToRigidBody)
    {
        std::tr1::shared_ptr<Body> body = bodyPair.first;
        unsigned int thisBodyID = body->GetMotionID();
        if ((thisBodyID == bodyID) && (bodyID == specialBodyID) && freeHigher)
        {
            LockConnectedJoints(body);
        }
    }
}

void KardanikConstructor::ResetBodyMotions()
{
    //    BOOST_FOREACH(const BodyToRigidBody::value_type& bodyPair, m_BodyToRigidBody) {
    //        std::tr1::shared_ptr<Body>  body = bodyPair.first;
    //        btRigidBody*  rigidBody = bodyPair.second.m_RigidBody;
    //        if (body->GetMotionType() == KardanikXML::Body::MOTION_STATIC) {
    //            rigidBody->setCollisionFlags(rigidBody->getCollisionFlags() | btCollisionObject::CF_STATIC_OBJECT);
    //        } else {
    //            rigidBody->setCollisionFlags(rigidBody->getCollisionFlags() & ~btCollisionObject::CF_STATIC_OBJECT);
    //        }
    //    }
}

void KardanikConstructor::LockConnectedJoints(std::tr1::shared_ptr<Body> body)
{
    BOOST_FOREACH (std::tr1::weak_ptr<Joint> weakJoint, body->GetConnectedJoints())
    {
        if (weakJoint.expired())
        {
            continue;
        }
        std::tr1::shared_ptr<Joint> joint = weakJoint.lock();
        btHingeConstraint *hinge = m_JointToHinge[joint];
        if (!hinge)
        {
            continue;
        }
        btScalar hingeAngle = hinge->getHingeAngle();
        hinge->setLimit(hingeAngle, hingeAngle, 0.9, 0.5, 0.5);
    }
}

void KardanikConstructor::MoveJointsToInitialPositions()
{
    BOOST_REVERSE_FOREACH(std::tr1::shared_ptr<Joint> joint, m_Joints)
    {
        if (!joint->GetInitialAngle())
        {
            continue;
        }
        float jointInitialAngle = joint->GetInitialAngle().get();
        btHingeConstraint *constraint = m_JointToHinge[joint];
        constraint->setLimit(jointInitialAngle, jointInitialAngle, 0.9, 0.5, 0.5);
    }
    m_DynamicsWorld->stepSimulation(25.0 / 50.0, 10, 1.0 / 240.0);
}

void KardanikConstructor::SetJointLimits()
{
    BOOST_FOREACH (const JointToHinge::value_type &jointHingePair, m_JointToHinge)
    {
        std::tr1::shared_ptr<Joint> joint = jointHingePair.first;
        if (!(joint->GetLowerLimit() && joint->GetUpperLimit()))
        {
            continue;
        }
        float jointLowerLimit = joint->GetLowerLimit().get();
        float jointUpperLimit = joint->GetUpperLimit().get();

        btHingeConstraint *constraint = jointHingePair.second;
        constraint->setLimit(jointLowerLimit, jointUpperLimit, 0.9, 0.5, 0.5);
    }
    m_DynamicsWorld->stepSimulation(25.0 / 50.0, 10, 1.0 / 240.0);
}

void KardanikConstructor::UnlockConnectedJoints(std::tr1::shared_ptr<Body> body, bool unlockLower)
{
    unsigned int bodyMotionID = body->GetMotionID();

    BOOST_FOREACH (std::tr1::weak_ptr<Joint> weakJoint, body->GetConnectedJoints())
    {
        if (weakJoint.expired())
        {
            continue;
        }
        std::tr1::shared_ptr<Joint> joint = weakJoint.lock();
        std::tr1::shared_ptr<Body> bodyA = joint->GetBodyA()->GetBody();
        std::tr1::shared_ptr<Body> bodyB = joint->GetBodyB()->GetBody();

        std::tr1::shared_ptr<Body> otherBody = bodyB;
        if (bodyB == body)
        {
            otherBody = bodyA;
        }

        if (otherBody->GetMotionID() > bodyMotionID && !unlockLower)
        {
            continue;
        }

        btHingeConstraint *hinge = m_JointToHinge[joint];
        if (!hinge)
        {
            continue;
        }

        if (joint->GetUpperLimit() && joint->GetLowerLimit())
        {
            hinge->setLimit(joint->GetLowerLimit().get(), joint->GetUpperLimit().get(), 0.9, 0.5);
        }
        else
        {
            hinge->setLimit(1.0, -1.0);
        }
    }
}

void KardanikConstructor::SetInitialBodyTransforms(const BodyTransforms &bodyTransforms)
{
    BOOST_FOREACH (const BodyToRigidBody::value_type &bodyRigid, m_BodyToRigidBody)
    {
        std::tr1::shared_ptr<Body> body = bodyRigid.first;
        std::tr1::shared_ptr<Construction> construction = body->GetParentConstruction().lock();
        BodyTransforms::const_iterator foundBody = bodyTransforms.find(construction->GetNamespace() + ":" + body->GetName());
        if (foundBody != bodyTransforms.end())
        {
            btTransform newCenterOfMass = foundBody->second;
            bodyRigid.second.m_RigidBody->setCenterOfMassTransform(newCenterOfMass);
        }
    }
}

KardanikConstructor::BodyTransforms KardanikConstructor::GetBodyTransforms() const
{
    BodyTransforms bodyTransforms;

    BOOST_FOREACH (const BodyToRigidBody::value_type &body, m_BodyToRigidBody)
    {
        std::tr1::shared_ptr<Body> bPtr = body.first;
        std::tr1::shared_ptr<Construction> construction = bPtr->GetParentConstruction().lock();

        bodyTransforms.insert(BodyTransforms::value_type(construction->GetNamespace() + ":" + bPtr->GetName(), body.second.m_RigidBody->getCenterOfMassTransform()));
    }
    return bodyTransforms;
}
