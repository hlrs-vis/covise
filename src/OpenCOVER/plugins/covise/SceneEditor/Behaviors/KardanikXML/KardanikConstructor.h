/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * Line.h
 *
 *  Created on: Jan 2, 2012
 *      Author: jw_te
 */

#pragma once
#ifndef WIN32
#include <boost/tr1/memory.hpp>
#endif
#include <memory>

#include <map>
#include <vector>
#include <bullet/btBulletDynamicsCommon.h>

namespace KardanikXML
{
class Construction;
class Body;
class Joint;
class Point;
class MotionStateFactory;
class MotionState;
}

class btDynamicsWorld;
class btCompoundShape;
class btTransform;
class btCollisionShape;
class btRigidBody;
class btTypedConstraint;
class btHingeConstraint;
class btVector3;
struct btBroadphaseProxy;
class btMotionState;

class KardanikConstructor : public std::tr1::enable_shared_from_this<KardanikConstructor>
{
private:
public:
    KardanikConstructor();

    void SetDynamicWorld(btDynamicsWorld *dynamicsWorld);
    void InstanciateConstruction(std::tr1::shared_ptr<KardanikXML::Construction> construction);

    void SetMotionStateFactory(KardanikXML::MotionStateFactory *motionStateFactory);
    btRigidBody *GetRigidBody(std::tr1::shared_ptr<KardanikXML::Body> body) const;

    void MakeHigherBodiesStatic(std::tr1::shared_ptr<KardanikXML::Body> body, bool freeHigher);
    void ResetBodyMotions();

    void MoveJointsToInitialPositions();
    void SetJointLimits();

    typedef std::map<std::string, btTransform> BodyTransforms;
    void SetInitialBodyTransforms(const BodyTransforms &bodyTransforms);

    BodyTransforms GetBodyTransforms() const;

private:
    void AddBody(std::tr1::shared_ptr<KardanikXML::Body> body);
    void AddJoint(std::tr1::shared_ptr<KardanikXML::Joint> body);
    void AddLineToCompound(btCompoundShape *shape, std::tr1::shared_ptr<KardanikXML::Point> start,
                           std::tr1::shared_ptr<KardanikXML::Point> end, float radius, bool objectIsEmpty = false);
    btRigidBody *CreateRigidBody(float mass, btCollisionShape *shape, KardanikXML::MotionState *motionState);
    btHingeConstraint *CreateHinge(btRigidBody &rbA, btRigidBody &rbB,
                                   const btVector3 &pivotInA, const btVector3 &pivotInB,
                                   const btVector3 &axisInA, const btVector3 &axisInB) const;

    btTransform GetCenterOfMass(std::tr1::shared_ptr<KardanikXML::Body> body) const;
    btVector3 PointToVector(std::tr1::shared_ptr<KardanikXML::Point> point) const;
    btTransform CalculateCenterOfMass(btCompoundShape *bodyShape) const;
    void TransformChildrenRelativeToCenterOfMass(btCompoundShape *bodyShape, const btTransform &centerOfMass) const;

    void LockConnectedJoints(std::tr1::shared_ptr<KardanikXML::Body> body);
    void UnlockConnectedJoints(std::tr1::shared_ptr<KardanikXML::Body> body, bool unlockLower = false);

    btDynamicsWorld *m_DynamicsWorld;

    struct RigidBodyInfo
    {
        btRigidBody *m_RigidBody;
        btTransform m_CenterOfMass;
        KardanikXML::MotionState *m_MotionState;
    };

    BodyTransforms m_InitialBodyTransforms;

    typedef std::map<std::tr1::shared_ptr<KardanikXML::Body>, RigidBodyInfo> BodyToRigidBody;
    BodyToRigidBody m_BodyToRigidBody;
    typedef std::vector<const btBroadphaseProxy *> BroadphaseProxies;
    BroadphaseProxies m_BroadphaseProxies;
    typedef std::map<std::tr1::shared_ptr<KardanikXML::Joint>, btHingeConstraint *> JointToHinge;
    JointToHinge m_JointToHinge;

    typedef std::vector<std::tr1::shared_ptr<KardanikXML::Joint> > Joints;
    Joints m_Joints;

    KardanikXML::MotionStateFactory *m_MotionStateFactory;
};
