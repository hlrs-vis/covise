/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodePhysicsWorld.h

#ifndef _VrmlNodeRigidBodyTransform_
#define _VrmlNodeRigidBodyTransform_

#include <util/coTypes.h>
#include <vrml97/vrml/VrmlNodeTransform.h>
#include <vrml97/vrml/VrmlSFVec3f.h>
#include <vrml97/vrml/VrmlSFFloat.h>
#include <vrml97/vrml/VrmlSFInt.h>
#include <btBulletDynamicsCommon.h>

#include <osgbCollision/GLDebugDrawer.h>
#include <osgbCollision/Version.h>
#include <osgbDynamics/MotionState.h>
#include <osgbDynamics/PhysicsState.h>
#include <osgbCollision/CollisionShapes.h>
#include <osgbCollision/RefBulletObject.h>
#include <osgbDynamics/RigidBody.h>
#include <osgbCollision/Utils.h>
#include "ViewerOsg.h"




namespace vrml
{

class VRML97PLUGINEXPORT VrmlNodeRigidBodyTransform : public VrmlNodeTransform
{

public:
    // Define the fields of Transform nodes
    static VrmlNodeType *defineType(VrmlNodeType *t = 0);
    virtual VrmlNodeType *nodeType() const;

    VrmlNodeRigidBodyTransform(VrmlScene *);
    virtual ~VrmlNodeRigidBodyTransform();

    virtual VrmlNode *cloneMe() const;

    virtual std::ostream &printFields(std::ostream &os, int indent);

    virtual void render(Viewer *);

    virtual void setField(const char *fieldName, const VrmlField &fieldValue);
    const VrmlField *getField(const char *fieldName) const;
    void VrmlNodeRigidBodyTransform::createDynamicBody(osg::MatrixTransform* node, btCollisionShape* shape, osg::Matrix parentTransform, osg::Vec3 centerOfMass);
    osg::Node* knoten;
    osgViewerObject* vo=nullptr;
   
   


protected:
    VrmlSFVec3f d_angularVelocity;
    VrmlSFVec3f d_centerOfMass;
    VrmlSFVec3f d_forcesApplied;
    VrmlSFFloat d_friction;
    VrmlSFFloat d_mass;
    VrmlSFFloat d_margin;
    VrmlSFNode  d_geometry;
    VrmlSFVec3f d_linearVelocity;
    VrmlSFRotation d_orientation;
    VrmlSFFloat d_restitution;
    VrmlSFVec3f d_impulse;
    VrmlSFVec3f d_relPos;
    VrmlSFInt d_shapeType;




};
}
#endif //_VrmlNodeRigidBodyTransform_




