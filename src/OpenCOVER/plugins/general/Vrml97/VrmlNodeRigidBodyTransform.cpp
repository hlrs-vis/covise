/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeRigidBodyTransform.cpp

#include "VrmlNodeRigidBodyTransform.h"
#include "VrmlMotionState.h"
#include "VrmlNodePhysicsWorld.h"
#include "VrmlNodeRigidBodyRoot.h"
#include <vrml97/vrml/MathUtils.h>
#include <vrml97/vrml/VrmlNodeType.h>
#include <osg/io_utils>
#include <osg/ShapeDrawable>
#include <osg/Geode>
#include <osgwTools/InsertRemove.h>
#include <osgwTools/FindNamedNode.h>
#include <osgwTools/Version.h>
#include "LinearMath/btAlignedObjectArray.h"

#include "VrmlNodeMotionState.h"




using namespace osgbDynamics;
using namespace vrml;
VrmlNode *creator(VrmlScene *s) { return new VrmlNodeRigidBodyTransform(s); }

// Define the built in VrmlNodeType:: "Transform" fields

VrmlNodeType *VrmlNodeRigidBodyTransform::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st;
        t = st = new VrmlNodeType("RigidBodyTransform", creator);
    }

    VrmlNodeTransform::defineType(t); // Parent class
    t->addExposedField("angularVelocity", VrmlField::SFVEC3F);
    t->addExposedField("centerOfMass", VrmlField::SFVEC3F);
    t->addExposedField("forcesApplied", VrmlField::SFVEC3F);
    t->addExposedField("friction", VrmlField::SFFLOAT);
    t->addExposedField("geometry", VrmlField::MFNODE);
    t->addExposedField("linearVelocity", VrmlField::SFVEC3F);
    t->addExposedField("mass", VrmlField::SFFLOAT);
    t->addExposedField("margin", VrmlField::SFFLOAT);
    t->addExposedField("restitution", VrmlField::SFFLOAT);
    t->addExposedField("orientation",VrmlField::SFROTATION);
    t->addExposedField("impulse", VrmlField::SFVEC3F);
    t->addExposedField("relPos", VrmlField::SFVEC3F);
    t->addExposedField("shapeType", VrmlField::SFINT32);

    return t;
}

VrmlNodeType *VrmlNodeRigidBodyTransform::nodeType() const { return defineType(0); }

VrmlNodeRigidBodyTransform::VrmlNodeRigidBodyTransform(VrmlScene* scene)
    : VrmlNodeTransform(scene)
    , d_angularVelocity(0, 0, 0)
    , d_centerOfMass(0, 0, 0)
    , d_forcesApplied(0, 0, 0)
    , d_geometry(NULL)
    , d_linearVelocity(0, 0, 0)
    , d_orientation(0, 0, 0)
    , d_impulse(0, 0, 0)
    , d_relPos(0, 0, 0)
    
    
{
    d_mass = 0.f; 
    d_friction = 0.5;
    d_restitution = 0.3;
    d_shapeType = 0;
    d_margin = 0.0;
   
   
    d_modified = true;
}
 
VrmlNodeRigidBodyTransform::~VrmlNodeRigidBodyTransform()
{
}


VrmlNode *VrmlNodeRigidBodyTransform::cloneMe() const
{
    return new VrmlNodeRigidBodyTransform(*this);
}

std::ostream &VrmlNodeRigidBodyTransform::printFields(std::ostream &os, int indent)
{
    PRINT_FIELD(angularVelocity);
    PRINT_FIELD(centerOfMass);
    PRINT_FIELD(linearVelocity);
    PRINT_FIELD(forcesApplied);
    PRINT_FIELD(friction);
    PRINT_FIELD(geometry);
    PRINT_FIELD(mass);
    PRINT_FIELD(orientation);
    PRINT_FIELD(restitution);
    PRINT_FIELD(impulse);
    PRINT_FIELD(relPos);    
    PRINT_FIELD(shapeType);
    PRINT_FIELD(margin);




    VrmlNodeTransform::printFields(os, indent);
    return os;
}

void VrmlNodeRigidBodyTransform::render(Viewer *viewer)
{

    VrmlNodeTransform::render(viewer);

    if (vo == nullptr) {
        vo = (osgViewerObject*)d_xformObject;
        
        osg::Matrix parentTrans;

        const bool isDynamic = (d_mass.get() != 0.f);
        parentTrans= vo->parentTransform;
        //osg::Matrix ptinv = parentTrans.inverse(parentTrans);
        
        knoten = vo->getNode();
        
        osg::MatrixTransform* mt = static_cast<osg::MatrixTransform*>(knoten);
        osg::Node* Group = mt->getChild(0);
        osg::Vec3 centerOfMass;
        if (!d_centerOfMass.get()) {
            centerOfMass=osg::Vec3(d_centerOfMass.get()[0], d_centerOfMass.get()[1], d_centerOfMass.get()[2]);
        }
        else {

            centerOfMass = mt->getBound().center();
            osg::Matrix invm = mt->getInverseMatrix();
            centerOfMass = centerOfMass * invm;
        }
        osg::Matrix centerOfMassMatrix = osg::Matrix::translate(-centerOfMass);
        osg::MatrixTransform* trans = new osg::MatrixTransform;
        trans->setMatrix(centerOfMassMatrix);
        trans->addChild(Group);
       
        btCollisionShape* shape;
        btCollisionShape* cShape;
        std::string BoundingBox;
        int a = d_shapeType.get();
        
        switch (d_shapeType.get())
        {
        case 1:
            shape = osgbCollision::btBoxCollisionShapeFromOSG(knoten);
            break;
        case 2:
            shape = osgbCollision::btSphereCollisionShapeFromOSG(knoten);
            break;
        case 3:
            shape = osgbCollision::btCylinderCollisionShapeFromOSG(knoten);
            break;
        case 4:
            cShape = osgbCollision::btTriMeshCollisionShapeFromOSG(trans);
            if (d_margin.get())
                cShape->setMargin(d_margin.get());
            shape = cShape;
            break;
            break;
        case 5:
        {
            cShape = osgbCollision::btConvexTriMeshCollisionShapeFromOSG(trans);
            if (d_margin.get())
                cShape->setMargin(d_margin.get());
            shape = cShape;
            break;
        }
        case 6:
        {
            cShape = osgbCollision::btConvexHullCollisionShapeFromOSG(trans);
            if (d_margin.get())
                cShape->setMargin(d_margin.get());
            shape = cShape;
            break;
        }
        default:
            if (isDynamic) {

                 cShape = osgbCollision::btConvexTriMeshCollisionShapeFromOSG(trans);
                
                if (d_margin.get())
                    cShape->setMargin(d_margin.get());
                shape = cShape;
                break;
            }
            else {
                shape = osgbCollision::btBoxCollisionShapeFromOSG(knoten);
          
            }

        }
        createDynamicBody(mt, shape, parentTrans,centerOfMass);
            
    }
    clearModified();
}

// Set the value of one of the node fields.

void VrmlNodeRigidBodyTransform::setField(const char *fieldName,
                                 const VrmlField &fieldValue)
{
    if
        TRY_FIELD(angularVelocity, SFVec3f)
    else if
        TRY_FIELD(centerOfMass, SFVec3f)
    else if
        TRY_FIELD(forcesApplied, SFVec3f)
    else if
        TRY_FIELD(friction, SFFloat)
    else if
        TRY_FIELD(geometry, SFNode)
    else if
        TRY_FIELD(linearVelocity, SFVec3f)
    else if
        TRY_FIELD(mass, SFFloat)
    else if
        TRY_FIELD(orientation, SFRotation)
    else if
        TRY_FIELD(restitution, SFFloat)
    else if
        TRY_FIELD(impulse, SFVec3f)
    else if
        TRY_FIELD(relPos, SFVec3f)
    else if
        TRY_FIELD(shapeType, SFInt)
    else if 
        TRY_FIELD(margin, SFFloat)
    else
   
        VrmlNodeTransform::setField(fieldName, fieldValue);
    d_modified = true;
}

const VrmlField *VrmlNodeRigidBodyTransform::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "angularVelocity") == 0)
        return &d_angularVelocity;
    else if (strcmp(fieldName, "centerOfMass") == 0)
        return &d_centerOfMass;
    else if (strcmp(fieldName, "forcesApplied") == 0)
        return &d_forcesApplied;
    else if (strcmp(fieldName, "friction") == 0)
        return &d_friction;
    else if (strcmp(fieldName, "linearVelocity") == 0)
        return &d_linearVelocity;
    else if (strcmp(fieldName, "mass") == 0)
        return &d_mass;
    else if (strcmp(fieldName, "orientation") == 0)
        return &d_orientation;
    else if (strcmp(fieldName, "restitution") == 0)
        return &d_restitution;
    else if (strcmp(fieldName, "impulse") == 0)
        return &d_impulse;
    else if (strcmp(fieldName, "relPos") == 0)
        return &d_impulse;
    else if (strcmp(fieldName, "shapeType") == 0)
        return &d_shapeType;
    else if (strcmp(fieldName, "margin") == 0)
        return &d_margin;

    return VrmlNodeTransform::getField(fieldName);
}


void VrmlNodeRigidBodyTransform::createDynamicBody(osg::MatrixTransform* node, btCollisionShape* shape, osg::Matrix parentTransform, osg::Vec3 centerOfMass) {


    
    btVector3 localInertia(0, 0, 0);

    if (node == NULL)
    {
        osg::notify(osg::WARN) << "createRigidBody: CreationRecord has NULL scene graph." << std::endl;

    }
    const bool isDynamic = (d_mass.get() != 0.f);
    if (isDynamic) {
        shape->calculateLocalInertia(d_mass.get(), localInertia);
  
    }

  


    osg::notify(osg::DEBUG_FP) << "createRigidBody: Creating rigid body." << std::endl;

   

    // Create MotionState to control OSG subgraph visual reprentation transform
    // from a Bullet world transform. To do this, the MotionState need the address
    // of the Transform node (must be either AbsoluteModelTransform or
    // MatrixTransform), center of mass, scale vector, and the parent (or initial)
    // transform (usually the non-scaled OSG local-to-world matrix obtained from
    // the parent node path).
   

    
    /*
    amt->setDataVariance(osg::Object::DYNAMIC);
    osgwTools::insertAbove(knoten, amt);
    */
    osg::Matrix m = node->getMatrix();

    //btMotionState(node, osgbCollision::asBtTransform(m),parentTransform);
    //osgbDynamics::MotionState* motion = new osgbDynamics::MotionState(); 
    //VrmlNodeMotionState::MotionState* motionStat = static_cast<VrmlNodeMotionState::MotionState*>(motion);
    VrmlMotionState* motion = new VrmlMotionState;
    osg::Matrix centerOfMassMatrix;
    if (!d_centerOfMass.get()) {
        osg::Vec3 com(d_centerOfMass.get()[0], d_centerOfMass.get()[1], d_centerOfMass.get()[2]);
        motion->setCenterOfMass(com);
    }
    else {
        if (d_mass.get() != 0) {
            
            motion->setCenterOfMass(centerOfMass);
            centerOfMassMatrix = osg::Matrix::translate(centerOfMass);
            
        }
    }
 
   
    osg::MatrixList ml = node->getWorldMatrices();
    osg::Matrix n = ml.at(0);
    //osg::Node* lol = amtRoot->getChild(0);
    motion->setLocalTransform(m);
    motion->setTransform(node);
    motion->setParentTransform(parentTransform);
    osg::Matrix absoluteTransform = centerOfMassMatrix * m * parentTransform;
    motion->setWorldTransform(osgbCollision::asBtTransform(absoluteTransform));





    
    //motion->setWorldTransform(osgbCollision::asBtTransform(parentTransform));
    


    // Finally, create rigid body.
    btRigidBody::btRigidBodyConstructionInfo rbInfo(d_mass.get(), motion, shape, localInertia);
    rbInfo.m_friction = btScalar(d_friction.get());
    rbInfo.m_restitution = btScalar(d_restitution.get());
    btRigidBody* rb = new btRigidBody(rbInfo);
  
    if (rb == NULL)
    {
        osg::notify(osg::WARN) << "createRigidBody: Created a NULL btRigidBody." << std::endl;
        
    }
    
    // Last thing to do: Position the rigid body in the world coordinate system. The
    // MotionState has the initial (parent) transform, and also knows how to account
    // for center of mass and scaling. Get the world transform from the MotionState,
    // then set it on the rigid body, which in turn sets the world transform on the
    // MotionState, which in turn transforms the OSG subgraph visual representation.

    if (d_impulse.get()) {
        btVector3 imp(d_impulse.get()[0], d_impulse.get()[1], d_impulse.get()[2]);
        btVector3 pos(d_relPos.get()[0], d_relPos.get()[1], d_relPos.get()[2]);
        if (pos==btVector3(0,0,0)) {
            rb->applyCentralForce(imp);
        }
        else {
            rb->applyImpulse(imp, pos);

        }

        

      
    }
    rb->setAngularVelocity( btVector3(d_angularVelocity.get()[0], d_angularVelocity.get()[1], d_angularVelocity.get()[2]));
    rb->setLinearVelocity( btVector3(d_linearVelocity.get()[0], d_linearVelocity.get()[1], d_linearVelocity.get()[2]));
   
    btTransform wt;
    motion->getWorldTransform(wt);
    rb->setWorldTransform(wt);
    VrmlNodePhysicsWorld::instance()->getWorld()->addRigidBody(rb);
    VrmlNodeRigidBodyRoot::instance()->addToRoot(node);

}



    