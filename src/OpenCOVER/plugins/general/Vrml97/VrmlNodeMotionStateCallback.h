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

#ifndef _VrmlNodeMationState_
#define _VrmlNodeMationState_


#include <util/coTypes.h>
#include <vrml97/vrml/VrmlNodeTransform.h>
#include <vrml97/vrml/VrmlSFVec3f.h>
#include <vrml97/vrml/VrmlSFFloat.h>
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
#include <osg/MatrixTransform>
#include <iostream>


namespace vrml
{
    struct OSGBDYNAMICS_EXPORT MotionStateCallback
    {
        MotionStateCallback() {}
        virtual ~MotionStateCallback() {}

        virtual void operator()(const btTransform& worldTrans) = 0;
    };
    typedef std::vector< MotionStateCallback* > MotionStateCallbackList;


    // forward declaration
    class TripleBuffer;
class VrmlNodeMotionStateCallback : public MotionStateCallback
{

public:
    VrmlNodeMotionStateCallback() {};
    ~VrmlNodeMotionStateCallback() {};
    void MotionStateCallback::operator()(const btTransform &worldTrans);
   


};
}
#endif //_VrmlNodeMationState_





/*
namespace vrml
{

    ///The btDefaultMotionState provides a common implementation to synchronize world transforms with offsets.
    class VrmlNodeMotionState : public btMotionState
    {
    public:
        VrmlNodeMotionState(osg::MatrixTransform* node, btTransform& localTransform, const osg::Matrix parentTransform);

        ///synchronizes world transform from user to physics
        virtual void getWorldTransform(btTransform& worldTransform) const;

        ///synchronizes world transform from physics to user
        ///Bullet only calls the update of worldtransform for active objects
        virtual void setWorldTransform(const btTransform& worldTrans);

    private:
        
        osg::MatrixTransform* m_AnchorNode;
        btTransform m_localTransform;
        osg::Matrix m_parentTransform;
        btTransform m_worldTransform;

    };
}
*/
