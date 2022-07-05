/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodePhysicsWorldcpp

#include "VrmlNodePhysicsWorld.h"
#include <vrml97/vrml/MathUtils.h>
#include <vrml97/vrml/VrmlNodeType.h>

#include <osgViewer/Viewer>
#include <osgDB/ReadFile>
#include <osgDB/FileNameUtils>
#include <osgDB/FileUtils>
#include <osgwTools/AbsoluteModelTransform.h>
#include <osg/ShapeDrawable>
#include <osg/Geode>

#include <btBulletDynamicsCommon.h>

#include <osgbCollision/GLDebugDrawer.h>
#include <osgbCollision/Version.h>
#include <osgbDynamics/MotionState.h>
#include <osgbDynamics/PhysicsState.h>
#include <osgbCollision/CollisionShapes.h>
#include <osgbCollision/RefBulletObject.h>
#include <osgbDynamics/RigidBody.h>
#include <osgbCollision/Utils.h>

#include <osgbInteraction/SaveRestoreHandler.h>
#include <osgbInteraction/DragHandler.h>
#include <osgGA/TrackballManipulator>

#include <osg/io_utils>

using namespace vrml;

VrmlNodePhysicsWorld* VrmlNodePhysicsWorld::s_singleton= NULL;

// Define the built in VrmlNodeType:: "Transform" fields




VrmlNodePhysicsWorld::~VrmlNodePhysicsWorld()
{
}

VrmlNodePhysicsWorld* VrmlNodePhysicsWorld:: instance()
{
    if (s_singleton == NULL)
    {
        s_singleton = new VrmlNodePhysicsWorld();                                                                                                                                                       
    }
    return s_singleton;
}

// Set the value of one of the node fields.

VrmlNodePhysicsWorld::VrmlNodePhysicsWorld()
{
    btDefaultCollisionConfiguration* collisionConfiguration = new btDefaultCollisionConfiguration();
    btCollisionDispatcher* dispatcher = new btCollisionDispatcher( collisionConfiguration );
    btConstraintSolver* solver = new btSequentialImpulseConstraintSolver;

    btVector3 worldAabbMin( -10000, -10000, -10000 );
    btVector3 worldAabbMax( 10000, 10000, 10000 );
    btBroadphaseInterface* inter = new btAxisSweep3( worldAabbMin, worldAabbMax, 1000 );

    dynamicsWorld = new btDiscreteDynamicsWorld( dispatcher, inter, solver, collisionConfiguration );

    dynamicsWorld->setGravity( btVector3( 0, -9.8, 0 ));

   
}



btDiscreteDynamicsWorld* VrmlNodePhysicsWorld::getWorld()
{
    return dynamicsWorld;
}


