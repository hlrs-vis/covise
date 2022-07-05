/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeMotionState.cpp

#include "VrmlNodeMotionStateCallback.h"

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




using namespace vrml;


// Define the built in VrmlNodeType:: "Transform" fields

 
VrmlNodeMotionStateCallback::~VrmlNodeMotionStateCallback()
{

}

VrmlNodeMotionStateCallback::VrmlNodeMotionStateCallback()
    {

  
}
void MotionStateCallback::operator() (const btTransform& worldTrans) {
    /*btScalar ogl[16];
    worldTrans.getOpenGLMatrix(ogl);
    osg::Matrix localMat(ogl);


    osg::Matrix worldMat = localMat * _parentTransform;

    if (_mscl.size() > 0)
    {
        // Call only if position changed.
        const btVector3 delta(worldTrans.getOrigin() - _transform.getOrigin());
        const btScalar eps((btScalar)(1e-5));
        const bool quiescent(osg::equivalent(delta[0], btScalar(0.), eps) &&
            osg::equivalent(delta[1], btScalar(0.), eps) &&
            osg::equivalent(delta[2], btScalar(0.), eps));
        if (!quiescent)
        {
            MotionStateCallbackList::iterator it;
            for (it = _mscl.begin(); it != _mscl.end(); ++it)
                (**it)(worldTrans);
        }
    }

    // _transform is the model-to-world transformation used to place collision shapes
    // in the physics simulation. Bullet queries this with getlocalTransform().

    _transform = osgbCollision::asBtTransform(worldMat);


    /*
        char* addr(_tb->writeAddress());
        if (addr == NULL)
        {
            osg::notify(osg::WARN) << "MotionState: No TripleBuffer write address." << std::endl;
            return;
        }
        btScalar* fAddr = reinterpret_cast<btScalar*>(addr + _tbIndex);
        localTrans.getOpenGLMatrix(fAddr);
    */
    
}
