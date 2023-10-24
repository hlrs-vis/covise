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

#ifndef _VrmlNodePhysicsWorld_
#define _VrmlNodePhysicsWorld_

#include <util/coTypes.h>
#include <vrml97/vrml/VrmlNodeTransform.h>
#include <vrml97/vrml/VrmlSFVec3f.h>
#include <vrml97/vrml/VrmlSFFloat.h>
#include <btBulletDynamicsCommon.h>
#include <BulletSoftBody/btSoftRigidDynamicsWorld.h>

namespace vrml
{
   
class  VRML97COVEREXPORT VrmlNodePhysicsWorld
{

public:
	static VrmlNodePhysicsWorld* instance();
	btDiscreteDynamicsWorld* getWorld();
	virtual ~VrmlNodePhysicsWorld();


private:

	VrmlNodePhysicsWorld();
	static VrmlNodePhysicsWorld* s_singleton;
	btDiscreteDynamicsWorld* dynamicsWorld =nullptr;
	btSoftBodyWorldInfo worldInfo;

};
}
#endif //_VrmlNodePhysicsWorld_
