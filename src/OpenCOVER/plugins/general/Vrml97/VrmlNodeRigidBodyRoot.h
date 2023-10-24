/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

 //
 //  Vrml 97 library
 //  Copyright (C) 1998 Chris Morley
 //
 //  %W% %G%
 //  VrmlNodeRigidBodyRoot.h

#ifndef _VrmlNodeRigidBodyRoot_
#define _VrmlNodeRigidBodyRoot_

#include <util/coTypes.h>
#include <vrml97/vrml/VrmlNodeTransform.h>
#include <vrml97/vrml/VrmlSFVec3f.h>
#include <vrml97/vrml/VrmlSFFloat.h>
#include <btBulletDynamicsCommon.h>
#include <BulletSoftBody/btSoftRigidDynamicsWorld.h>
#include <osg/io_utils>
#include <osg/ShapeDrawable>
#include <osg/Geode>

namespace vrml
{

	class  VRML97COVEREXPORT VrmlNodeRigidBodyRoot
	{

	public:
		static VrmlNodeRigidBodyRoot* instance();
		btAlignedObjectArray<osg::Node*> &getRoot();
		virtual ~VrmlNodeRigidBodyRoot();
		void addToRoot(osg::Node* node);
		int getRootSize();


	private:

		VrmlNodeRigidBodyRoot();
		static VrmlNodeRigidBodyRoot* s_singleton;
		btAlignedObjectArray < osg::Node*> root;
		btSoftBodyWorldInfo worldInfo;

	};
}
#endif //_VrmlNodeRigidBodyRoot_