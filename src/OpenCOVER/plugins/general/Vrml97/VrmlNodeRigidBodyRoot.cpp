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

#include "VrmlNodeRigidBodyRoot.h"
using namespace vrml;
VrmlNodeRigidBodyRoot* VrmlNodeRigidBodyRoot::s_singleton = nullptr;
VrmlNodeRigidBodyRoot* VrmlNodeRigidBodyRoot::instance() {
	if (s_singleton == nullptr) s_singleton
		= new VrmlNodeRigidBodyRoot();
	return s_singleton;
}
btAlignedObjectArray<osg::Node*> &VrmlNodeRigidBodyRoot::getRoot() { return root; }
VrmlNodeRigidBodyRoot::~VrmlNodeRigidBodyRoot() {}
void VrmlNodeRigidBodyRoot::addToRoot(osg::Node* node) {
	root.push_back(node);
}
int VrmlNodeRigidBodyRoot::getRootSize() { return 0; }

VrmlNodeRigidBodyRoot::VrmlNodeRigidBodyRoot() {};
