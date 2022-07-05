

/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public LicenseVrmlNodeRigidBodyGroup
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

 //
 //  Vrml 97 library
 //  Copyright (C) 1998 Chris Morley
 //
 //  %W% %G%
 //  VrmlNodeRigidBodyGroup.h

#ifndef _VrmlNodeRigidBodyGroup_
#define _VrmlNodeRigidBodyGroup_

#include "VrmlNodeRigidBodyTransform.h"
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

    class VRML97COVEREXPORT VrmlNodeRigidBodyGroup : public VrmlNodeRigidBodyTransform
    {

    public:
        // Define the fields of Transform nodes
        static VrmlNodeType* defineType(VrmlNodeType* t = 0);
        virtual VrmlNodeType* nodeType() const;

        VrmlNodeRigidBodyGroup(VrmlScene*);
        virtual ~VrmlNodeRigidBodyGroup();

        virtual VrmlNode* cloneMe() const;


        virtual void render(Viewer*);

        void VrmlNodeRigidBodyGroup::createDynamicBody(osg::Node* node, btCollisionShape* shape);




    protected:


    };
}
#endif //_VrmlNodeRigidBodyGroup_

