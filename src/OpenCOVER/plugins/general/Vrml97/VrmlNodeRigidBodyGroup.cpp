/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

 //
 //  Vrml 97 library
 //  Copyright (C) 1998 Chris Morley
 //
 //  %W% %G%
 //  VrmlNodeRigidBodyRootcpp

#include "VrmlNodeRigidBodyGroup.h"
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
static VrmlNode* creator(VrmlScene* s) { return new VrmlNodeRigidBodyGroup(s); }

// Define the built in VrmlNodeType:: "Transform" fields

VrmlNodeType* VrmlNodeRigidBodyGroup::defineType(VrmlNodeType* t)
{
    static VrmlNodeType* st = 0;

    if (!t)
    {
        if (st)
            return st;
        t = st = new VrmlNodeType("RigidBodyGroup", creator);
    }


    return t;
}

VrmlNodeType* VrmlNodeRigidBodyGroup::nodeType() const { return defineType(0); }

VrmlNodeRigidBodyGroup::VrmlNodeRigidBodyGroup(VrmlScene* scene)
    : VrmlNodeRigidBodyTransform(scene)
{

    d_modified = true;
}

VrmlNodeRigidBodyGroup::~VrmlNodeRigidBodyGroup()
{

}


VrmlNode* VrmlNodeRigidBodyGroup::cloneMe() const
{
    return new VrmlNodeRigidBodyGroup(*this);
}


void VrmlNodeRigidBodyGroup::render(Viewer* viewer)
{

    VrmlNodeTransform::render(viewer);

    if (vo == nullptr) {
        vo = (osgViewerObject*)d_xformObject;

        osg::Matrix parentTrans;

      
        parentTrans = vo->parentTransform;

        knoten = vo->getNode();

        osg::MatrixTransform* mt = static_cast<osg::MatrixTransform*>(knoten);

        osg::Vec3 centerOfMass;
        if (!d_centerOfMass.get()) {
            centerOfMass = osg::Vec3(d_centerOfMass.get()[0], d_centerOfMass.get()[1], d_centerOfMass.get()[2]);
        }
        else {

            centerOfMass = mt->getBound().center();
            osg::Matrix invm = mt->getInverseMatrix();
            centerOfMass = centerOfMass * invm;
        }


        VrmlNodeRigidBodyTransform::createDynamicBody(mt, osgbCollision::btBoxCollisionShapeFromOSG(knoten), parentTrans, centerOfMass);

    }
clearModified();
}

