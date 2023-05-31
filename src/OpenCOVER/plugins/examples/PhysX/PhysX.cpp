/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                            (C)2008 HLRS  **
 **                                                                          **
 ** Description: PhysX OpenCOVER Plugin (draws a cube)                          **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner		                                                  **
 **                                                                          **
 ** History:  								                                         **
 ** June 2008  v1	    				       		                                **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include "PhysX.h"
#include <cover/coVRPluginSupport.h>
#include <osg/ShapeDrawable>
#include <osg/Material>
#include <osg/Vec4>
#include <physics/PhysicsUtil.h>
#include <physics/Callbacks.h>
using namespace opencover;
PhysX::PhysX()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    fprintf(stderr, "PhysX World\n");

    // The scene and scene updater
    osgPhysics::Engine::instance()->addScene(
        "def", osgPhysics::createScene(osg::Vec3(0.0f, 0.0f, -9.8f)));
    osg::ref_ptr<osgPhysics::UpdatePhysicsSystemCallback> physicsUpdater =
        new osgPhysics::UpdatePhysicsSystemCallback("def");
    physicsUpdater->setMaxSimuationDeltaTime(0.005);  // to fix jitter in slower system

    // Build the scene graph
    osg::ref_ptr<osg::Group> root = new osg::Group;
    _root = root;
    root->setName("PhysXRoot");

    root->setNodeMask(root->getNodeMask() | Isect::Update);
    root->addUpdateCallback(physicsUpdater.get());

    // Create the ground
    physx::PxRigidActor* planeActor = osgPhysics::createPlaneActor(osg::Plane(osg::Z_AXIS, 0.0));
    osgPhysics::Engine::instance()->addActor("def", planeActor);

    // Create many cubes
    for (int y = 0; y < 10; ++y)
        for (int x = 0; x < 10; ++x)
        {
            physx::PxRigidActor* actor = osgPhysics::createBoxActor(osg::Vec3(1.0f, 1.0f, 1.0f), 1.0);
            actor->setGlobalPose(physx::PxTransform(osgPhysics::toPxMatrix(
                osg::Matrix::translate(1.05f * (float)x, 0.0f, 1.5f * (float)y))));
            osgPhysics::Engine::instance()->addActor("def", actor);

            osg::ref_ptr<osg::MatrixTransform> mt = dynamic_cast<osg::MatrixTransform*>(
                osgPhysics::createNodeForActor(actor));
            mt->addUpdateCallback(new osgPhysics::UpdateActorCallback(actor));
            root->addChild(mt.get());
        }

    cover->getObjectsRoot()->addChild(root);
}

bool PhysX::destroy()
{
    cover->getObjectsRoot()->removeChild(basicShapesGeode.get());
    return true;
}

// this is called if the plugin is removed at runtime
PhysX::~PhysX()
{
    fprintf(stderr, "Goodbye\n");
}

void PhysX::key(int type, int keySym, int mod)
{
    if (type == osgGA::GUIEventAdapter::KEYDOWN)
    {
        osg::Matrix VM = cover->getViewerMat();
        osg::Matrix invVM;
        invVM.invert(VM);
        osg::Vec3 start = VM.getTrans() * cover->getInvBaseMat(), target = osg::Y_AXIS * invVM * cover->getInvBaseMat();
        target = (target - start); target.normalize(); target *= 50.0f;
            if (keySym == '1' && _root.valid())
            {
                physx::PxRigidActor* actor = osgPhysics::createBoxActor(osg::Vec3(1.0f, 1.0f, 1.0f), 5.0);
                actor->setGlobalPose(physx::PxTransform(osgPhysics::toPxMatrix(
                    osg::Matrix::translate(start))));

                physx::PxRigidDynamic* dynActor = actor->is<physx::PxRigidDynamic>();
                dynActor->setLinearVelocity(physx::PxVec3(target[0], target[1], target[2]));
                osgPhysics::Engine::instance()->addActor("def", actor);

                osg::ref_ptr<osg::MatrixTransform> mt = dynamic_cast<osg::MatrixTransform*>(
                    osgPhysics::createNodeForActor(actor));
                mt->addUpdateCallback(new osgPhysics::UpdateActorCallback(actor));
                _root->addChild(mt.get());
            }
    }
}
COVERPLUGIN(PhysX)
