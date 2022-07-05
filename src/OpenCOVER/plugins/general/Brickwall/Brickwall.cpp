/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                            (C)2008 HLRS  **
 **                                                                          **
 ** Description: Brickwall OpenCOVER Plugin (is polite)                          **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner		                                                  **
 **                                                                          **
 ** History:  								                                         **
 ** June 2008  v1	    				       		                                **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include "Brickwall.h"
#include <cover/ui/Menu.h>
#include <cover/ui/Button.h>

#include <OpenVRUI/coCombinedButtonInteraction.h>
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


#include <osg/io_utils>
#include <iostream>
#include <sstream>
#include <osg/Material>

#include <osgViewer/ViewerEventHandlers>

#include <osg/LightModel>
#include <osg/Texture2D>
#include <osgUtil/SmoothingVisitor>

#include <osgbDynamics/GroundPlane.h>

#include <osgbInteraction/LaunchHandler.h>


#include <osgwTools/Shapes.h>


#include <BulletSoftBody/btSoftBodyHelpers.h>
#include <BulletSoftBody/btSoftRigidDynamicsWorld.h>
#include <BulletSoftBody/btSoftBodyRigidBodyCollisionConfiguration.h>
#include <BulletSoftBody/btSoftBody.h>

#include "btBulletDynamicsCommon.h"
#include "CommonInterfaces/CommonRigidBodyBase.h"
#include "OpenGLWindow/ShapeData.h"

#include <string>

#include <cover/coVRPluginSupport.h>

using namespace opencover;
using namespace vrui;
Brickwall::Brickwall()
    : ui::Owner("BrickWallPlugin", cover->ui)
{
    fprintf(stderr, "Brickwall World\n");
}

// this is called if the plugin is removed at runtime
Brickwall::~Brickwall()
{
    if (InteractionA->isRegistered())
    {
        coInteractionManager::the()->unregisterInteraction(InteractionA);
    }
    
    fprintf(stderr, "Goodbye\n");

}
/*
bool Brickwall::destroy()
{
    root->removeChild(bricks);
    root->removeChild(ground);
    root->removeChild(ball);
    cover->getObjectsRoot()->removeChild(root);
    bulletWorld->removeRigidBody(brickBody);
    bulletWorld->removeRigidBody(groundBody);
    bulletWorld->removeRigidBody(ballBody);

    return true;
}
*/

COVERPLUGIN(Brickwall)


btSoftBodyWorldInfo	worldInfo;

struct BasicExample : public CommonRigidBodyBase
{
    BasicExample(struct GUIHelperInterface* helper)
        : CommonRigidBodyBase(helper)
    {
    }
};
btSoftRigidDynamicsWorld* Brickwall::initPhysics()
{
    btSoftBodyRigidBodyCollisionConfiguration* collision = new btSoftBodyRigidBodyCollisionConfiguration();
    btCollisionDispatcher* dispatcher = new btCollisionDispatcher(collision);
    worldInfo.m_dispatcher = dispatcher;
    btConstraintSolver* solver = new btSequentialImpulseConstraintSolver;

    btVector3 worldAabbMin(-10000, -10000, -10000);
    btVector3 worldAabbMax(10000, 10000, 10000);
    btBroadphaseInterface* broadphase = new btAxisSweep3(worldAabbMin, worldAabbMax, 1000);
    worldInfo.m_broadphase = broadphase;

    btSoftRigidDynamicsWorld* dynamicsWorld = new btSoftRigidDynamicsWorld(dispatcher, broadphase, solver, collision);

    btVector3 gravity(0, 0, -32.17);
    dynamicsWorld->setGravity(gravity);
    worldInfo.m_gravity = gravity;

    worldInfo.air_density = btScalar(1.2);
    worldInfo.water_density = 0;
    worldInfo.water_offset = 0;
    worldInfo.water_normal = btVector3(0, 0, 0);
    worldInfo.m_sparsesdf.Initialize();


    return(dynamicsWorld);
}

osg::Transform* Brickwall::createOSGBox(osg::Vec3 lengths, osg::Vec4 rotation, osg::Vec4 color)
{
    osg::Box* box = new osg::Box();
    box->setHalfLengths(lengths);

    osg::ShapeDrawable* shape = new osg::ShapeDrawable(box);
    shape->setColor(color);
    osg::Geode* geode = new osg::Geode();
    geode->addDrawable(shape);

    osg::MatrixTransform* mt = new osg::MatrixTransform();
    mt->addChild(geode);

    return(mt);
}
osg::Transform* Brickwall::makeBrick(const osg::Vec3 lengths, osg::Vec4 rotation, osg::Vec4 color)
{


    osg::Geometry* geom = osgwTools::makeBox(lengths, osg::Vec3s(1., 1., 1.));
    osg::Geode* geode = new osg::Geode();
    geode->addDrawable(geom);
    {
        osg::StateSet* stateSet(geom->getOrCreateStateSet());

        osg::LightModel* lm(new osg::LightModel());
        lm->setTwoSided(true);
        stateSet->setAttributeAndModes(lm);

        const std::string texName("bricks.png");
        osg::Texture2D* tex(new osg::Texture2D(
            osgDB::readImageFile(texName)));
        if ((tex == NULL) || (tex->getImage() == NULL))
            osg::notify(osg::WARN) << "Unable to read texture: \"" << texName << "\"." << std::endl;
        else
        {
            tex->setResizeNonPowerOfTwoHint(false);
            stateSet->setTextureAttributeAndModes(0, tex);
        }



    }
    geode->addDrawable(geom);

    osg::MatrixTransform* mt = new osg::MatrixTransform();
    mt->addChild(geode);

    return(mt);


}

osg::Node* Brickwall::createGround(float w, float h, const osg::Vec3& center, btDynamicsWorld* dw)
{
    osg::Transform* ground = createOSGBox(osg::Vec3(w, h, 0.3), osg::Vec4(0., 0., 0., 45.), osg::Vec4(1., 1., 1., 1.));


    osg::ref_ptr< osgbDynamics::CreationRecord > cr = new osgbDynamics::CreationRecord;
    cr->_sceneGraph = ground;
    cr->_shapeType = BOX_SHAPE_PROXYTYPE;
    cr->_mass = 0.f;
    cr->_restitution = 0.5f;
    cr->_friction = 0.9f;
    groundBody = osgbDynamics::createRigidBody(cr.get(), osgbCollision::btBoxCollisionShapeFromOSG(ground));


    // Transform the box explicitly.
    osgbDynamics::MotionState* motion = dynamic_cast<osgbDynamics::MotionState*>(groundBody->getMotionState());
    osg::Matrix m(osg::Matrix::translate(center));
    motion->setParentTransform(m);
    groundBody->setWorldTransform(osgbCollision::asBtTransform(m));

    dw->addRigidBody(groundBody);

    return(ground);
}

osg::Transform* Brickwall::createBall(float radius)
{
    osg::Sphere* sphere = new osg::Sphere();
    sphere->setRadius(radius);
    osg::ShapeDrawable* shape = new osg::ShapeDrawable(sphere);
    shape->setColor(osg::Vec4(3, 3, 1, 1));
    osg::Geode* geode = new osg::Geode();
    geode->addDrawable(shape);

    osg::MatrixTransform* mt = new osg::MatrixTransform();
    mt->addChild(geode);

    return (mt);
}

osg::Transform* Brickwall::createDynamicBall(float mass, float radius, const osg::Vec3& center, btDynamicsWorld* dw,const osg::Vec3& direction) {
    osg::Transform* sphere = createBall(radius);
    osgbCollision::AXIS axis(osgbCollision::Z);

    osg::ref_ptr< osgbDynamics::CreationRecord> cr = new osgbDynamics::CreationRecord;
    cr->_sceneGraph = sphere;
    cr->_shapeType = SPHERE_SHAPE_PROXYTYPE;
    cr->_mass = mass;
    cr->_axis = axis;
    cr->_restitution = 0.6;
    cr->_friction = 0.5f;
    cr->_reductionLevel = osgbDynamics::CreationRecord::MINIMAL;
    ballBody = osgbDynamics::createRigidBody(cr.get(), osgbCollision::btBoxCollisionShapeFromOSG(sphere));
    ballBody->applyImpulse(osgbCollision::asBtVector3(direction)*5000, osgbCollision::asBtVector3(center));

    // Transform the box explicitly.
    osgbDynamics::MotionState* motion = dynamic_cast<osgbDynamics::MotionState*>(ballBody->getMotionState());
    osg::Matrix m(osg::Matrix::translate(center));
    motion->setParentTransform(m);
    ballBody->setWorldTransform(osgbCollision::asBtTransform(m));

    dw->addRigidBody(ballBody);

    return(sphere);
}
osg::Node* Brickwall::createCube(const osg::Vec3& center, btDynamicsWorld* dw)
{
    osg::Node* brickNode = makeBrick(osg::Vec3(20, 5, 5), osg::Vec4(0, 0, 0, 0), osg::Vec4(255, 0, 0, 0.3));
    osg::ref_ptr< osgbDynamics::CreationRecord > cr = new osgbDynamics::CreationRecord;
    cr->_sceneGraph = brickNode;
    cr->_shapeType = BOX_SHAPE_PROXYTYPE;
    cr->_mass = 1.f;
    cr->_restitution = 0.f;
    cr->_com = osg::Vec3(2., 2., 2.);
    cr->_friction = 0.5f;




    brickBody = osgbDynamics::createRigidBody(cr.get(),
        osgbCollision::btBoxCollisionShapeFromOSG(brickNode));

    // Transform the box explicitly.
    osgbDynamics::MotionState* motion = dynamic_cast<osgbDynamics::MotionState*>(brickBody->getMotionState());
    osg::Matrix m(osg::Matrix::translate(center));
    m = m * osg::Matrix::rotate(0, 0, 0., 0.);
    motion->setParentTransform(m);
    brickBody->setWorldTransform(osgbCollision::asBtTransform(m));


    dw->addRigidBody(brickBody);
    return(brickNode);
}



bool Brickwall::init() {

    InteractionA = new coCombinedButtonInteraction(coInteraction::ButtonA, "Shoot", coInteraction::Medium);
    bulletWorld = initPhysics();
    root = new osg::Group;
    btCompoundShape* cs = new btCompoundShape;

    // create new Tab in TabFolder
    BrickTab = new ui::Menu("BrickWall", this);


    // create new ToggleButton in Tab1
    enableShooting = new ui::Button(BrickTab, "Shooting");
    enableShooting->setCallback([this](bool state) {
        if (state)
        {
            if (!InteractionA->isRegistered())
            {
                coInteractionManager::the()->registerInteraction(InteractionA);
            }
        }
        else
        {
            if (InteractionA->isRegistered())
            {
                coInteractionManager::the()->unregisterInteraction(InteractionA);
            }
        }
        });
    //create Ground
    float xDim(20.);
    float yDim(20.);
    centerOM.set(0., -50., 50.);

    ground = createGround(5 * xDim, 5 * yDim, osg::Vec3(0., 0., 0.), bulletWorld);
    root->addChild(ground.get());

    //create brickwall
    float xCen(-2);
    int zCen(0.);
    index = 0;
   

    for (int u = 1; u <= 10; u++) {

        for (int i = 0; i <= 3; i++) {

            osg::Vec3 brickCenter(20 + 40 * xCen, 90, 5 + 10 * zCen);
            bricks = createCube(brickCenter, bulletWorld);

            osg::ref_ptr<osg::Material> mat = new osg::Material;

            root->addChild(bricks.get());
            xCen++;
            if (u % 2 == 0 && i == 2) {
                i++;
                index++;
            }

        }

        xCen = -2 + 0.5 * (u % 2);

        zCen++;
    }
    /*motionState = new osgbDynamics::MotionState();
    int mass = 1;
    osg::Transform* box = makeBrick(osg::Vec3(20, 5, 5), osg::Vec4(0, 0, 0, 0), osg::Vec4(255, 0, 0, 0.3));
    
    btBoxShape* groundShape = osgbCollision::btBoxCollisionShapeFromOSG(box);

    btVector3 localInertia(0, 0, 0);
    btRigidBody::btRigidBodyConstructionInfo cInfo(mass, motionState, groundShape, localInertia);
   
    platform  = new btRigidBody(cInfo);

    bulletWorld -> addRigidBody(platform);
   */
  
    




    cover->getObjectsRoot()->addChild(root);

    return true;

}
bool Brickwall::update() {

    if (InteractionA->wasStarted())
    {
        osg::Matrix pm = cover->getPointerMat();
        osg::Vec3 start = pm.getTrans() * cover->getInvBaseMat(), target = osg::Y_AXIS * pm * cover->getInvBaseMat();
        target = (target - start);
        target.normalize(); 
        ball = createDynamicBall(4, 3, start, bulletWorld,target);
        root->addChild(ball);

    }
    
    static double lastTime = 0.;
    double curTime = cover->frameTime();
    double dt = curTime - lastTime;
    if (lastTime == 0.)
        dt = 0.01;

    /*osg::Vec3Array* verts(dynamic_cast<osg::Vec3Array*>(geom->getVertexArray()));
    size = 10;
    // Update the vertex array from the soft body node array.
    const btSoftBody::tNodeArray& nodes = softBody->m_nodes;
    osg::Vec3Array::iterator it(verts->begin());
    unsigned int idx;
    for (idx = 0; idx < size; idx++)
    {
       *it++ = osgbCollision::asOsgVec3(nodes[idx].m_n);
    }
    
    verts->dirty();
    geom->dirtyBound();
    
    // Generate new normals.
    osgUtil::SmoothingVisitor smooth;
    smooth.smooth(*geom);
    geom->getNormalArray()->dirty();
   */

    bulletWorld->stepSimulation(curTime - lastTime);
    worldInfo.m_sparsesdf.GarbageCollect();
    lastTime = curTime;



   
    
    return true;


}

/*void Brickwall::preFrame() {
    osg::Node* node;
    node = cover->getIntersectedNode();
    if (InteractionA->getState() == coInteraction::Active) {
        fprintf(stderr, "ESGEHT\n");
    }
}*/



