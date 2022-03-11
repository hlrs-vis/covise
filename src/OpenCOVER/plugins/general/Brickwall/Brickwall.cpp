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

#include <OpenVRUI/coTrackerButtonInteraction.h>
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

#include "LinearMath/btVector3.h"
#include "LinearMath/btAlignedObjectArray.h"

#include "btBulletDynamicsCommon.h"
#include "CommonInterfaces/CommonRigidBodyBase.h"
#include "OpenGLWindow/ShapeData.h"

#include <OpenVRUI/coPanel.h>
#include <OpenVRUI/coFrame.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/coNavInteraction.h>
#include <OpenVRUI/coMenu.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coPotiMenuItem.h>
#include <OpenVRUI/coFlatPanelGeometry.h>
#include <OpenVRUI/coFlatButtonGeometry.h>
#include <OpenVRUI/coRectButtonGeometry.h>
#include <OpenVRUI/coMouseButtonInteraction.h>



#include <string>

#include "Brickwall.h"
#include <cover/coVRPluginSupport.h>

using namespace opencover;
Brickwall::Brickwall()
{
    fprintf(stderr, "Brickwall World\n");
}

// this is called if the plugin is removed at runtime
Brickwall::~Brickwall()
{
    fprintf(stderr, "Goodbye\n");
    delete InteractionA;
}


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

osg::Node* Brickwall::createGround(float w, float h, const osg::Vec3& center, btSoftRigidDynamicsWorld* dw)
{
    osg::Transform* ground = createOSGBox(osg::Vec3(w, h, 0.3), osg::Vec4(0., 0., 0., 45.), osg::Vec4(1., 1., 1., 1.));


    osg::ref_ptr< osgbDynamics::CreationRecord > cr = new osgbDynamics::CreationRecord;
    cr->_sceneGraph = ground;
    cr->_shapeType = BOX_SHAPE_PROXYTYPE;
    cr->_mass = 0.f;
    cr->_restitution = 0.5f;
    cr->_friction = 0.9f;
    btRigidBody* body = osgbDynamics::createRigidBody(cr.get(), osgbCollision::btBoxCollisionShapeFromOSG(ground));


    // Transform the box explicitly.
    osgbDynamics::MotionState* motion = dynamic_cast<osgbDynamics::MotionState*>(body->getMotionState());
    osg::Matrix m(osg::Matrix::translate(center));
    motion->setParentTransform(m);
    body->setWorldTransform(osgbCollision::asBtTransform(m));

    dw->addRigidBody(body);

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

osg::Transform* Brickwall::createDynamicBall(float mass, float radius, const osg::Vec3& center, btDynamicsWorld* dw) {
    osg::Transform* ball = createBall(radius);
    osgbCollision::AXIS axis(osgbCollision::Z);

    osg::ref_ptr< osgbDynamics::CreationRecord> cr = new osgbDynamics::CreationRecord;
    cr->_sceneGraph = ball;
    cr->_shapeType = SPHERE_SHAPE_PROXYTYPE;
    cr->_mass = mass;
    cr->_axis = axis;
    cr->_restitution = 0.6;
    cr->_friction = 0.5f;
    cr->_reductionLevel = osgbDynamics::CreationRecord::MINIMAL;
    btRigidBody* body = osgbDynamics::createRigidBody(cr.get(), osgbCollision::btBoxCollisionShapeFromOSG(ball));
    body->applyImpulse(btVector3(0., 500., 0.), btVector3(0., -50., 50.));

    // Transform the box explicitly.
    osgbDynamics::MotionState* motion = dynamic_cast<osgbDynamics::MotionState*>(body->getMotionState());
    osg::Matrix m(osg::Matrix::translate(center));
    motion->setParentTransform(m);
    body->setWorldTransform(osgbCollision::asBtTransform(m));

    dw->addRigidBody(body);

    return(ball);
}
osg::Node* Brickwall::createCube(const osg::Vec3& center, btSoftRigidDynamicsWorld* dw)
{
    osg::Node* brick = makeBrick(osg::Vec3(20, 5, 5), osg::Vec4(0, 0, 0, 0), osg::Vec4(255, 0, 0, 0.3));





    osg::ref_ptr< osgbDynamics::CreationRecord > cr = new osgbDynamics::CreationRecord;
    cr->_sceneGraph = brick;
    cr->_shapeType = BOX_SHAPE_PROXYTYPE;
    cr->_mass = 1.f;
    cr->_restitution = 0.f;
    cr->_com = osg::Vec3(2., 2., 2.);
    cr->_friction = 0.5f;




    btRigidBody* body = osgbDynamics::createRigidBody(cr.get(),
        osgbCollision::btBoxCollisionShapeFromOSG(brick));

    // Transform the box explicitly.
    osgbDynamics::MotionState* motion = dynamic_cast<osgbDynamics::MotionState*>(body->getMotionState());
    osg::Matrix m(osg::Matrix::translate(center));
    m = m * osg::Matrix::rotate(0, 0, 0., 0.);
    motion->setParentTransform(m);
    body->setWorldTransform(osgbCollision::asBtTransform(m));





    dw->addRigidBody(body);
    return(brick);
}



bool Brickwall::init() {

    //InteractionA = new coMouseButtonInteraction(coInteraction::ButtonA, "Shoot", coInteraction::Menu);
    bulletWorld = initPhysics();
    root = new osg::Group;
    btCompoundShape* cs = new btCompoundShape;

    //create Ground
    float xDim(20.);
    float yDim(20.);
    osg::Vec3 centerOM(0., -50., 50.);

    osg::ref_ptr< osg::Node > ground = createGround(5 * xDim, 5 * yDim, osg::Vec3(0., 0., 0.), bulletWorld);
    root->addChild(ground.get());

    //create brickwall
    float xCen(-2);
    int zCen(0.);

    ->enableIntersection();

    for (int u = 1; u <= 10; u++) {

        for (int i = 0; i <= 3; i++) {

            osg::Vec3 brickCenter(20 + 40 * xCen, 90, 5 + 10 * zCen);
            osg::ref_ptr< osg::Node > brick = createCube(brickCenter, bulletWorld);

            osg::ref_ptr<osg::Material> mat = new osg::Material;

            root->addChild(brick.get());
            xCen++;
            if (u % 2 == 0 && i == 2) {
                i++;
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
    float mass = 10;
    float radius = 10;

    osg::Transform* ball = createDynamicBall(mass, radius, centerOM, bulletWorld);
    root->addChild(ball);




    cover->getObjectsRoot()->addChild(root);

    return true;

}
bool Brickwall::update() {


    
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

void Brickwall::preFrame() {
    osg::Node* node;
    node= node = cover->getIntersectedNode();
    if (InteractionA->getState() == coInteraction::Active) {
        fprintf(stderr, "ESGEHT\n");
    }
}



