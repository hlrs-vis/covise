/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
**                                                            (C)2001 HLRS  **
**                                                                          **
** Description: Template Plugin (does nothing)                              **
**                                                                          **
**                                                                          **
** Author: U.Woessner		                                                **
**                                                                          **
** History:  								                                **
** Nov-01  v1	    				       		                            **
**                                                                          **
**                                                                          **
\****************************************************************************/

#include "BulletPlugin.h"
#include <cover/coVRPluginSupport.h>
#include <cover/RenderObject.h>
#include "VrmlNodePhysics.h"
#include <vrml97/vrml/Player.h>
#include <vrml97/vrml/VrmlNodeCOVER.h>
#include <vrml97/vrml/VrmlScene.h>
#include <vrml97/vrml/VrmlMFString.h>
#include <vrml97/vrml/Doc.h>
#include <vrml97/vrml/coEventQueue.h>
#include <vrml97/vrml/VrmlNamespace.h>
using namespace covise;

BulletPlugin::BulletPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    fprintf(stderr, "BulletPlugin::BulletPlugin\n");
    bulletWorld = initPhysics();
    osg::Group *root = cover->getObjectsRoot();

    root->addChild(makeDie(bulletWorld));
    root->addChild(makeDie(bulletWorld));

    // BEGIN: Create environment boxes
    float xDim(60.);
    float yDim(60.);
    float zDim(22.);
    float thick(2.5);

    osg::MatrixTransform *shakeBox = new osg::MatrixTransform;
    btCompoundShape *cs = new btCompoundShape;
    { // floor -Z (far back of the shake cube)
        osg::Vec3 halfLengths(xDim * .5, yDim * .5, thick * .5);
        osg::Vec3 center(0., 0., -zDim * .5);
        shakeBox->addChild(osgBox(center, halfLengths));
        btBoxShape *box = new btBoxShape(osgbBullet::asBtVector3(halfLengths));
        btTransform trans;
        trans.setIdentity();
        trans.setOrigin(osgbBullet::asBtVector3(center));
        cs->addChildShape(trans, box);
    }
    { // top +Z (invisible, to allow user to see through; no OSG analogue
        osg::Vec3 halfLengths(xDim * .5, yDim * .5, thick * .5);
        osg::Vec3 center(0., 0., zDim * .5);
        //shakeBox->addChild( osgBox( center, halfLengths ) );
        btBoxShape *box = new btBoxShape(osgbBullet::asBtVector3(halfLengths));
        btTransform trans;
        trans.setIdentity();
        trans.setOrigin(osgbBullet::asBtVector3(center));
        cs->addChildShape(trans, box);
    }
    { // left -X
        osg::Vec3 halfLengths(thick * .5, yDim * .5, zDim * .5);
        osg::Vec3 center(-xDim * .5, 0., 0.);
        shakeBox->addChild(osgBox(center, halfLengths));
        btBoxShape *box = new btBoxShape(osgbBullet::asBtVector3(halfLengths));
        btTransform trans;
        trans.setIdentity();
        trans.setOrigin(osgbBullet::asBtVector3(center));
        cs->addChildShape(trans, box);
    }
    { // right +X
        osg::Vec3 halfLengths(thick * .5, yDim * .5, zDim * .5);
        osg::Vec3 center(xDim * .5, 0., 0.);
        shakeBox->addChild(osgBox(center, halfLengths));
        btBoxShape *box = new btBoxShape(osgbBullet::asBtVector3(halfLengths));
        btTransform trans;
        trans.setIdentity();
        trans.setOrigin(osgbBullet::asBtVector3(center));
        cs->addChildShape(trans, box);
    }
    { // bottom of window -Y
        osg::Vec3 halfLengths(xDim * .5, thick * .5, zDim * .5);
        osg::Vec3 center(0., -yDim * .5, 0.);
        shakeBox->addChild(osgBox(center, halfLengths));
        btBoxShape *box = new btBoxShape(osgbBullet::asBtVector3(halfLengths));
        btTransform trans;
        trans.setIdentity();
        trans.setOrigin(osgbBullet::asBtVector3(center));
        cs->addChildShape(trans, box);
    }
    { // bottom of window -Y
        osg::Vec3 halfLengths(xDim * .5, thick * .5, zDim * .5);
        osg::Vec3 center(0., yDim * .5, 0.);
        shakeBox->addChild(osgBox(center, halfLengths));
        btBoxShape *box = new btBoxShape(osgbBullet::asBtVector3(halfLengths));
        btTransform trans;
        trans.setIdentity();
        trans.setOrigin(osgbBullet::asBtVector3(center));
        cs->addChildShape(trans, box);
    }
    // END: Create environment boxes

    btScalar mass(0.0);
    btVector3 inertia(0, 0, 0);

    shakeMotion = new osgbBullet::MotionState();
    shakeMotion->setTransform(shakeBox);
    btRigidBody::btRigidBodyConstructionInfo rb(mass, shakeMotion, cs, inertia);
    btRigidBody *shakeBody = new btRigidBody(rb);
    shakeBody->setCollisionFlags(shakeBody->getCollisionFlags() | btCollisionObject::CF_KINEMATIC_OBJECT);
    shakeBody->setActivationState(DISABLE_DEACTIVATION);
    bulletWorld->addRigidBody(shakeBody);

    root->addChild(shakeBox);

    pointerShape = new btConeShape(100, 2000);
    pointerMotion = new osgbBullet::MotionState();
    btRigidBody::btRigidBodyConstructionInfo prb(mass, pointerMotion, pointerShape, inertia);

    pointerBody = new btRigidBody(prb);
    pointerBody->setCollisionFlags(pointerBody->getCollisionFlags() | btCollisionObject::CF_KINEMATIC_OBJECT);
    pointerBody->setActivationState(DISABLE_DEACTIVATION);
    bulletWorld->addRigidBody(pointerBody);

    VrmlNodePhysics::bw = bulletWorld;
    VrmlNamespace::addBuiltIn(VrmlNodePhysics::defineType());
}

// this is called if the plugin is removed at runtime
BulletPlugin::~BulletPlugin()
{
    fprintf(stderr, "BulletPlugin::~BulletPlugin\n");
}

osg::MatrixTransform *
BulletPlugin::makeDie(btDynamicsWorld *bw)
{
    osg::MatrixTransform *root = new osg::MatrixTransform;
    const std::string fileName("dice.osg");
    osg::Node *node = osgDB::readNodeFile(fileName);
    if (node == NULL)
    {
        osg::notify(osg::FATAL) << "Can't find \"" << fileName << "\". Make sure OSG_FILE_PATH includes the osgBullet data directory." << std::endl;
        return NULL;
    }
    root->addChild(node);

    btCollisionShape *cs = osgbBullet::btBoxCollisionShapeFromOSG(node);
    osgbBullet::MotionState *motion = new osgbBullet::MotionState();
    motion->setTransform(root);
    btScalar mass(1.);
    btVector3 inertia(0, 0, 0);
    cs->calculateLocalInertia(mass, inertia);
    btRigidBody::btRigidBodyConstructionInfo rb(mass, motion, cs, inertia);
    btRigidBody *body = new btRigidBody(rb);
    //body->setActivationState( DISABLE_DEACTIVATION );
    bw->addRigidBody(body);

    return (root);
}

osg::Geode *BulletPlugin::osgBox(const osg::Vec3 &center, const osg::Vec3 &halfLengths)
{
    osg::Vec3 l(halfLengths * 2.);
    osg::Box *box = new osg::Box(center, l.x(), l.y(), l.z());
    osg::ShapeDrawable *shape = new osg::ShapeDrawable(box);
    shape->setColor(osg::Vec4(1., 1., 1., 1.));
    osg::Geode *geode = new osg::Geode();
    geode->addDrawable(shape);
    return (geode);
}

btDynamicsWorld *
BulletPlugin::initPhysics()
{
    btDefaultCollisionConfiguration *collisionConfiguration = new btDefaultCollisionConfiguration();
    btCollisionDispatcher *dispatcher = new btCollisionDispatcher(collisionConfiguration);
    btConstraintSolver *solver = new btSequentialImpulseConstraintSolver;

    btVector3 worldAabbMin(-10000, -10000, -10000);
    btVector3 worldAabbMax(10000, 10000, 10000);
    btBroadphaseInterface *inter = new btAxisSweep3(worldAabbMin, worldAabbMax, 1000);

    btDynamicsWorld *dynamicsWorld = new btDiscreteDynamicsWorld(dispatcher, inter, solver, collisionConfiguration);

    dynamicsWorld->setGravity(btVector3(0, 0, -9.8));
    //dynamicsWorld->setGravity( btVector3( 0, -1000, 0 ) );//vrml

    return (dynamicsWorld);
}

void
BulletPlugin::preFrame()
{

    osg::Matrix base = cover->getXformMat();
    /*base.makeIdentity();
   //base.makeRotate(M_PI,osg::Vec3(1,0,0));
   static float x=0,y=0,z=0;
   x+=0.05;
   if(x > 2.0)
   x=0;
   base.setTrans(x,y,z);*/

    /*btMatrix3x3 rot(base(0,0),base(0,1),base(0,2),base(1,0),base(1,1),base(1,2),base(2,0),base(2,1),base(2,2));
   btVector3 trans (base(3,0),base(3,1),base(3,2));
   btTransform world(rot,trans);*/

    /*btTransform world=osgbBullet::asBtTransform(base);
   shakeMotion->setWorldTransform( world );*/

    osg::Matrix pointerInObject = cover->getPointerMat() * cover->getInvBaseMat();
    osg::Vec3 v1(pointerInObject(0, 0), pointerInObject(0, 1), pointerInObject(0, 2));
    osg::Vec3 v2(pointerInObject(1, 0), pointerInObject(1, 1), pointerInObject(1, 2));
    osg::Vec3 v3(pointerInObject(2, 0), pointerInObject(2, 1), pointerInObject(2, 2));
    v1.normalize();
    v2.normalize();
    v3.normalize();
    pointerInObject(0, 0) = v1[0];
    pointerInObject(0, 1) = v1[1];
    pointerInObject(0, 2) = v1[2];
    pointerInObject(1, 0) = v2[0];
    pointerInObject(1, 1) = v2[1];
    pointerInObject(1, 2) = v2[2];
    pointerInObject(2, 0) = v3[0];
    pointerInObject(2, 1) = v3[1];
    pointerInObject(2, 2) = v3[2];

    pointerMotion->setWorldTransform(osgbBullet::asBtTransform(pointerInObject));
    static float oldScale(1.0);
    if (oldScale != cover->getScale())
    {
        oldScale = cover->getScale();
        bulletWorld->removeRigidBody(pointerBody);
        delete pointerBody;
        btScalar mass(0.0);
        btVector3 inertia(0, 0, 0);

        pointerShape = new btConeShape(100 / oldScale, 2000);
        btRigidBody::btRigidBodyConstructionInfo prb(mass, pointerMotion, pointerShape, inertia);

        pointerBody = new btRigidBody(prb);
        pointerBody->setCollisionFlags(pointerBody->getCollisionFlags() | btCollisionObject::CF_KINEMATIC_OBJECT);
        pointerBody->setActivationState(DISABLE_DEACTIVATION);
        bulletWorld->addRigidBody(pointerBody);
    }

    double duration = cover->frameDuration();
    if (duration > 10.0)
        duration = 1 / 60.0;
    bulletWorld->stepSimulation(duration, 4, duration / 4.);
}

COVERPLUGIN(BulletPlugin)
