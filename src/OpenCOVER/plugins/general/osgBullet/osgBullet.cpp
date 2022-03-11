/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                            (C)2008 HLRS  **
 **                                                                          **
 ** Description: osgBulletPlugin OpenCOVER Plugin (is polite)                          **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner		                                                  **
 **                                                                          **
 ** History:  								                                         **
 ** June 2008  v1	    				       		                                **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include "osgBullet.h"
#include <cover/coVRPluginSupport.h>
using namespace opencover;


#include <osgDB/ReadFile>
#include <osgViewer/Viewer>
#include <osgViewer/ViewerEventHandlers>
#include <osgGA/TrackballManipulator>
#include <osg/Geode>
#include <osg/LightModel>
#include <osg/Texture2D>
#include <osgUtil/SmoothingVisitor>

#include <osgbDynamics/GroundPlane.h>
#include <osgbCollision/GLDebugDrawer.h>
#include <osgbCollision/Utils.h>
#include <osgbInteraction/DragHandler.h>
#include <osgbInteraction/LaunchHandler.h>
#include <osgbInteraction/SaveRestoreHandler.h>

#include <osgwTools/Shapes.h>

#include <btBulletDynamicsCommon.h>
#include <BulletSoftBody/btSoftBodyHelpers.h>
#include <BulletSoftBody/btSoftRigidDynamicsWorld.h>
#include <BulletSoftBody/btSoftBodyRigidBodyCollisionConfiguration.h>
#include <BulletSoftBody/btSoftBody.h>

#include <osg/io_utils>
#include <string>


osgBulletPlugin::osgBulletPlugin()
{
    fprintf(stderr, "osgBulletPlugin World\n");
}

// this is called if the plugin is removed at runtime
osgBulletPlugin::~osgBulletPlugin()
{
    fprintf(stderr, "Goodbye\n");
}

COVERPLUGIN(osgBulletPlugin)

btSoftBodyWorldInfo	worldInfo;



osg::Node* osgBulletPlugin:: makeFlag(btSoftRigidDynamicsWorld* bw)
{
    resX = 12;
    resY = 9;

    osg::ref_ptr< osg::Geode > geode(new osg::Geode);

    const osg::Vec3 llCorner(-2., 0., 5.);
    const osg::Vec3 uVec(4., 0., 0.);
    const osg::Vec3 vVec(0., 0.1, 3.); // Must be at a slight angle for wind to catch it.
 
    geom=osgwTools::makePlane(llCorner,uVec, vVec, osg::Vec2s(resX - 1, resY - 1));

    geode->addDrawable(geom);

    // Set up for dynamic data buffer objects
    geom->setDataVariance(osg::Object::DYNAMIC);
    geom->setUseDisplayList(false);
    geom->setUseVertexBufferObjects(true);
    geom->getOrCreateVertexBufferObject()->setUsage(GL_DYNAMIC_DRAW);

    // Flag state: 2-sided lighting and a texture map.
    {
        osg::StateSet* stateSet(geom->getOrCreateStateSet());

        osg::LightModel* lm(new osg::LightModel());
        lm->setTwoSided(true);
        stateSet->setAttributeAndModes(lm);

        const std::string texName("fort_mchenry_flag.jpg");
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


    // Create the soft body using a Bullet helper function. Note that
    // our update callback will update vertex data from the soft body
    // node data, so it's important that the corners and resolution
    // parameters produce node data that matches the vertex data.
        softBody = btSoftBodyHelpers::CreatePatch(worldInfo,
        osgbCollision::asBtVector3(llCorner),
        osgbCollision::asBtVector3(llCorner + uVec),
        osgbCollision::asBtVector3(llCorner + vVec),
        osgbCollision::asBtVector3(llCorner + uVec + vVec),
        resX, resY, 1 + 4, true);


    // Configure the soft body for interaction with the wind.
    softBody->getCollisionShape()->setMargin(0.1);
    softBody->m_materials[0]->m_kLST = 0.3;
    softBody->generateBendingConstraints(2, softBody->m_materials[0]);
    softBody->m_cfg.kLF = 0.05;
    softBody->m_cfg.kDG = 0.01;
    softBody->m_cfg.piterations = 2;
#if( BT_BULLET_VERSION >= 279 )
    softBody->m_cfg.aeromodel = btSoftBody::eAeroModel::V_TwoSidedLiftDrag;
#else
    // Hm. Not sure how to make the wind blow on older Bullet.
    // This doesn't seem to work.
    softBody->m_cfg.aeromodel = btSoftBody::eAeroModel::V_TwoSided;
#endif
    softBody->setWindVelocity(btVector3(50., 0., 0.));
    softBody->setTotalMass(1.);

    bw->addSoftBody(softBody);
    

    return(geode.release());
}

btSoftRigidDynamicsWorld* initPhysics()
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

/** \cond */

bool osgBulletPlugin::update() 
{
   
    static double lastTime = 0.;
    double curTime = cover->frameTime();
    double dt = curTime - lastTime;
    if (lastTime == 0.)
        dt = 0.01;

    osg::Vec3Array* verts(dynamic_cast<osg::Vec3Array*>(geom->getVertexArray()));
    size = resX * resY;
    // Update the vertex array from the soft body node array.
    const btSoftBody::tNodeArray& nodes = softBody->m_nodes;
    osg::Vec3Array::iterator it(verts->begin());
    unsigned int idx;
    for (idx = 0; idx < size; idx++)
    {
        *it++ = osgbCollision::asOsgVec3(nodes[idx].m_x);
    }
    verts->dirty();
    geom->dirtyBound();

    // Generate new normals.
    osgUtil::SmoothingVisitor smooth;
    smooth.smooth(*geom);
    geom->getNormalArray()->dirty();

    
    bw->stepSimulation(curTime - lastTime);
    worldInfo.m_sparsesdf.GarbageCollect();
    lastTime = curTime;

 

    return true;
}
 



bool osgBulletPlugin::init() {
    bw = initPhysics();
    osg::Group* root = new osg::Group;

    osg::Group* launchHandlerAttachPoint = new osg::Group;
    root->addChild(launchHandlerAttachPoint);


    osg::ref_ptr< osg::Node > rootModel(makeFlag(bw));
    if (!rootModel.valid())
    {
        osg::notify(osg::FATAL) << "mesh: Can't create flag mesh." << std::endl;
        return(1);
    }

    root->addChild(rootModel.get());




    // Add ground
    const osg::Vec4 plane(0., 0., 1., 0.);
    root->addChild(osgbDynamics::generateGroundPlane(plane, bw));

    cover->getObjectsRoot()->addChild(root);

    return true;
}


