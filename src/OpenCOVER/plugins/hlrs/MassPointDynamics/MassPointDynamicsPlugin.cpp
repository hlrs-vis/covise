/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                            (C)2001 HLRS  **
 **                                                                          **
 ** Description: MassPointDynamics Plugin (does nothing)                              **
 **                                                                          **
 **                                                                          **
 ** Author: Florian Seybold		                                                **
 **                                                                          **
 ** History:  								                                **
 ** Nov-01  v1	    				       		                            **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include "MassPointDynamicsPlugin.h"
#include <cover/coVRPluginSupport.h>
#include <cover/RenderObject.h>

MassPointDynamicsPlugin::MassPointDynamicsPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
, system()
, massPointVector(system.getMassPointVector())
, jointVector(system.getJointVector())
, integrator(massPointVector, jointVector)
, time(0.0)
, sphere(NULL)
, sphereDrawable(NULL)
, massPointGeode(NULL)
, planeVertices(NULL)
, planeBase(NULL)
, planeGeometry(NULL)
, planeGeode(NULL)

{
    fprintf(stderr, "MassPointDynamicsPlugin::MassPointDynamicsPlugin\n");

    //system.buildDoubleTetrahedron(Vec3d(0.0, 0.0, 20.0), Vec3d(0.0,0.0,0.0), Vec3d(10.0,10.0,0.0));
    system.buildBall(Vec3d(0.0, 0.0, 20.0), Vec3d(0.0, 0.0, 0.0), Vec3d(0.0, 0.0, 0.0), 10.0);

    //systemIntegrator.addState(system);
    /*mp1 = new MassPoint(Vec3d(1.443, 0.0, 24.082), Vec3d(-50.0, 5.0, 0.0)); system.addMassPoint(mp1);
   mp2 = new MassPoint(Vec3d(0.0, -2.5, 20.0), Vec3d(-50.0, 0.0, 0.0)); system.addMassPoint(mp2);
   mp3 = new MassPoint(Vec3d(0.0, 2.5, 20.0), Vec3d(-50.0, 0.0, 0.0)); system.addMassPoint(mp3);
   mp4 = new MassPoint(Vec3d(4.33, 0.0, 20.0), Vec3d(-50.0, 0.0, 0.0)); system.addMassPoint(mp4);
   mp5 = new MassPoint(Vec3d(1.443, 0.0, 15.918), Vec3d(-50.0, -5.0, 0.0)); system.addMassPoint(mp5);*/
    /*massPointVector.push_back(MassPoint(Vec3d(1.443, 0.0, 24.082), Vec3d(-50.0, 5.0, 0.0))); unsigned int mp1 = massPointVector.size() - 1;
   massPointVector.push_back(MassPoint(Vec3d(0.0, -2.5, 20.0), Vec3d(-50.0, 0.0, 0.0))); unsigned int mp2 = massPointVector.size() - 1;
   massPointVector.push_back(MassPoint(Vec3d(0.0, 2.5, 20.0), Vec3d(-50.0, 0.0, 0.0))); unsigned int mp3 = massPointVector.size() - 1;
   massPointVector.push_back(MassPoint(Vec3d(4.33, 0.0, 20.0), Vec3d(-50.0, 0.0, 0.0))); unsigned int mp4 = massPointVector.size() - 1;
   massPointVector.push_back(MassPoint(Vec3d(1.443, 0.0, 15.918), Vec3d(-50.0, -5.0, 0.0))); unsigned int mp5 = massPointVector.size() - 1;

   jointVector.push_back(new SpringDamperJoint(mp1, mp2, &massPointVector));
   jointVector.push_back(new SpringDamperJoint(mp1, mp3, &massPointVector));
   jointVector.push_back(new SpringDamperJoint(mp1, mp4, &massPointVector));
   jointVector.push_back(new SpringDamperJoint(mp2, mp3, &massPointVector));
   jointVector.push_back(new SpringDamperJoint(mp2, mp4, &massPointVector));
   jointVector.push_back(new SpringDamperJoint(mp2, mp5, &massPointVector));
   jointVector.push_back(new SpringDamperJoint(mp3, mp4, &massPointVector));
   jointVector.push_back(new SpringDamperJoint(mp3, mp5, &massPointVector));
   jointVector.push_back(new SpringDamperJoint(mp4, mp5, &massPointVector));
   for(unsigned int i=0; i<massPointVector.size(); ++i) {
      jointVector.push_back(new GroundContactJoint(i));
      jointVector.push_back(new GravityJoint(i));
   }*/

    /*system.addJoint(new SpringDamperJoint(mp1, mp2));
   system.addJoint(new SpringDamperJoint(mp1, mp3));
   system.addJoint(new SpringDamperJoint(mp1, mp4));
   system.addJoint(new SpringDamperJoint(mp2, mp3));
   system.addJoint(new SpringDamperJoint(mp2, mp4));
   system.addJoint(new SpringDamperJoint(mp2, mp5));
   system.addJoint(new SpringDamperJoint(mp3, mp4));
   system.addJoint(new SpringDamperJoint(mp3, mp5));
   system.addJoint(new SpringDamperJoint(mp4, mp5));
   system.addJoint(new GroundContactJoint(mp1));
   system.addJoint(new GroundContactJoint(mp2));
   system.addJoint(new GroundContactJoint(mp3));
   system.addJoint(new GroundContactJoint(mp4));
   system.addJoint(new GroundContactJoint(mp5));
   system.addJoint(new GravityJoint(mp1));
   system.addJoint(new GravityJoint(mp2));
   system.addJoint(new GravityJoint(mp3));
   system.addJoint(new GravityJoint(mp4));
   system.addJoint(new GravityJoint(mp5));*/
}

bool MassPointDynamicsPlugin::init()
{
    sphere = new osg::Sphere(osg::Vec3(0, 0, 0), 1.0f);
    sphereDrawable = new osg::ShapeDrawable(sphere);
    massPointGeode = new osg::Geode();
    massPointGeode->addDrawable(sphereDrawable);

    for (unsigned int i = 0; i < massPointVector->size(); ++i)
    {
        transformVector.push_back(new osg::PositionAttitudeTransform());
        transformVector[i]->addChild(massPointGeode);
        cover->getObjectsRoot()->addChild(transformVector[i]);
    }

    planeVertices = new osg::Vec3Array;
    planeVertices->push_back(osg::Vec3(1000.0, 1000.0, 0.0));
    planeVertices->push_back(osg::Vec3(-1000.0, 1000.0, 0.0));
    planeVertices->push_back(osg::Vec3(1000.0, -1000.0, 0.0));
    planeVertices->push_back(osg::Vec3(-1000.0, -1000.0, 0.0));

    planeBase = new osg::DrawArrays(osg::PrimitiveSet::TRIANGLE_STRIP, 0, 4);

    planeGeometry = new osg::Geometry;
    planeGeometry->setVertexArray(planeVertices);
    planeGeometry->addPrimitiveSet(planeBase);
    planeGeode = new osg::Geode;
    planeGeode->addDrawable(planeGeometry);
    cover->getObjectsRoot()->addChild(planeGeode);

    return true;
}

// this is called if the plugin is removed at runtime
MassPointDynamicsPlugin::~MassPointDynamicsPlugin()
{
    fprintf(stderr, "MassPointDynamicsPlugin::~MassPointDynamicsPlugin\n");
}

void
MassPointDynamicsPlugin::preFrame()
{
    double dt = cover->frameDuration();
    time += dt;

    integrator.integrate(dt);

    for (unsigned int i = 0; i < massPointVector->size(); ++i)
    {
        transformVector[i]->setPosition(osg::Vec3((*massPointVector)[i].r.x, (*massPointVector)[i].r.y, (*massPointVector)[i].r.z));
    }

    /*Vec3d rFirst = mp1->getR();
   firstBoxTransform->setPosition(osg::Vec3(rFirst.x, rFirst.y, rFirst.z));
   Vec3d rSecond = mp2->getR();
   secondBoxTransform->setPosition(osg::Vec3(rSecond.x, rSecond.y, rSecond.z));
   Vec3d rThird = mp3->getR();
   thirdBoxTransform->setPosition(osg::Vec3(rThird.x, rThird.y, rThird.z));
   Vec3d rFourth = mp4->getR();
   fourthBoxTransform->setPosition(osg::Vec3(rFourth.x, rFourth.y, rFourth.z));
   Vec3d rFifth = mp5->getR();
   fifthBoxTransform->setPosition(osg::Vec3(rFifth.x, rFifth.y, rFifth.z));*/
}

COVERPLUGIN(MassPointDynamicsPlugin)
