/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 2001 Uwe Woessner
//
//  %W% %G%
//  VrmlNodeCrawler.cpp
#ifdef _WIN32
#if (_MSC_VER >= 1300) && !(defined(MIDL_PASS) || defined(RC_INVOKED))
#define POINTER_64 __ptr64
#else
#define POINTER_64
#endif
#include <winsock2.h>
#include <windows.h>
#endif
#include <util/common.h>
#include <vrml97/vrml/config.h>
#include <vrml97/vrml/VrmlNodeType.h>
#include <vrml97/vrml/coEventQueue.h>

#include <vrml97/vrml/MathUtils.h>
#include <vrml97/vrml/System.h>
#include <vrml97/vrml/Viewer.h>
#include <vrml97/vrml/VrmlScene.h>
#include <cover/VRViewer.h>
#include <cover/VRSceneGraph.h>
#include <cover/coVRAnimationManager.h>
#include <cover/coVRPluginSupport.h>
#include <math.h>
#include "VrmlNodeCrawler.h"
#include "CrawlerPlugin.h"



// Crawler factory.

static VrmlNode *creator(VrmlScene *scene)
{
    
    VrmlNodeCrawler *var = new VrmlNodeCrawler(scene);
    return var;
}


// Define the built in VrmlNodeType:: "Crawler" fields

VrmlNodeType *VrmlNodeCrawler::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st; // Only define the type once.
        t = st = new VrmlNodeType("Crawler", creator);
    }

    VrmlNodeChild::defineType(t); // Parent class
    
    t->addExposedField("speed", VrmlField::SFFLOAT);
    t->addExposedField("lademodus", VrmlField::SFBOOL);
    t->addEventOut("position",VrmlField::SFVEC3F);
    t->addEventOut("rotation",VrmlField::SFROTATION);
    t->addEventOut("main0Angle",VrmlField::SFFLOAT);
    t->addEventOut("main1Angle",VrmlField::SFFLOAT);
    t->addEventOut("main2Angle",VrmlField::SFFLOAT);
    t->addEventOut("sec0Angle",VrmlField::SFFLOAT);
    t->addEventOut("sec1Angle",VrmlField::SFFLOAT);
    t->addEventOut("sec2Angle",VrmlField::SFFLOAT);
    
    VrmlSFFloat d_main0Angle;
    VrmlSFFloat d_main1Angle;
    VrmlSFFloat d_main2Angle;
    VrmlSFFloat d_sec0Angle;
    VrmlSFFloat d_sec1Angle;
    VrmlSFFloat d_sec2Angle;

    return t;
}

VrmlNodeType *VrmlNodeCrawler::nodeType() const
{
    return defineType(0);
}

VrmlNodeCrawler::VrmlNodeCrawler(VrmlScene *scene)
    : VrmlNodeChild(scene)
    , d_speed(1.0)
    , d_lademodus(true)
    , d_position(0,0,0)
    , d_rotation(1,0,0,0)
    , d_main0Angle(0.0)
    , d_main1Angle(0.0)
    , d_main2Angle(0.0)
    , d_sec0Angle(0.0)
    , d_sec1Angle(0.0)
    , d_sec2Angle(0.0)
{
    setModified();
    
    tetraederActor = NULL;


    revoluteJointBMF = NULL;
    revoluteJointBSF = NULL;


    gSelectedActorDynamic = NULL;
    /*-----------------------------------------------------------------------------------------------------------------------------------------*/
    //-----Create Tetraeder-----
    PxVec3 vertsTetraeder[4] = {PxVec3(0.3 , 0, (-0.173205)), PxVec3((-0.3), 0, (-0.173205)), PxVec3(0, 0, 0.34641), PxVec3(0, 0.489898, 0)};
    PxConvexMesh* tetraederconvexMesh = CrawlerPlugin::plugin->createConvexMesh(vertsTetraeder, 4, *CrawlerPlugin::plugin->gPhysics, *CrawlerPlugin::plugin->gCooking);
    PxShape* tetraederShape = CrawlerPlugin::plugin->gPhysics->createShape(PxConvexMeshGeometry(tetraederconvexMesh), &CrawlerPlugin::plugin->gMaterial, PxShapeFlag::eSIMULATION_SHAPE);
    tetraederShape->setContactOffset(0.001);

    PxTransform localPosTetraeder(PxVec3(1.2, 0.014, 1.2));
    tetraederActor = CrawlerPlugin::plugin->gPhysics->createRigidDynamic(localPosTetraeder);
    tetraederActor->attachShape(*tetraederShape);
    tetraederActor->setMass(19.475);
    tetraederActor->setCMassLocalPose(PxTransform(PxVec3(0, 0.122, 0)));

    tetraederActor->setActorFlag(PxActorFlag::eVISUALIZATION, false);					//Koordinatensystem an/aus

    CrawlerPlugin::plugin->gScene->addActor(*tetraederActor);		//Add Tetraeder to Scene

    /*-----------------------------------------------------------------------------------------------------------------------------------------*/
    //-----Create Bottom Flaps-----
    //-----Create Bottom Main Flap-----
    PxVec3 BMFverts[6] = {PxVec3(0.255, 0.0015, 0.01), PxVec3((-0.255), 0.0015, 0.01), PxVec3(0, 0.0015, 0.451673), PxVec3(0.255, (-0.0015), 0.01), PxVec3((-0.255), (-0.0015), 0.01), PxVec3(0, (-0.0015), 0.451673)};
    PxConvexMesh* BMFconvexMesh = CrawlerPlugin::plugin->createConvexMesh(BMFverts, 6, *CrawlerPlugin::plugin->gPhysics, *CrawlerPlugin::plugin->gCooking);
    PxShape* BMFShape = CrawlerPlugin::plugin->gPhysics->createShape(PxConvexMeshGeometry(BMFconvexMesh), &CrawlerPlugin::plugin->gMaterial, PxShapeFlag::eSIMULATION_SHAPE);
    BMFShape->setRestOffset(0.0);
    PxRigidDynamic* BMFActor = CrawlerPlugin::plugin->gPhysics->createRigidDynamic(PxTransform(PxVec3(0,0,0)));

    BMFActor->setGlobalPose(PxTransform(PxVec3(0.13616,0.009,0.078612), PxQuat(physx::PxMat33(PxVec3(-0.5,0,0.86603), PxVec3(0,1,0), PxVec3(-0.86603,0,-0.5)))));	//-120 deg Y

    BMFActor->attachShape(*BMFShape);
    BMFActor->setMass(0.059);
    BMFActor->setCMassLocalPose(PxTransform(PxVec3(0, 0, 0.157)));
    BMFActor->setActorFlag(PxActorFlag::eVISUALIZATION, false);						//Koordinatensystem an/aus

    CrawlerPlugin::plugin->gScene->addActor(*BMFActor);		//Add BMFActor to Scene

    revoluteJointBMF = PxRevoluteJointCreate(*CrawlerPlugin::plugin->gPhysics, tetraederActor, PxTransform(PxVec3(0.13616,-0.005,0.078612), PxQuat(physx::PxMat33(PxVec3(-0.5,0,0.86603), PxVec3(0,1,0), PxVec3(-0.86603,0,-0.5)))), BMFActor, PxTransform(PxVec3(0,0,0)));
    revoluteJointBMF->setConstraintFlag(PxConstraintFlag::eCOLLISION_ENABLED, true);

    revoluteJointBMF->setProjectionLinearTolerance(0.1f);
    revoluteJointBMF->setConstraintFlag(PxConstraintFlag::ePROJECTION, true);

    PxJointAngularLimitPair BotMainFlapLimitPair(-PxPi*PxReal(0.002466513), PxPi*PxReal(0.611111), 0.1f);		//0,611111 Pi = 110 deg
    revoluteJointBMF->setLimit(BotMainFlapLimitPair);
    revoluteJointBMF->setRevoluteJointFlag(PxRevoluteJointFlag::eLIMIT_ENABLED, true);

    revoluteJointBMF->setDriveVelocity(-1.0f);
    revoluteJointBMF->setRevoluteJointFlag(PxRevoluteJointFlag::eDRIVE_ENABLED, true);

    revoluteJointBMF->setConstraintFlag(PxConstraintFlag::eVISUALIZATION, true);			//Joint Koordinatensysteme/Limits an/aus

    //-----Create Bottom Secondary Flap-----
    PxVec3 BSFverts[6] = {PxVec3(0.2, 0.0015, 0.01), PxVec3((-0.2), 0.0015, 0.01), PxVec3(0, 0.0015, 0.356), PxVec3(0.2, (-0.0015), 0.01), PxVec3((-0.2), (-0.0015), 0.01), PxVec3(0, (-0.0015), 0.356)};
    //PxVec3 BSFverts[6] = {PxVec3(0.255, 0.0015, 0.01), PxVec3((-0.255), 0.0015, 0.01), PxVec3(0, 0.0015, 0.451673), PxVec3(0.255, (-0.0015), 0.01), PxVec3((-0.255), (-0.0015), 0.01), PxVec3(0, (-0.0015), 0.451673)};
    PxConvexMesh* BSFconvexMesh = CrawlerPlugin::plugin->createConvexMesh(BSFverts, 6, *CrawlerPlugin::plugin->gPhysics, *CrawlerPlugin::plugin->gCooking);
    PxShape* BSFShape = CrawlerPlugin::plugin->gPhysics->createShape(PxConvexMeshGeometry(BSFconvexMesh), &CrawlerPlugin::plugin->gMaterial, PxShapeFlag::eSIMULATION_SHAPE);
    BSFShape->setRestOffset(0.0);
    PxRigidDynamic* BSFActor = CrawlerPlugin::plugin->gPhysics->createRigidDynamic(PxTransform(PxVec3(0,0,0)));

    BSFActor->setGlobalPose(PxTransform(PxVec3(-0.13616, 0.005, 0.078612), PxQuat(physx::PxMat33(PxVec3(-0.5,0,-0.86603),PxVec3(0,1,0),PxVec3(0.86603,0,-0.5)))));		//120 deg Y-Achse 

    BSFActor->attachShape(*BSFShape);
    BSFActor->setMass(0.049);
    BSFActor->setCMassLocalPose(PxTransform(PxVec3(0, 0, 0.125)));
    //BSFActor->setCMassLocalPose(PxTransform(PxVec3(0, 0, 0.157)));
    BSFActor->setActorFlag(PxActorFlag::eVISUALIZATION, false);						//Koordinatensystem an/aus

    CrawlerPlugin::plugin->gScene->addActor(*BSFActor);		//Add BSFActor to Scene

    revoluteJointBSF = PxRevoluteJointCreate(*CrawlerPlugin::plugin->gPhysics, BMFActor, PxTransform(PxVec3(0.13616, -0.004, 0.235836), PxQuat(physx::PxMat33(PxVec3(-0.5,0,0.86603),PxVec3(0,1,0),PxVec3(-0.86603,0,-0.5)))), BSFActor, PxTransform(PxVec3(0,0,0)));
    revoluteJointBSF->setConstraintFlag(PxConstraintFlag::eCOLLISION_ENABLED, true);

    revoluteJointBSF->setProjectionLinearTolerance(0.1f);
    revoluteJointBSF->setConstraintFlag(PxConstraintFlag::ePROJECTION, true);

    PxJointAngularLimitPair BotSecFlapLimitPair(-PxPi*PxReal(0.000894126), PxPi*PxReal(0.611111), 0.1f);		//0,611111 Pi = 110deg
    revoluteJointBSF->setLimit(BotSecFlapLimitPair);
    revoluteJointBSF->setRevoluteJointFlag(PxRevoluteJointFlag::eLIMIT_ENABLED, true);

    revoluteJointBSF->setDriveVelocity(-1.0f);
    revoluteJointBSF->setRevoluteJointFlag(PxRevoluteJointFlag::eDRIVE_ENABLED, true);

    revoluteJointBSF->setConstraintFlag(PxConstraintFlag::eVISUALIZATION, true);			//Joint Koordinatensysteme/Limits an/aus

    /*-----------------------------------------------------------------------------------------------------------------------------------------*/
    //-----Create Side Flap Systems-----
    //-----Create Main Flaps-----	
    for (int i=0; i<=2; i++)
    {
        MainFlap[i].verts[0] = PxVec3(0.255, 0.01, 0.0015);
        MainFlap[i].verts[1] = PxVec3((-0.255), 0.01, 0.0015);
        MainFlap[i].verts[2] = PxVec3(0, 0.451673, 0.0015);
        MainFlap[i].verts[3] = PxVec3(0.255, 0.01, (-0.0015));
        MainFlap[i].verts[4] = PxVec3((-0.255), 0.01, (-0.0015));
        MainFlap[i].verts[5] = PxVec3(0, 0.451673, (-0.0015));

        MainFlap[i].convexMesh = CrawlerPlugin::plugin->createConvexMesh(MainFlap[i].verts, 6, *CrawlerPlugin::plugin->gPhysics, *CrawlerPlugin::plugin->gCooking);
        MainFlap[i].Shape = CrawlerPlugin::plugin->gPhysics->createShape(PxConvexMeshGeometry(MainFlap[i].convexMesh), &CrawlerPlugin::plugin->gMaterial, PxShapeFlag::eSIMULATION_SHAPE);
        MainFlap[i].Shape->setRestOffset(0.0);

        MainFlap[i].Actor = CrawlerPlugin::plugin->gPhysics->createRigidDynamic(PxTransform(PxVec3(0, 0, 0)));

        switch (i)
        {
        case 0:	MainFlap[0].Actor->setGlobalPose(PxTransform(PxVec3(0,0.016734,-0.172592))); break;	//SideMainFlap1 position
        case 1: MainFlap[1].Actor->setGlobalPose(PxTransform(PxVec3(0.149469,0.016734,0.086296), PxQuat(physx::PxMat33(PxVec3(-0.5,0,0.86603), PxVec3(0,1,0), PxVec3(-0.86603,0,-0.5))))); break;  //-120deg Y-Achse
        case 2: MainFlap[2].Actor->setGlobalPose(PxTransform(PxVec3(-0.149469,0.016734,0.086296), PxQuat(physx::PxMat33(PxVec3(-0.5,0,-0.86603), PxVec3(0,1,0), PxVec3(0.86603,0,-0.5))))); break; //+120deg Y-Achse
        default: break;
        }

        MainFlap[i].Actor->attachShape(*MainFlap[i].Shape);
        MainFlap[i].Actor->setMass(0.116);
        MainFlap[i].Actor->setCMassLocalPose(PxTransform(PxVec3(0, 0.157, 0)));
        MainFlap[i].Actor->setActorFlag(PxActorFlag::eVISUALIZATION, false);			//Koordinatensystem an/aus

        CrawlerPlugin::plugin->gScene->addActor(*MainFlap[i].Actor);		//Add MainFlap[i].Actor's to Scene

        MainFlap[i].revoluteJoint = PxRevoluteJointCreate(*CrawlerPlugin::plugin->gPhysics, tetraederActor, MainFlap[i].Actor->getGlobalPose(), MainFlap[i].Actor, PxTransform(PxVec3(0,0,0)));
        MainFlap[i].revoluteJoint->setConstraintFlag(PxConstraintFlag::eCOLLISION_ENABLED, true);

        MainFlap[i].revoluteJoint->setProjectionLinearTolerance(0.1f);
        MainFlap[i].revoluteJoint->setConstraintFlag(PxConstraintFlag::ePROJECTION, true);

        PxJointAngularLimitPair MainFlapLimitPair(-PxPi*PxReal(0.5), PxPi*PxReal(0.11111), 0.1f);			//0,5 Pi = 90deg  //0,11111 Pi = 20deg
        MainFlap[i].revoluteJoint->setLimit(MainFlapLimitPair);
        MainFlap[i].revoluteJoint->setRevoluteJointFlag(PxRevoluteJointFlag::eLIMIT_ENABLED, true);

        MainFlap[i].revoluteJoint->setDriveVelocity(1.0f);
        MainFlap[i].revoluteJoint->setRevoluteJointFlag(PxRevoluteJointFlag::eDRIVE_ENABLED, true);

        MainFlap[i].revoluteJoint->setConstraintFlag(PxConstraintFlag::eVISUALIZATION, true);			//Joint Koordinatensysteme/Limits an/aus

        //-----Create Secondary Flaps-----
        SecFlap[i].verts[0] = PxVec3(0.150, 0.01, 0.0015);
        SecFlap[i].verts[1] = PxVec3((-0.150), 0.01, 0.0015);
        SecFlap[i].verts[2] = PxVec3(0, 0.269808, 0.0015);
        SecFlap[i].verts[3] = PxVec3(0.150, 0.01, (-0.0015));
        SecFlap[i].verts[4] = PxVec3((-0.150), 0.01, (-0.0015));
        SecFlap[i].verts[5] = PxVec3(0, 0.269808, (-0.0015));

        SecFlap[i].convexMesh = CrawlerPlugin::plugin->createConvexMesh(SecFlap[i].verts, 6, *CrawlerPlugin::plugin->gPhysics, *CrawlerPlugin::plugin->gCooking);
        SecFlap[i].Shape = CrawlerPlugin::plugin->gPhysics->createShape(PxConvexMeshGeometry(SecFlap[i].convexMesh), &CrawlerPlugin::plugin->gMaterial, PxShapeFlag::eSIMULATION_SHAPE);
        SecFlap[i].Shape->setRestOffset(0.0);

        SecFlap[i].Actor = CrawlerPlugin::plugin->gPhysics->createRigidDynamic(PxTransform(PxVec3(0,0,0)));

        switch (i)
        {
        case 0:	SecFlap[0].Actor->setGlobalPose(PxTransform(PxVec3(-0.13616,0.26657,-0.176592), PxQuat(physx::PxMat33(PxVec3(-0.5,-0.86603,0), PxVec3(0.86603,-0.5,0), PxVec3(0,0,1))))); break;	//-120deg Z-Achse
        case 1: SecFlap[1].Actor->setGlobalPose(PxTransform(PxVec3(0.221013,0.26657,-0.029622), PxQuat(physx::PxMat33(PxVec3(0.25,-0.86603,-0.43301), PxVec3(-0.43301,-0.5,0.75), PxVec3(-0.86603,0,-0.5))))); break;	//-120deg Y * -120deg Z
        case 2: SecFlap[2].Actor->setGlobalPose(PxTransform(PxVec3(-0.084853,0.26657,0.206214), PxQuat(physx::PxMat33(PxVec3(0.25,-0.86603,0.43301), PxVec3(-0.43301,-0.5,-0.75), PxVec3(0.86603,0,-0.5))))); break;	//120deg Y * -120deg Z
        default: break;
        }

        SecFlap[i].Actor->attachShape(*SecFlap[i].Shape);
        SecFlap[i].Actor->setMass(0.023);
        SecFlap[i].Actor->setCMassLocalPose(PxTransform(PxVec3(0, 0.097, 0)));
        SecFlap[i].Actor->setActorFlag(PxActorFlag::eVISUALIZATION, false);				//Koordinatensystem an/aus

        CrawlerPlugin::plugin->gScene->addActor(*SecFlap[i].Actor);		//Add SecFlap[i].Actor's to Scene

        SecFlap[i].revoluteJoint = PxRevoluteJointCreate(*CrawlerPlugin::plugin->gPhysics, MainFlap[i].Actor, PxTransform(PxVec3(-0.13616,0.235836,-0.004), PxQuat(physx::PxMat33(PxVec3(-0.5,-0.86603,0), PxVec3(0.86603,-0.5,0), PxVec3(0,0,1)))), SecFlap[i].Actor, PxTransform(PxVec3(0,0,0)));
        SecFlap[i].revoluteJoint->setConstraintFlag(PxConstraintFlag::eCOLLISION_ENABLED, true);

        SecFlap[i].revoluteJoint->setProjectionLinearTolerance(0.1f);
        SecFlap[i].revoluteJoint->setConstraintFlag(PxConstraintFlag::ePROJECTION, true);

        PxJointAngularLimitPair SecFlapLimitPair(-PxPi*PxReal(0.611111), PxPi*PxReal(0.001179759), 0.1f);		//0,611111 Pi = 110deg
        SecFlap[i].revoluteJoint->setLimit(SecFlapLimitPair);
        SecFlap[i].revoluteJoint->setRevoluteJointFlag(PxRevoluteJointFlag::eLIMIT_ENABLED, true);

        SecFlap[i].revoluteJoint->setDriveVelocity(1.0f);
        SecFlap[i].revoluteJoint->setRevoluteJointFlag(PxRevoluteJointFlag::eDRIVE_ENABLED, true);

        SecFlap[i].revoluteJoint->setConstraintFlag(PxConstraintFlag::eVISUALIZATION, true);			//Joint Koordinatensysteme/Limits an/aus
    }

    gSelectedActorDynamic = SecFlap[2].Actor;		//Set Actor as selected Actor
}

void VrmlNodeCrawler::addToScene(VrmlScene *s, const char *relUrl)
{
    (void)relUrl;
    d_scene = s;
    if (s)
    {
    }
    else
    {
        cerr << "no Scene" << endl;
    }
}

// need copy constructor for new markerName (each instance definitely needs a new marker Name) ...

VrmlNodeCrawler::VrmlNodeCrawler(const VrmlNodeCrawler &n)
    : VrmlNodeChild(n.d_scene)
    , d_speed(n.d_speed)
    , d_lademodus(n.d_lademodus)
    , d_position(n.d_position)
    , d_rotation(n.d_rotation)
    , d_main0Angle(n.d_main0Angle)
    , d_main1Angle(n.d_main1Angle)
    , d_main2Angle(n.d_main2Angle)
    , d_sec0Angle(n.d_sec0Angle)
    , d_sec1Angle(n.d_sec1Angle)
    , d_sec2Angle(n.d_sec2Angle)
{
    setModified();
}

VrmlNodeCrawler::~VrmlNodeCrawler()
{
}

VrmlNode *VrmlNodeCrawler::cloneMe() const
{
    return new VrmlNodeCrawler(*this);
}

VrmlNodeCrawler *VrmlNodeCrawler::toCrawler() const
{
    return (VrmlNodeCrawler *)this;
}

void VrmlNodeCrawler::render(Viewer *viewer)
{
    (void)viewer;
    
    PxTransform pos = tetraederActor->getGlobalPose();
    d_position.set(pos.p[0],pos.p[1],pos.p[2]);
    PxReal angle; PxVec3 axis;
    pos.q.toRadiansAndUnitAxis(angle,axis);
    d_rotation.set(axis[0],axis[1],axis[2],angle);
    d_main0Angle.set(MainFlap[0].revoluteJoint->getAngle());
    d_main1Angle.set(MainFlap[1].revoluteJoint->getAngle());
    d_main2Angle.set(MainFlap[2].revoluteJoint->getAngle());
    d_sec0Angle.set(SecFlap[0].revoluteJoint->getAngle());
    d_sec1Angle.set(SecFlap[1].revoluteJoint->getAngle());
    d_sec2Angle.set(SecFlap[2].revoluteJoint->getAngle());
    
    double timeNow = System::the->time();
    eventOut(timeNow, "position", d_position);
    eventOut(timeNow, "rotation", d_rotation);
    eventOut(timeNow, "main0Angle", d_main0Angle);
    eventOut(timeNow, "main1Angle", d_main1Angle);
    eventOut(timeNow, "main2Angle", d_main2Angle);
    eventOut(timeNow, "sec0Angle", d_sec0Angle);
    eventOut(timeNow, "sec1Angle", d_sec1Angle);
    eventOut(timeNow, "sec2Angle", d_sec2Angle);
    MainFlap[0].revoluteJoint->getAngle();
    setModified();
}

ostream &VrmlNodeCrawler::printFields(ostream &os, int indent)
{
    if (!d_speed.get())
        PRINT_FIELD(speed);
    if (!d_lademodus.get())
        PRINT_FIELD(lademodus);
    if (!d_position.get())
        PRINT_FIELD(position);
    if (!d_rotation.get())
        PRINT_FIELD(rotation);
    if (!d_main0Angle.get())
        PRINT_FIELD(main0Angle);
    if (!d_main1Angle.get())
        PRINT_FIELD(main1Angle);
    if (!d_main2Angle.get())
        PRINT_FIELD(main2Angle);
    if (!d_sec0Angle.get())
        PRINT_FIELD(sec0Angle);
    if (!d_sec1Angle.get())
        PRINT_FIELD(sec1Angle);
    if (!d_sec2Angle.get())
        PRINT_FIELD(sec2Angle);

    return os;
}

// Set the value of one of the node fields.

void VrmlNodeCrawler::setField(const char *fieldName,
                                 const VrmlField &fieldValue)
{
    if
        TRY_FIELD(speed, SFFloat)
    else if
        TRY_FIELD(lademodus, SFBool)
    else if
        TRY_FIELD(position, SFVec3f)
    else if
        TRY_FIELD(rotation, SFRotation)
    else if
        TRY_FIELD(main0Angle, SFFloat)
    else if
        TRY_FIELD(main1Angle, SFFloat)
    else if
        TRY_FIELD(main2Angle, SFFloat)
    else if
        TRY_FIELD(sec0Angle, SFFloat)
    else if
        TRY_FIELD(sec1Angle, SFFloat)
    else if
        TRY_FIELD(sec2Angle, SFFloat)
    else
        VrmlNodeChild::setField(fieldName, fieldValue);

}

const VrmlField *VrmlNodeCrawler::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "speed") == 0)
        return &d_speed;
    else if (strcmp(fieldName, "lademodus") == 0)
        return &d_lademodus;
    else if (strcmp(fieldName, "position") == 0)
        return &d_position;
    else if (strcmp(fieldName, "rotation") == 0)
        return &d_rotation;
    else if (strcmp(fieldName, "main0Angle") == 0)
        return &d_main0Angle;
    else if (strcmp(fieldName, "main1Angle") == 0)
        return &d_main1Angle;
    else if (strcmp(fieldName, "main2Angle") == 0)
        return &d_main2Angle;
    else if (strcmp(fieldName, "sec0Angle") == 0)
        return &d_sec0Angle;
    else if (strcmp(fieldName, "sec1Angle") == 0)
        return &d_sec1Angle;
    else if (strcmp(fieldName, "sec2Angle") == 0)
        return &d_sec2Angle;
    else
        cerr << "Node does not have this eventOut or exposed field " << nodeType()->getName() << "::" << name() << "." << fieldName << endl;
    return 0;
    
}

/*-----------------------------------------------------------------------------------------------------------------------------------------*/
void VrmlNodeCrawler::OpenFlap()
{
    double speed = d_speed.get();
    switch(CrawlerPlugin::plugin->numActiveActor)
    {
    case 1: revoluteJointBMF->setDriveVelocity(speed);
        revoluteJointBMF->setRevoluteJointFlag(PxRevoluteJointFlag::eDRIVE_ENABLED, true); break;
    case 2: revoluteJointBSF->setDriveVelocity(speed);
        revoluteJointBSF->setRevoluteJointFlag(PxRevoluteJointFlag::eDRIVE_ENABLED, true); break;
    case 3: MainFlap[0].revoluteJoint->setDriveVelocity(-speed);
        MainFlap[0].revoluteJoint->setRevoluteJointFlag(PxRevoluteJointFlag::eDRIVE_ENABLED, true); break;
    case 4: SecFlap[0].revoluteJoint->setDriveVelocity(-speed);
        SecFlap[0].revoluteJoint->setRevoluteJointFlag(PxRevoluteJointFlag::eDRIVE_ENABLED, true); break;
    case 5: MainFlap[1].revoluteJoint->setDriveVelocity(-speed);
        MainFlap[1].revoluteJoint->setRevoluteJointFlag(PxRevoluteJointFlag::eDRIVE_ENABLED, true); break;
    case 6: SecFlap[1].revoluteJoint->setDriveVelocity(-speed);
        SecFlap[1].revoluteJoint->setRevoluteJointFlag(PxRevoluteJointFlag::eDRIVE_ENABLED, true); break;
    case 7: MainFlap[2].revoluteJoint->setDriveVelocity(-speed);
        MainFlap[2].revoluteJoint->setRevoluteJointFlag(PxRevoluteJointFlag::eDRIVE_ENABLED, true); break;
    case 8: SecFlap[2].revoluteJoint->setDriveVelocity(-speed);
        SecFlap[2].revoluteJoint->setRevoluteJointFlag(PxRevoluteJointFlag::eDRIVE_ENABLED, true); break;
    default: break;
    }
}

/*-----------------------------------------------------------------------------------------------------------------------------------------*/
void VrmlNodeCrawler::CloseFlap()
{
    double speed = d_speed.get();
    switch(CrawlerPlugin::plugin->numActiveActor)
    {
    case 1: revoluteJointBMF->setDriveVelocity(-speed);
        revoluteJointBMF->setRevoluteJointFlag(PxRevoluteJointFlag::eDRIVE_ENABLED, true); break;
    case 2: revoluteJointBSF->setDriveVelocity(-speed);
        revoluteJointBSF->setRevoluteJointFlag(PxRevoluteJointFlag::eDRIVE_ENABLED, true); break;
    case 3: MainFlap[0].revoluteJoint->setDriveVelocity(speed);
        MainFlap[0].revoluteJoint->setRevoluteJointFlag(PxRevoluteJointFlag::eDRIVE_ENABLED, true); break;
    case 4: SecFlap[0].revoluteJoint->setDriveVelocity(speed);
        SecFlap[0].revoluteJoint->setRevoluteJointFlag(PxRevoluteJointFlag::eDRIVE_ENABLED, true); break;
    case 5: MainFlap[1].revoluteJoint->setDriveVelocity(speed);
        MainFlap[1].revoluteJoint->setRevoluteJointFlag(PxRevoluteJointFlag::eDRIVE_ENABLED, true); break;
    case 6: SecFlap[1].revoluteJoint->setDriveVelocity(speed);
        SecFlap[1].revoluteJoint->setRevoluteJointFlag(PxRevoluteJointFlag::eDRIVE_ENABLED, true); break;
    case 7: MainFlap[2].revoluteJoint->setDriveVelocity(speed);
        MainFlap[2].revoluteJoint->setRevoluteJointFlag(PxRevoluteJointFlag::eDRIVE_ENABLED, true); break;
    case 8: SecFlap[2].revoluteJoint->setDriveVelocity(speed);
        SecFlap[2].revoluteJoint->setRevoluteJointFlag(PxRevoluteJointFlag::eDRIVE_ENABLED, true); break;
    default: break;
    }
}

/*-----------------------------------------------------------------------------------------------------------------------------------------*/
void VrmlNodeCrawler::CrawlerLademodus()
{
    for(int i=0; i<=2; i++)
    {
        if(!d_lademodus.get())
        {
            MainFlap[i].revoluteJoint->setDriveVelocity(-0.1);
            MainFlap[i].revoluteJoint->setRevoluteJointFlag(PxRevoluteJointFlag::eDRIVE_ENABLED, true);
        }
        else
            MainFlap[i].revoluteJoint->setDriveVelocity(0.1);
        MainFlap[i].revoluteJoint->setRevoluteJointFlag(PxRevoluteJointFlag::eDRIVE_ENABLED, true);
    }
}
