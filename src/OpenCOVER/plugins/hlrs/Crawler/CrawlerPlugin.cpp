/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

/*********************************************************************************\
**                                                            2009 HLRS         **
**                                                                              **
** Description:  Show/Hide of CrawlerPlugins, defined in Collect Module               **
**                                                                              **
**                                                                              **
** Author: A.Gottlieb                                                           **
**                                                                              **
** Jul-09  v1                                                                   **
**                                                                              **
**                                                                              **
\*********************************************************************************/

#include "CrawlerPlugin.h"
#include <cover/coVRPluginSupport.h>
#include <cover/RenderObject.h>
#include <cover/VRSceneGraph.h>
#include <cover/coVRFileManager.h>
#include <osg/CullStack>
#include <iostream>
#include <cover/coTabletUI.h>
#include <cover/coVRPluginSupport.h>
#include <PluginUtil/PluginMessageTypes.h>
#include <net/tokenbuffer.h>
#include <osg/Node>
#include <algorithm>
#include <map>
#include <vector>
#include <iterator>
#include <numeric>
#include "VrmlNodeCrawler.h"
#include <vrml97/vrml/VrmlNamespace.h>
#include <OpenVRUI/coMenuItem.h>
#include <OpenVRUI/coMenu.h>
#include <OpenVRUI/coRowMenu.h>
//#include <OpenVRUI/coMenuChangeListener.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/coToolboxMenuItem.h>
#include <pvd/PxProfileZoneManager.h>

using namespace covise;
using namespace opencover;
using namespace vrui;

CrawlerPlugin *CrawlerPlugin::plugin = NULL;

//------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------

CrawlerPlugin::CrawlerPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    plugin = this;

    VrmlNamespace::addBuiltIn(VrmlNodeCrawler::defineType());


    gFoundation = NULL;
    gPhysics	= NULL;

    gDispatcher = NULL;
    gScene		= NULL;

    gCooking	= NULL;

    gMaterial	= NULL;

    //gConnection	= NULL;


    gGroundPlane = NULL;

}
//------------------------------------------------------------------------------------------------------------------------------

bool CrawlerPlugin::init()
{
    cover_menu = NULL;
    button = new coSubMenuItem("Crawlers");
    crawler_menu = new coRowMenu("Crawlers");
    cover->getMenu()->add(button);
    //tuTab
    CrawlerPluginTab = new coTUITab("Crawlers", coVRTui::instance()->mainFolder->getID());
    CrawlerPluginTab->setPos(0, 0);
    coTUILabel *lbl_CrawlerPlugins = new coTUILabel("Crawlers", CrawlerPluginTab->getID());
    lbl_CrawlerPlugins->setPos(0, 0);
    coTUIToggleButton *lbl_X = new coTUIToggleButton("X-trans", CrawlerPluginTab->getID());
    lbl_X->setPos(1, 0);
    lbl_X->setEventListener(this);
    coTUIToggleButton *lbl_Y = new coTUIToggleButton("Y-trans", CrawlerPluginTab->getID());
    lbl_Y->setPos(2, 0);
    lbl_Y->setEventListener(this);
    coTUIToggleButton *lbl_Z = new coTUIToggleButton("Z-trans", CrawlerPluginTab->getID());
    lbl_Z->setPos(3, 0);
    lbl_Z->setEventListener(this);
    coTUILabel *space = new coTUILabel("                     ", CrawlerPluginTab->getID());
    space->setPos(4, 0);

    initPhysics();
    return true;
}
//------------------------------------------------------------------------------------------------------------------------------
// this is called if the plugin is removed at runtime

CrawlerPlugin::~CrawlerPlugin()
{
    //fprintf ( stderr,"CrawlerPlugin::~CrawlerPlugin\n" );

    delete crawler_menu;
    delete button;
    delete CrawlerPluginTab;
    cleanupPhysics();
}
//------------------------------------------------------------------------------------------------------------------------------

void
    CrawlerPlugin::preFrame()
{
    stepPhysics();
}

void CrawlerPlugin::menuEvent(coMenuItem *item)
{

}
//------------------------------------------------------------------------------------------------------------------------------

void CrawlerPlugin::tabletEvent(coTUIElement *elem)
{
}
//------------------------------------------------------------------------------------------------------------------------------


//===============================================================================						  
//							   VRC - CrawlerSim
//
//					  Written by Christian Teichwart, 01.12.2015
//===============================================================================

// Copyright (c) 2008-2013 NVIDIA Corporation. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.  

#define PVD_HOST "127.0.0.1"

// ****************************************************************************
// It is a good idea to record and playback with pvd (PhysX Visual Debugger).
// ****************************************************************************


/*-----------------------------------------------------------------------------------------------------------------------------------------*/
//-----WRL Importer fuer Elevation-Gird-----


bool CrawlerPlugin::loadWRL(const char *path, PxU32 &xDimension, PxU32 &zDimension, PxU32 &xSpacing, PxU32 &zSpacing, PxReal &heightscale, std::vector<vector<PxReal>> &HeightMatrix)				
{
    //	printf("Loading WRL file %s...\n\n", path);
    FILE *file = fopen(path, "r");
    if(file == NULL) { printf("Impossible to open the file! Are you in the right path?\n"); return false; }

    while(1)
    {
        char lineHeader[128];	// read the first word of the line
        int res = fscanf(file, "%s", lineHeader);
        float fdummy;
        if (res == EOF) break;	// EOF = End Of File. Quit the loop.

        // else : parse lineHeader
        else if (strcmp(lineHeader, "xDimension") == 0)	{ fscanf(file, "%i", &xDimension);	/*printf("%i\n", xDimension);*/ }
        else if (strcmp(lineHeader, "zDimension") == 0)	{ fscanf(file, "%i", &zDimension);	/*printf("%i\n", zDimension);*/ }
        else if (strcmp(lineHeader, "xSpacing") == 0)	{ fscanf(file, "%i", &xSpacing);	/*printf("%i\n", xSpacing);*/	}
        else if (strcmp(lineHeader, "zSpacing") == 0)	{ fscanf(file, "%i", &zSpacing);	/*printf("%i\n", zSpacing);*/	}

        else if (strcmp(lineHeader, "scale") == 0) { double dummy = 0; fscanf(file, "%f %f", &fdummy, &heightscale); /*printf("%f\n", heightscale);*/ }

        else if (strcmp(lineHeader, "height[") == 0)
        {
            PxReal height;
            for(unsigned int i=0; i<xDimension; i++)
            {
                std::vector<PxReal> hilfsvektor;
                for (unsigned int j=0; j<zDimension; j++)
                {
                    hilfsvektor.push_back(0);
                }
                HeightMatrix.push_back(hilfsvektor);
            }
            for (unsigned int i=0; i<xDimension; i++)
            {
                for (unsigned int j=0; j<zDimension; j++)
                {
                    fscanf(file, "%f", &height); HeightMatrix[j][i] = height; //printf("%i. %f\n", i+1, height);
                }
            }
        }
    }
    return true;
}

/*-----------------------------------------------------------------------------------------------------------------------------------------*/
//OBJ-Importer fuer Kometenoberflaeche von Chury



//bool loadOBJ(const char * path,	std::vector<PxVec3> &out_vertices, int &out_numVertices, std::vector<PxU32> &out_VertexIndices, int &out_numTriangles) 
//{
//	printf("Loading OBJ file %s...\n\n", path);
//	std::vector<PxU32> vertexIndices;
//	std::vector<PxVec3> temp_vertices; 
//	int numVertices=0, numTriangles=0;
//
//	FILE * file = fopen(path, "r");
//	if( file == NULL ){ printf("Impossible to open the file ! Are you in the right path ?\n"); getchar(); return false; }
//
//	while( 1 )
//	{
//		char lineHeader[128];
//		// read the first word of the line
//		int res = fscanf(file, "%s", lineHeader);
//		if (res == EOF) break; // EOF = End Of File. Quit the loop.
//
//		// else : parse lineHeader
//		if ( strcmp( lineHeader, "v" ) == 0 )
//		{
//			PxVec3 vertex;
//			fscanf(file, "%f %f %f\n", &vertex.x, &vertex.y, &vertex.z );
//			temp_vertices.push_back(vertex);
//			numVertices++;
//		}
//		else if ( strcmp( lineHeader, "f" ) == 0 )
//		{
//			std::string vertex1, vertex2, vertex3;
//			unsigned int vertexIndex[3];
//			int matches = fscanf(file, "%d//%*d %d//%*d %d//%*d\n", 
//										&vertexIndex[0], 
//										&vertexIndex[1], 
//										&vertexIndex[2] );
//			numTriangles++;
//			if (matches != 3)
//			{
//				printf("File can't be read by our simple parser :-( Try exporting with other options\n");
//				return false;
//			}
//			vertexIndices.push_back(vertexIndex[0]);
//			vertexIndices.push_back(vertexIndex[1]);
//			vertexIndices.push_back(vertexIndex[2]);
//		}
//	}
//	printf("-----------------------\n\n");
//
//	// Put the attributes in buffers
//	out_vertices = temp_vertices;
//	out_VertexIndices = vertexIndices;
//
//	out_numVertices = numVertices;
//	out_numTriangles = numTriangles;
//		
//	return true;
//}

/*-----------------------------------------------------------------------------------------------------------------------------------------*/
void CrawlerPlugin::initPhysics()
{
    //-----Creating foundation for PhysX-----
    gFoundation	= PxCreateFoundation(PX_PHYSICS_VERSION, *gAllocator, gErrorCallback);

    profile::PxProfileZoneManager* profileZoneManager = NULL;//&PxProfileZoneManager::createProfileZoneManager(gFoundation);

    PxTolerancesScale MyTolerancesScale;
    MyTolerancesScale.length = 0.01;
    //MyTolerancesScale.mass = 1000;
    MyTolerancesScale.speed = 10;

    physx::PxPvd* mPvd=nullptr;

    //-----Creating instance of PhysX SDK-----
    gPhysics = PxCreatePhysics(PX_PHYSICS_VERSION, *gFoundation, PxTolerancesScale(MyTolerancesScale), true, mPvd);

    if(gPhysics == NULL) { cerr<<"Error creating PhysX3 device, Exiting..."<<endl; exit(1); }

  /*  if (gPhysics->getPvdConnectionManager())
    {
        gPhysics->getVisualDebugger()->setVisualizeConstraints(true);
        gPhysics->getVisualDebugger()->setVisualDebuggerFlag(PxVisualDebuggerFlag::eTRANSMIT_CONTACTS, true);
        gPhysics->getVisualDebugger()->setVisualDebuggerFlag(PxVisualDebuggerFlag::eTRANSMIT_SCENEQUERIES, true);	
        //gConnection = PxVisualDebuggerExt::createConnection(gPhysics->getPvdConnectionManager(), PVD_HOST, 5425, 10);
    }*/

    //-----Creating Scene-----
    PxSceneDesc sceneDesc(gPhysics->getTolerancesScale());

    //	sceneDesc.gravity		= PxVec3(0.0f, -9.81f, 0.0f);			//Setting Earth gravity
    sceneDesc.gravity		= PxVec3(0.0f, -0.0445f, 0.0f);			//Setting Asteroid gravity
    gDispatcher				= PxDefaultCpuDispatcherCreate(1);
    sceneDesc.cpuDispatcher	= gDispatcher;
    sceneDesc.filterShader	= PxDefaultSimulationFilterShader;

    gScene = gPhysics->createScene(sceneDesc);						//Creating Scene

    //This will enable basic visualization of PhysX objects like- actors collision shapes and their axes. 
    //gScene->setVisualizationParameter(PxVisualizationParameter::eSCALE,				1.0);	//Global visualization scale which gets multiplied with the individual scales
    //gScene->setVisualizationParameter(PxVisualizationParameter::eWORLD_AXES,		2.0f);
    //gScene->setVisualizationParameter(PxVisualizationParameter::eCOLLISION_SHAPES,	1.0f);	//Enable visualization of actor's shape
    //gScene->setVisualizationParameter(PxVisualizationParameter::eACTOR_AXES,		1.0f);	//Enable visualization of actor's axis

    //-----Creating PhysX material (staticFriction, dynamicFriction, restitution)-----
    gMaterial = gPhysics->createMaterial(0.45f, 0.2f, 0.1f);		//restitution = 0 (vollkommen plastischer Sto�)		
    //restitution = 1 (vollkommen elastischer Sto�)
    //-----Create Cooking-----
    gCooking = PxCreateCooking(PX_PHYSICS_VERSION, *gFoundation, PxCookingParams(MyTolerancesScale));

    //-----Create Ground Plane-----
    //	PxRigidStatic* groundPlane = PxCreatePlane(*gPhysics, PxPlane(0,1,0,0), *gMaterial);
    //	gScene->addActor(*groundPlane);

    /*-----------------------------------------------------------------------------------------------------------------------------------------*/
    //-----WRL einlesen und davon ein Heightfield erstellen-----


    PxU32 xDimension, zDimension, xSpacing, zSpacing;
    std::vector<vector<PxReal>> HeightMatrix;
    PxReal heightscale;
    const char *fn = coVRFileManager::instance()->getName("src/OpenCOVER/plugins/hlrs/Crawler/Crawler_VRML/ElevationGrid_10x10.wrl");
    if(fn)
    {
        bool res = loadWRL(fn, xDimension, zDimension, xSpacing, zSpacing, heightscale, HeightMatrix); 
    }
    //if(res == false) { res = loadWRL("../../CrawlerSim/Crawler_VRML/ElevationGrid_10x10.wrl", xDimension, zDimension, xSpacing, zSpacing, heightscale, HeightMatrix); }

    //-----Create Heightfield-----
    PxHeightFieldSample* samples = (PxHeightFieldSample*)malloc(sizeof(PxHeightFieldSample)*(xDimension * zDimension));

    for(unsigned int j=0; j<xDimension; j++)
    {
        for(unsigned int k=0; k<zDimension; k++)
        {
            samples[k+j*xDimension].height = HeightMatrix[j][k]; 
            samples[k+j*xDimension].clearTessFlag();
        }
    }

    PxHeightFieldDesc hfDesc;
    hfDesc.format             = PxHeightFieldFormat::eS16_TM;
    hfDesc.nbColumns          = xDimension;
    hfDesc.nbRows             = zDimension;
    //hfDesc.thickness          = -1.0f;
    hfDesc.samples.data       = samples;
    hfDesc.samples.stride     = sizeof(PxHeightFieldSample);

    PxHeightField* aHeightField = gCooking->createHeightField(hfDesc,*this);

    PxHeightFieldGeometry hfGeom(aHeightField, PxMeshGeometryFlags(), heightscale, xSpacing, zSpacing);		//heightScale, rowScale, colScale);

    PxVec3 HeightfieldOffset = PxVec3(xDimension/2 - 0.5, 0.0, zDimension/2 - 0.5);
    PxRigidStatic* aHeightFieldActor = gPhysics->createRigidStatic(PxTransform(-HeightfieldOffset));

#if (PX_PHYSICS_VERSION_MAJOR > 3)
    PxShape * aHeightFieldShape = PxRigidActorExt::createExclusiveShape(*aHeightFieldActor,hfGeom, *gMaterial);
    aHeightFieldActor->attachShape(*aHeightFieldShape);
#else
    PxShape* aHeightFieldShape = aHeightFieldActor->createShape(hfGeom, *gMaterial);
#endif

    aHeightFieldActor->setActorFlag(PxActorFlag::eVISUALIZATION, false);					//Koordinatensystem an/aus

    gScene->addActor(*aHeightFieldActor);		//Add Heightfield to Scene

    /*-----------------------------------------------------------------------------------------------------------------------------------------*/
    // Erstellen des Triangle-meshes fuer Kometenoberflaeche von Chury



    //char *filename = "../../Chury_Teilflaeche.obj";
    //char *filenameCooked = "../../temp.bin";

    //std::vector<PxVec3> vertices;
    //std::vector<PxU32> vertexIndices;
    //int numVertices, numTriangles;

    //bool res = loadOBJ(filename, vertices, numVertices, vertexIndices, numTriangles); 
    //	
    //PxVec3 *verticesChury = new PxVec3[numVertices];  //Verts
    //PxU32 *verticesIndices = new PxU32[numTriangles*3]; //Anzahl Triangles *3

    //for(int i=0; i<numVertices; i++)  
    //{
    //	verticesChury[i] = vertices[i];
    //}
    //
    //for(int i=0; i<(numTriangles*3); i++)
    //{
    //	verticesIndices[i] = vertexIndices[i]-1;
    //}

    //PxTriangleMesh *mesh = NULL;
    //
    //PxTriangleMeshDesc meshDesc;
    //meshDesc.points.count				= numVertices;
    //meshDesc.triangles.count			= numTriangles;
    //meshDesc.points.stride				= sizeof(PxVec3); 
    //meshDesc.triangles.stride			= sizeof(PxU32) * 3;
    //meshDesc.points.data				= verticesChury;
    //meshDesc.triangles.data				= verticesIndices;	

    //bool ok;
    //{
    //	PxDefaultFileOutputStream stream(filenameCooked);
    //	ok = gCooking->cookTriangleMesh(meshDesc, stream);
    //}
    //if ( ok )
    //{
    //	PxDefaultFileInputData stream(filenameCooked);
    //	mesh = gPhysics->createTriangleMesh(stream);
    //}

    //PxReal scalef = 10.0f;
    //PxVec3 scale = PxVec3(scalef,scalef,scalef);
    //PxMeshScale meshScale = PxMeshScale(scale, PxQuat(PxIdentity));
    //PxShape* churyShape = gPhysics->createShape(PxTriangleMeshGeometry(mesh, meshScale), &gMaterial, PxShapeFlag::eSIMULATION_SHAPE);	
    //PxRigidStatic *churyActor = gPhysics->createRigidStatic(PxTransform(PxVec3(0.0f, -3.5f, 0.0f)));
    //churyActor->attachShape(*churyShape);
    //gScene->addActor(*churyActor);

}


PxConvexMesh* CrawlerPlugin::createConvexMesh(const PxVec3* verts, const PxU32 numVerts, PxPhysics& physics, PxCooking& cooking)
{
	// Create descriptor for convex mesh
	PxConvexMeshDesc convexDesc;
	convexDesc.points.count			= numVerts;
	convexDesc.points.stride		= sizeof(PxVec3);
	convexDesc.points.data			= verts;
	convexDesc.flags				= PxConvexFlag::eCOMPUTE_CONVEX 
#if !(PX_PHYSICS_VERSION_MAJOR > 3)
        | PxConvexFlag::eINFLATE_CONVEX
#endif
        ;

	PxConvexMesh* convexMesh = NULL;
	PxDefaultMemoryOutputStream buf;
	if(cooking.cookConvexMesh(convexDesc, buf))
	{
		PxDefaultMemoryInputData id(buf.getData(), buf.getSize());
		convexMesh = physics.createConvexMesh(id);
	}
	return convexMesh;
}

/*-----------------------------------------------------------------------------------------------------------------------------------------*/
void CrawlerPlugin::stepPhysics()
{
    const PxF32 timestep = cover->frameDuration();

    //Scene update.
    gScene->simulate(timestep);
    gScene->fetchResults(true);
}

void CrawlerPlugin::cleanupPhysics()
{
    gCooking->release();
    gScene->release();
    gDispatcher->release();
    #if !(PX_PHYSICS_VERSION_MAJOR > 3)
    PxProfileZoneManager* profileZoneManager = gPhysics->getProfileZoneManager();
    if(gConnection != NULL)
        gConnection->release();
    profileZoneManager->release();
#endif
    gPhysics->release();
    gFoundation->release();
}


//------------------------------------------------------------------------------------------------------------------------------
COVERPLUGIN(CrawlerPlugin)
