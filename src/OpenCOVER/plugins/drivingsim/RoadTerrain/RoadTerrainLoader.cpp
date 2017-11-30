/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                            (C)2011 HLRS  **
 **                                                                          **
 ** Description: RoadTerrain Plugin                                          **
 **                                                                          **
 **                                                                          **
 ** Author: Florian Seybold            		                                **
 **                                                                          **
 ** History:  								                                         **
 ** Nov-01  v1	    				       		                                   **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include "RoadTerrainLoader.h"

#include <cover/coVRPluginSupport.h>
#include <cover/RenderObject.h>
#include <cover/coVRShader.h>
#include <cover/coVRFileManager.h>
#include <cover/coVRTui.h>
#include <cover/coVRMSController.h>
#include <cover/coVRConfig.h>
#include <cover/VRViewer.h>

#include <config/CoviseConfig.h>

#include <osgDB/ReadFile>
#include <osgDB/WriteFile>
#include <osg/Node>
#include <osg/Group>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Texture2D>
#include <osg/StateSet>
#include <osg/Material>
#include <osg/LOD>
#include <osg/PolygonOffset>
#include <osgUtil/Optimizer>
#include <osgTerrain/Terrain>
#include <osgViewer/Renderer>

#include <xercesc/parsers/XercesDOMParser.hpp>

#include "LevelTerrainCallback.h"

using namespace covise;
using namespace opencover;


RoadTerrainLoader* RoadTerrainLoader::s_instance = nullptr;

RoadTerrainLoader::RoadTerrainLoader()
: terrainLoaded(false)
, layerLoaded(false)
{
    assert(s_instance == nullptr);
    s_instance = this;

    OGRRegisterAll();
}

// this is called if the plugin is removed at runtime
RoadTerrainLoader::~RoadTerrainLoader()
{
    s_instance = nullptr;
}


RoadTerrainLoader *RoadTerrainLoader::instance()
{
    if (!s_instance)
        s_instance = new RoadTerrainLoader;
    return s_instance;
}


bool RoadTerrainLoader::loadTerrain(std::string filename, osg::Vec3d offset,
                                    const std::vector<BoundingArea> &voidBoundingAreaVector,
                                    const std::vector<std::string> &shapeFileNameVector)
{
    VRViewer *viewer = VRViewer::instance();
    osgDB::DatabasePager *pager = viewer->getDatabasePager();
    std::cout << "DatabasePager:" << std::endl;
    std::cout << "\t num threads: " << pager->getNumDatabaseThreads() << std::endl;
    std::cout << "\t getDatabasePagerThreadPause: " << (pager->getDatabasePagerThreadPause() ? "yes" : "no") << std::endl;
    std::cout << "\t getDoPreCompile: " << (pager->getDoPreCompile() ? "yes" : "no") << std::endl;
    std::cout << "\t getApplyPBOToImages: " << (pager->getApplyPBOToImages() ? "yes" : "no") << std::endl;
    std::cout << "\t getTargetMaximumNumberOfPageLOD: " << pager->getTargetMaximumNumberOfPageLOD() << std::endl;
    viewer->setIncrementalCompileOperation(new osgUtil::IncrementalCompileOperation());
    //pager->setIncrementalCompileOperation(new osgUtil::IncrementalCompileOperation());
    osgUtil::IncrementalCompileOperation *coop = pager->getIncrementalCompileOperation();
    if (coop)
    {
        for (int screenIt = 0; screenIt < coVRConfig::instance()->numScreens(); ++screenIt)
        {
            osgViewer::Renderer *renderer = dynamic_cast<osgViewer::Renderer *>(coVRConfig::instance()->channels[screenIt].camera->getRenderer());
            if (renderer)
            {
                renderer->getSceneView(0)->setAutomaticFlush(false);
                renderer->getSceneView(1)->setAutomaticFlush(false);
            }
        }

        coop->setTargetFrameRate(60);
        coop->setMinimumTimeAvailableForGLCompileAndDeletePerFrame(0.001);
        coop->setMaximumNumOfObjectsToCompilePerFrame(2);
        coop->setConservativeTimeRatio(0.1);
        coop->setFlushTimeRatio(0.1);

        std::cout << "IncrementedCompileOperation:" << std::endl;
        std::cout << "\t is active: " << (coop->isActive() ? "yes" : "no") << std::endl;
        std::cout << "\t target frame rate: " << coop->getTargetFrameRate() << std::endl;
        std::cout << "\t getMinimumTimeAvailableForGLCompileAndDeletePerFrame: " << coop->getMinimumTimeAvailableForGLCompileAndDeletePerFrame() << std::endl;
        std::cout << "\t getMaximumNumOfObjectsToCompilePerFrame: " << coop->getMaximumNumOfObjectsToCompilePerFrame() << std::endl;
        std::cout << "\t flush time ratio: " << coop->getFlushTimeRatio() << std::endl;
        std::cout << "\t conservative time ratio: " << coop->getConservativeTimeRatio() << std::endl;
    }

    std::vector<osg::ref_ptr<osg::Geode> > shapeFileVector;
    for (int shapeIt = 0; shapeIt < shapeFileNameVector.size(); ++shapeIt)
    {
        osg::Node *shapeNode = osgDB::readNodeFile(shapeFileNameVector[shapeIt]);
        shapeNode->ref();
        osg::Geode *shapeGeode = dynamic_cast<osg::Geode *>(shapeNode);
        if (shapeGeode)
        {
            shapeFileVector.push_back(shapeGeode);
        }
    }

    if (terrainLoaded)
        return false;
    osg::Node *terrain = osgDB::readNodeFile(filename);

    if (terrain)
    {
        /*osgTerrain::Terrain* terrainNode = dynamic_cast<osgTerrain::Terrain*>(terrain);
      if(terrainNode) {
         std::cout << "Casted to terrainNode!" << std::endl;
         std::cout << "Num children: " << terrainNode->getNumChildren() << std::endl;
         if(terrainNode->getNumChildren()>0) {
            terrain = terrainNode->getChild(0);
         }
      }*/
        terrain->setDataVariance(osg::Object::DYNAMIC);

        osg::StateSet *terrainStateSet = terrain->getOrCreateStateSet();

        terrainStateSet->setMode(GL_LIGHTING, osg::StateAttribute::ON);
        terrainStateSet->setMode(GL_LIGHT0, osg::StateAttribute::ON);
        //terrainStateSet->setMode ( GL_LIGHT1, osg::StateAttribute::ON);

        /*osg::StateSet* terrainStateSet = terrain->getOrCreateStateSet();
      osg::PolygonOffset* offset = new osg::PolygonOffset(1.0, 1.0);
      osg::PolygonOffset* offset = new osg::PolygonOffset(0.0, 0.0);
      terrainStateSet->setAttributeAndModes(offset, osg::StateAttribute::OVERRIDE|osg::StateAttribute::ON
      */
        //osgUtil::Optimizer optimizer;
        //optimizer.optimize(terrain);

        terrain->setNodeMask(terrain->getNodeMask() & (~Isect::Intersection) & (~Isect::Pick));

        //std::vector<BoundingArea> voidBoundingAreaVector;
        //voidBoundingAreaVector.push_back(BoundingArea(osg::Vec2(506426.839,5398055.357),osg::Vec2(508461.865,5399852.0)));
        osgTerrain::TerrainTile::setTileLoadedCallback(new LevelTerrainCallback(-offset, voidBoundingAreaVector, shapeFileVector));

        osg::PositionAttitudeTransform *terrainTransform = new osg::PositionAttitudeTransform();
        terrainTransform->setName("Terrain");
        terrainTransform->addChild(terrain);
        terrainTransform->setPosition(-offset);

        cover->getObjectsRoot()->addChild(terrainTransform);

        const osg::BoundingSphere &terrainBS = terrain->getBound();
        std::cout << "Terrain BB: center: (" << terrainBS.center()[0] << ", " << terrainBS.center()[1] << ", " << terrainBS.center()[2] << "), radius: " << terrainBS.radius() << std::endl;

        terrainLoaded = true;
        return true;
    }
    else
    {
        return false;
    }
}

bool RoadTerrainLoader::addLayer(std::string filename)
{
#if GDAL_VERSION_MAJOR<2
        if (OGRSFDriverRegistrar::GetRegistrar()->GetDriverCount() == 0)
            OGRRegisterAll();

        // Try to open data source
        OGRDataSource* poDS = OGRSFDriverRegistrar::Open(filename.c_str(), FALSE);
#else
        if (GDALGetDriverCount() == 0)
            GDALAllRegister();

        // Try to open data source
        GDALDataset* poDS  = (GDALDataset*) GDALOpenEx( filename.c_str(), GDAL_OF_VECTOR, NULL, NULL, NULL );
#endif

    if (poDS == NULL)
    {
        std::cout << "RoadTerrainLoader: Could not open layer " << filename << "!" << std::endl;
        return false;
    }

    for (int layerIt = 0; layerIt < poDS->GetLayerCount(); ++layerIt)
    {
        OGRLayer *poLayer = poDS->GetLayer(layerIt);
        layerVector.push_back(poLayer);
    }

    return true;
}
