/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

 /****************************************************************************\
  **                                                            (C)2024 HLRS  **
  **                                                                          **
  ** Description: GeoDataLoader Plugin                                        **
  **                                                                          **
  **                                                                          **
  ** Author: Uwe Woessner 		                                              **
  **                                                                          **
  ** History:  								                                  **
  **                                                                          **
  **                                                                          **
 \****************************************************************************/

#include "GeoDataLoader.h"

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


using namespace covise;
using namespace opencover;


GeoDataLoader* GeoDataLoader::s_instance = nullptr;

GeoDataLoader::GeoDataLoader(): coVRPlugin(COVER_PLUGIN_NAME)
{
    assert(s_instance == nullptr);
    s_instance = this;

}
bool GeoDataLoader::init()
{
    return loadTerrain("D:/QSync/visnas/Data/Suedlink/out/vpb_DGM1m_FDOP20/vpb_DGM1m_FDOP20.ive",osg::Vec3d(0,0,0));
}

// this is called if the plugin is removed at runtime
GeoDataLoader::~GeoDataLoader()
{
    s_instance = nullptr;
}


GeoDataLoader *GeoDataLoader::instance()
{
    if (!s_instance)
        s_instance = new GeoDataLoader;
    return s_instance;
}


bool GeoDataLoader::loadTerrain(std::string filename, osg::Vec3d offset)
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
        osgTerrain::TerrainTile::setTileLoadedCallback(new CutGeometry(this));

        osg::PositionAttitudeTransform *terrainTransform = new osg::PositionAttitudeTransform();
        terrainTransform->setName("Terrain");
        terrainTransform->addChild(terrain);
        terrainTransform->setPosition(-offset);

        cover->getObjectsRoot()->addChild(terrainTransform);

        const osg::BoundingSphere &terrainBS = terrain->getBound();
        std::cout << "Terrain BB: center: (" << terrainBS.center()[0] << ", " << terrainBS.center()[1] << ", " << terrainBS.center()[2] << "), radius: " << terrainBS.radius() << std::endl;

        return true;
    }
    else
    {
        return false;
    }
}

bool GeoDataLoader::addLayer(std::string filename)
{
#if GDAL_VERSION_MAJOR<2

#else
        if (GDALGetDriverCount() == 0)
            GDALAllRegister();

        // Try to open data source
        GDALDataset* poDS  = (GDALDataset*) GDALOpenEx( filename.c_str(), GDAL_OF_VECTOR, NULL, NULL, NULL );
#endif
        /*
    if (poDS == NULL)
    {
        std::cout << "GeoDataLoader: Could not open layer " << filename << "!" << std::endl;
        return false;
    }

    for (int layerIt = 0; layerIt < poDS->GetLayerCount(); ++layerIt)
    {
    }*/

    return true;
}

COVERPLUGIN(GeoDataLoader)
