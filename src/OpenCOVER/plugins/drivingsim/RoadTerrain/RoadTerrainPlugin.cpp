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

#include "RoadTerrainPlugin.h"

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

static VrmlNode *creator(VrmlScene *scene)
{
    return new VrmlNodeRoadTerrain(scene);
}

// Define the built in VrmlNodeType:: "Sky" fields

VrmlNodeType *VrmlNodeRoadTerrain::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st; // Only define the type once.
        t = st = new VrmlNodeType("RoadTerrain", creator);
    }

    VrmlNodeChild::defineType(t); // Parent class
    t->addExposedField("url", VrmlField::SFSTRING);
    t->addExposedField("layersUrl", VrmlField::MFSTRING);
    t->addExposedField("offset", VrmlField::SFVEC3F);
    t->addExposedField("minPositions", VrmlField::MFVEC2F);
    t->addExposedField("maxPositions", VrmlField::MFVEC2F);

    return t;
}

VrmlNodeType *VrmlNodeRoadTerrain::nodeType() const
{
    return defineType(0);
}

VrmlNodeRoadTerrain::VrmlNodeRoadTerrain(VrmlScene *scene)
    : VrmlNodeChild(scene)
{
    setModified();
}

VrmlNodeRoadTerrain::VrmlNodeRoadTerrain(const VrmlNodeRoadTerrain &n)
    : VrmlNodeChild(n.d_scene)
{

    setModified();
}

VrmlNodeRoadTerrain::~VrmlNodeRoadTerrain()
{
}

VrmlNode *VrmlNodeRoadTerrain::cloneMe() const
{
    return new VrmlNodeRoadTerrain(*this);
}

VrmlNodeRoadTerrain *VrmlNodeRoadTerrain::toRoadTerrain() const
{
    return (VrmlNodeRoadTerrain *)this;
}

ostream &VrmlNodeRoadTerrain::printFields(ostream &os, int indent)
{

    return os;
}

// Set the value of one of the node fields.

void VrmlNodeRoadTerrain::setField(const char *fieldName,
                                   const VrmlField &fieldValue)
{

    if
        TRY_FIELD(offset, SFVec3f)
    else if
        TRY_FIELD(minPositions, MFVec2f)
    else if
        TRY_FIELD(maxPositions, MFVec2f)
    else if
        TRY_FIELD(url, SFString)
    else if
        TRY_FIELD(layersUrl, MFString)
    else
        VrmlNodeChild::setField(fieldName, fieldValue);
}

const VrmlField *VrmlNodeRoadTerrain::getField(const char *fieldName)
{

    if (strcmp(fieldName, "offset") == 0)
        return &d_offset;
    else if (strcmp(fieldName, "url") == 0)
        return &d_url;
    else if (strcmp(fieldName, "layersUrl") == 0)
        return &d_layersUrl;
    else if (strcmp(fieldName, "minPositions") == 0)
        return &d_minPositions;
    else if (strcmp(fieldName, "maxPositions") == 0)
        return &d_maxPositions;
    else
        cerr << "Node does not have this eventOut or exposed field " << nodeType()->getName() << "::" << name() << "." << fieldName << endl;
    return 0;
}

void VrmlNodeRoadTerrain::eventIn(double timeStamp,
                                  const char *eventName,
                                  const VrmlField *fieldValue)
{

    // Check exposedField
    {
        VrmlNode::eventIn(timeStamp, eventName, fieldValue);
    }

    setModified();
}

void VrmlNodeRoadTerrain::render(Viewer *)
{
    osg::Vec3d offset;
    offset.set(d_offset.x(), d_offset.y(), d_offset.z());
    if (RoadTerrainPlugin::plugin->loadTerrain(d_url.get(), offset))
    {
        for (int i = 0; i < d_layersUrl.size(); i++)
            RoadTerrainPlugin::plugin->addLayer(d_layersUrl.get(i));
    }
    clearModified();
}

RoadTerrainPlugin::RoadTerrainPlugin()
    : terrainLoaded(false)
    , layerLoaded(false)
{
    plugin = this;

    OGRRegisterAll();
}

RoadTerrainPlugin *RoadTerrainPlugin::plugin = NULL;

// this is called if the plugin is removed at runtime
RoadTerrainPlugin::~RoadTerrainPlugin()
{
}

bool RoadTerrainPlugin::init()
{
    //cover->setScale(1000);

    VrmlNamespace::addBuiltIn(VrmlNodeRoadTerrain::defineType());

    pluginTab = new coTUITab("Road Terrain", coVRTui::instance()->mainFolder->getID());
    pluginTab->setPos(0, 0);
    loadTerrainButton = new coTUIFileBrowserButton("Load VPB terrain...", pluginTab->getID());
    loadTerrainButton->setEventListener(this);
    loadTerrainButton->setPos(0, 0);
    loadTerrainButton->setMode(coTUIFileBrowserButton::OPEN);
    loadTerrainButton->setFilterList("*.ive");

    loadLayerButton = new coTUIFileBrowserButton("Load layer...", pluginTab->getID());
    loadLayerButton->setEventListener(this);
    loadLayerButton->setPos(0, 1);
    loadLayerButton->setMode(coTUIFileBrowserButton::OPEN);
    loadLayerButton->setFilterList("*.shp");

    return true;
}

// here we get the size and the current center of the cube
void
RoadTerrainPlugin::newInteractor(RenderObject *, coInteractor *i)
{
    (void)i;
    fprintf(stderr, "RoadTerrainPlugin::newInteractor\n");
}

void RoadTerrainPlugin::addObject(RenderObject *container,
                                  RenderObject *obj, RenderObject *normObj,
                                  RenderObject *colorObj, RenderObject *texObj,
                                  osg::Group *parent,
                                  int numCol, int colorBinding, int colorPacking,
                                  float *r, float *g, float *b, int *packedCol,
                                  int numNormals, int normalBinding,
                                  float *xn, float *yn, float *zn, float transparency)
{
    (void)container;
    (void)obj;
    (void)normObj;
    (void)colorObj;
    (void)texObj;
    (void)parent;
    (void)numCol;
    (void)colorBinding;
    (void)colorPacking;
    (void)r;
    (void)g;
    (void)b;
    (void)packedCol;
    (void)numNormals;
    (void)normalBinding;
    (void)xn;
    (void)yn;
    (void)zn;
    (void)transparency;
    fprintf(stderr, "RoadTerrainPlugin::addObject\n");
}

void
RoadTerrainPlugin::removeObject(const char *objName, bool replace)
{
    (void)objName;
    (void)replace;
    fprintf(stderr, "RoadTerrainPlugin::removeObject\n");
}

void
RoadTerrainPlugin::preFrame()
{
    //double dt = cover->frameDuration();
}

bool RoadTerrainPlugin::loadTerrain(std::string filename, osg::Vec3d offset,
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

bool RoadTerrainPlugin::addLayer(std::string filename)
{
    OGRDataSource *poDS = OGRSFDriverRegistrar::Open(filename.c_str(), FALSE);
    if (poDS == NULL)
    {
        std::cout << "RoadTerrainPlugin: Could not open layer " << filename << "!" << std::endl;
        return false;
    }

    for (int layerIt = 0; layerIt < poDS->GetLayerCount(); ++layerIt)
    {
        OGRLayer *poLayer = poDS->GetLayer(layerIt);
        layerVector.push_back(poLayer);
    }

    return true;
}

void RoadTerrainPlugin::tabletEvent(coTUIElement *tUIItem)
{
    if (tUIItem == loadTerrainButton)
    {
        osg::Vec3d offset(0, 0, 0);
        const RoadSystemHeader &header = RoadSystem::Instance()->getHeader();
        offset.set(header.xoffset, header.yoffset, 0.0);
        std::cout << "Header: xoffset: " << header.xoffset << ", yoffset: " << header.yoffset << std::endl;

        loadTerrain(loadTerrainButton->getFilename(loadTerrainButton->getSelectedPath()), offset);
    }

    else if (tUIItem == loadLayerButton)
    {
        //if(layerLoaded) return; // why?

        addLayer(loadLayerButton->getFilename(loadLayerButton->getSelectedPath()));

        layerLoaded = true;
    }
}

void RoadTerrainPlugin::tabletPressEvent(coTUIElement *tUIItem)
{
}

void RoadTerrainPlugin::key(int type, int keySym, int mod)
{
}

COVERPLUGIN(RoadTerrainPlugin)
