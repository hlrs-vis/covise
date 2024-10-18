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

using namespace covise;
using namespace opencover;

void VrmlNodeRoadTerrain::initFields(VrmlNodeRoadTerrain *node, VrmlNodeType *t)
{
    VrmlNodeChild::initFields(node, t);
    initFieldsHelper(node, t,
                     exposedField("url", node->d_url),
                     exposedField("layersUrl", node->d_layersUrl),
                     exposedField("offset", node->d_offset),
                     exposedField("minPositions", node->d_minPositions),
                     exposedField("maxPositions", node->d_maxPositions));
}

const char *VrmlNodeRoadTerrain::name() { return "RoadTerrain"; }

VrmlNodeRoadTerrain::VrmlNodeRoadTerrain(VrmlScene *scene)
    : VrmlNodeChild(scene, name())
{
    setModified();
}

VrmlNodeRoadTerrain::VrmlNodeRoadTerrain(const VrmlNodeRoadTerrain &n)
    : VrmlNodeChild(n)
{
    setModified();
}

VrmlNodeRoadTerrain *VrmlNodeRoadTerrain::toRoadTerrain() const
{
    return (VrmlNodeRoadTerrain *)this;
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
: coVRPlugin(COVER_PLUGIN_NAME)
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

    VrmlNamespace::addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeRoadTerrain>());

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

bool RoadTerrainPlugin::loadTerrain(std::string filename, osg::Vec3d offset,
                                    const std::vector<BoundingArea> &voidBoundingAreaVector,
                                    const std::vector<std::string> &shapeFileNameVector)
{
    return RoadTerrainLoader::instance()->loadTerrain(filename, offset, voidBoundingAreaVector, shapeFileNameVector);
}

bool RoadTerrainPlugin::addLayer(std::string filename)
{
    return RoadTerrainLoader::instance()->addLayer(filename);
}

void RoadTerrainPlugin::tabletEvent(coTUIElement *tUIItem)
{
    if (tUIItem == loadTerrainButton)
    {
        osg::Vec3d offset(0, 0, 0);
        const vehicleUtil::RoadSystemHeader &header = vehicleUtil::RoadSystem::Instance()->getHeader();
        offset.set(header.xoffset, header.yoffset, 0.0);
        std::cout << "Header: xoffset: " << header.xoffset << ", yoffset: " << header.yoffset << std::endl;

        loadTerrain(loadTerrainButton->getFilename(loadTerrainButton->getSelectedPath()), offset);
    }

    else if (tUIItem == loadLayerButton)
    {
        addLayer(loadLayerButton->getFilename(loadLayerButton->getSelectedPath()));
    }
}

COVERPLUGIN(RoadTerrainPlugin)
