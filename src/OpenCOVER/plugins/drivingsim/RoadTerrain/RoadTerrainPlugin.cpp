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
