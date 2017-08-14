/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _ROADTERRAIN_PLUGIN_H
#define _ROADTERRAIN_PLUGIN_H
/****************************************************************************\
 **                                                            (C)2011 HLRS  **
 **                                                                          **
 ** Description: RoadTerrain Plugin                                          **
 **                                                                          **
 **                                                                          **
 ** Author: Florian Seybold		                                            **
 **                                                                          **
 ** History:  								                                         **
 ** Nov-01  v1	    				       		                                   **
 **                                                                          **
 **                                                                          **
\****************************************************************************/
#include <cover/coVRPlugin.h>
#include <cover/coTabletUI.h>
#include <osg/Texture2D>
#include <vector>
#include <osg/PositionAttitudeTransform>

#include <VehicleUtil/RoadSystem/RoadSystem.h>
#include "ogrsf_frmts.h"
#include "DecoratedGeometryTechnique.h"

#include <vrml97/vrml/config.h>
#include <vrml97/vrml/VrmlNodeType.h>
#include <vrml97/vrml/coEventQueue.h>
#include <vrml97/vrml/MathUtils.h>
#include <vrml97/vrml/System.h>
#include <vrml97/vrml/Viewer.h>
#include <vrml97/vrml/VrmlScene.h>
#include <vrml97/vrml/VrmlNamespace.h>
#include <vrml97/vrml/VrmlNode.h>
#include <vrml97/vrml/VrmlSFBool.h>
#include <vrml97/vrml/VrmlSFFloat.h>
#include <vrml97/vrml/VrmlSFVec2f.h>
#include <vrml97/vrml/VrmlMFVec2f.h>
#include <vrml97/vrml/VrmlSFString.h>
#include <vrml97/vrml/VrmlMFString.h>
#include <vrml97/vrml/VrmlSFInt.h>
#include <vrml97/vrml/VrmlMFInt.h>
#include <vrml97/vrml/VrmlNodeChild.h>
#include <vrml97/vrml/VrmlScene.h>

using namespace vrml;
using namespace opencover;

class ROADTERRAINPLUGINEXPORT VrmlNodeRoadTerrain : public VrmlNodeChild
{
public:
    // Define the fields of Sky nodes
    static VrmlNodeType *defineType(VrmlNodeType *t = 0);
    virtual VrmlNodeType *nodeType() const;

    VrmlNodeRoadTerrain(VrmlScene *scene = 0);
    VrmlNodeRoadTerrain(const VrmlNodeRoadTerrain &n);
    virtual ~VrmlNodeRoadTerrain();

    virtual VrmlNode *cloneMe() const;

    virtual VrmlNodeRoadTerrain *toRoadTerrain() const;

    virtual std::ostream &printFields(std::ostream &os, int indent);

    virtual void setField(const char *fieldName, const VrmlField &fieldValue);
    const VrmlField *getField(const char *fieldName);

    void eventIn(double timeStamp, const char *eventName,
                 const VrmlField *fieldValue);

    virtual void render(Viewer *);

private:
    // Fields

    VrmlSFString d_url;
    VrmlMFString d_layersUrl;
    VrmlSFVec3f d_offset;
    VrmlMFVec2f d_minPositions;
    VrmlMFVec2f d_maxPositions;
};

class ROADTERRAINPLUGINEXPORT RoadTerrainPlugin : public coVRPlugin, public coTUIListener
{
public:
    RoadTerrainPlugin();
    ~RoadTerrainPlugin();

    bool init();

    // key handling
    void key(int type, int keySym, int mod);

    // this will be called in PreFrame
    void preFrame();

    bool loadTerrain(std::string filename, osg::Vec3d offset,
                     const std::vector<BoundingArea> &voidBoundingAreaVector = std::vector<BoundingArea>(),
                     const std::vector<std::string> &shapeFileVector = std::vector<std::string>());
    bool addLayer(std::string filename);

    static RoadTerrainPlugin *plugin;

private:
    osg::PositionAttitudeTransform *roadGroup;
    xercesc::DOMElement *rootElement;

    coTUITab *pluginTab;
    coTUIFileBrowserButton *loadTerrainButton;
    coTUIFileBrowserButton *loadLayerButton;

    void tabletEvent(coTUIElement *tUIItem);
    void tabletPressEvent(coTUIElement *tUIItem);

    bool terrainLoaded;
    bool layerLoaded;

    std::vector<OGRLayer *> layerVector;
};
#endif
