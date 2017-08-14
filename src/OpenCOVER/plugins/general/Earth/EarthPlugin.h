/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _EARTH_PLUGIN_H
#define _EARTH_PLUGIN_H
/****************************************************************************\ 
 **                                                            (C)2001 HLRS  **
 **                                                                          **
 ** Description: Earth Plugin (does nothing)                              **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner		                                                **
 **                                                                          **
 ** History:  								                                **
 ** Nov-01  v1	    				       		                            **
 **                                                                          **
 **                                                                          **
\****************************************************************************/
#include <cover/coVRPlugin.h>
#include <cover/coVRFileManager.h>

#include <osgEarth/MapNode>
#include <osgEarth/XmlUtils>
#include <osgEarth/Version>
#include <osgEarth/Viewpoint>

#include <osgEarthSymbology/Color>

#include <osgEarthAnnotation/AnnotationData>
//#include <osgEarthAnnotation/Decluttering>

#include <osgEarthDrivers/kml/KML>

#include <osgEarthUtil/EarthManipulator>
#include <osgEarthUtil/AutoClipPlaneHandler>
#include <osgEarthUtil/Controls>
#if OSGEARTH_VERSION_LESS_THAN(2,6,0)
#include <osgEarthUtil/SkyNode>
#else
#include <osgEarthUtil/Sky>
#endif
#include <osgEarthUtil/Formatter>
#include <osgEarthUtil/MouseCoordsTool>
#include "EarthLayerManager.h"
#include "EarthViewpointManager.h"
#include <cover/coTabletUI.h>

using namespace opencover;

using namespace osgEarth::Util;
using namespace osgEarth::Util::Controls;
using namespace osgEarth::Symbology;
using namespace osgEarth::Drivers;
using namespace osgEarth::Annotation;

class EarthPlugin : public coVRPlugin, public coTUIListener
{
public:
    EarthPlugin();
    ~EarthPlugin();

    int loadFile(const char *name, osg::Group *parent);
    int unloadFile(const char *name);
    int loadKmlFile(const char *name, osg::Group *parent);
    int unloadKmlFile(const char *name);
    float getRPM()
    {
        return viewpointManager->getRPM();
    };

    static int sLoadFile(const char *name, osg::Group *parent, const char *covise_key);
    static int sUnloadFile(const char *name, const char *covise_key);
    static int sLoadKmlFile(const char *name, osg::Group *parent, const char *covise_key);
    static int sUnloadKmlFile(const char *name, const char *covise_key);

    bool init();
    // this will be called in PreFrame
    void preFrame();

    const osgEarth::SpatialReference *getSRS() const;

    osgEarth::MapNode *getMapNode()
    {
        return mapNode;
    };
    void toggle(osg::Group *g, const std::string &name, bool onoff);
    void tabletEvent(coTUIElement *e);

    static EarthPlugin *plugin;

private:
    osg::Node *loadedModel;
    osgEarth::MapNode *mapNode;
    bool useSky;
    SkyNode *s_sky;
    bool useOcean;
#if 0
	OceanSurfaceNode* s_ocean;
#endif
    bool useAutoClip;
    EarthLayerManager *layerManager;
    EarthViewpointManager *viewpointManager;
    osg::Node *oldDump;
    coTUITab *earthTab;

    coTUIButton *requestDump;
    coTUIToggleButton *DebugCheckBox;
    coTUIToggleButton *cameraCheckBox;
    coTUIToggleButton *intersectionsCheckBox;
    coTUIToggleButton *rttCheckBox;
};
#endif
