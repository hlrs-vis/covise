/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

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
#include <osgEarth/Map>
#include <osgEarth/MapNode>
#include <osgEarth/TMS>
#include <osgEarth/GDAL>
#include <osgEarth/Registry>
#include <osgEarth/OverlayDecorator>
#include <osgEarth/Common>
#include <osgEarth/CachePolicy>
#include <osgEarth/Units>
#include <osgEarth/Capabilities>
#include <osg/Depth>
#include <osg/LineWidth>

using namespace osgEarth;

#include "EarthPlugin.h"
#include <cover/coVRPluginSupport.h>
#include <cover/RenderObject.h>
#include <cover/VRViewer.h>
#include <cover/coVRTui.h>
#include <osgDB/ReadFile>
#include <osgEarthDrivers/kml/KML>
#include <osgEarthDrivers/kml/KMLOptions>
//viewpoints are in a plugin now have to access theem through the plugin interface #include <osgEarthDrivers/viewpoints/ViewpointsExtension>
#include <osgEarth/GLUtils>

EarthPlugin *EarthPlugin::plugin = NULL;

FileHandler fileHandler[] = {
    { NULL,
      EarthPlugin::sLoadFile,
      EarthPlugin::sUnloadFile,
      "earth" },
    { NULL,
      EarthPlugin::sLoadKmlFile,
      EarthPlugin::sUnloadKmlFile,
      "kml" }
};

int EarthPlugin::sLoadFile(const char *fn, osg::Group *parent, const char *)
{
    if (plugin)
        return plugin->loadFile(fn, parent);

    return -1;
}

int EarthPlugin::sUnloadFile(const char *fn, const char *)
{
    if (plugin)
    {
        plugin->unloadFile(fn);
        return 0;
    }

    return -1;
}

int EarthPlugin::sLoadKmlFile(const char *fn, osg::Group *parent, const char *)
{
    if (plugin)
        return plugin->loadKmlFile(fn, parent);
        
    return -1;
}

int EarthPlugin::sUnloadKmlFile(const char *fn, const char *)
{
    if (plugin)
    {
       // plugin->unloadKmlFile(fn);
        return 0;
    }

    return -1;
}

struct ToggleNodeHandler : public ControlEventHandler
{
    ToggleNodeHandler(osg::Node *node)
        : _node(node)
    {
    }

    virtual void onValueChanged(class Control *control, bool value)
    {
        osg::ref_ptr<osg::Node> safeNode = _node.get();
        if (safeNode.valid())
            safeNode->setNodeMask(value ? ~0 : 0);
    }

    osg::observer_ptr<osg::Node> _node;
};

struct ClickViewpointHandler : public ControlEventHandler
{
    ClickViewpointHandler(const Viewpoint &vp)
        : _vp(vp)
    {
    }
    Viewpoint _vp;

    virtual void onClick(class Control *control)
    {
        //s_manip->setViewpoint( _vp, 4.5 );
    }
};

/**
 * Visitor that builds a UI control for a loaded KML file.
 */
 /*
struct KMLUIBuilder : public osg::NodeVisitor
{
    KMLUIBuilder(ControlCanvas *canvas)
        : osg::NodeVisitor(osg::NodeVisitor::TRAVERSE_ALL_CHILDREN)
        , _canvas(canvas)
    {
        _grid = new Grid();
        _grid->setAbsorbEvents(true);
        _grid->setPadding(5);
        _grid->setVertAlign(Control::ALIGN_TOP);
        _grid->setHorizAlign(Control::ALIGN_LEFT);
        _grid->setBackColor(Color(Color::Black, 0.5));
        _canvas->addControl(_grid);
    }

    void apply(osg::Node &node)
    {
        AnnotationData *data = dynamic_cast<AnnotationData *>(node.getUserData());
        if (data)
        {
            ControlVector row;
            CheckBoxControl *cb = new CheckBoxControl(node.getNodeMask() != 0, new ToggleNodeHandler(&node));
            cb->setSize(12, 12);
            row.push_back(cb);
            std::string name = data->getName().empty() ? "<unnamed>" : data->getName();
            LabelControl *label = new LabelControl(name, 14.0f);
            label->setMargin(Gutter(0, 0, 0, (this->getNodePath().size() - 3) * 20));
            if (data->getViewpoint())
            {
                label->addEventHandler(new ClickViewpointHandler(*data->getViewpoint()));
                label->setActiveColor(Color::Blue);
            }
            row.push_back(label);
            _grid->addControls(row);
        }
        traverse(node);
    }

    ControlCanvas *_canvas;
    Grid *_grid;
};*/

EarthPlugin::EarthPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    int _maxLights = Registry::instance()->getCapabilities().getMaxLights();
    plugin = this;
    mapNode = NULL;
    oldDump = NULL;

    earthTab = new coTUITab("Earth", coVRTui::instance()->mainFolder->getID());
    earthTab->setPos(0, 0);
    layerManager = new EarthLayerManager(earthTab);
    viewpointManager = new EarthViewpointManager(earthTab);

    DebugCheckBox = new coTUIToggleButton("Debug", earthTab->getID(), false);
    DebugCheckBox->setPos(3, 0);
    DebugCheckBox->setEventListener(this);
    cameraCheckBox = new coTUIToggleButton("camera", earthTab->getID(), false);
    cameraCheckBox->setPos(3, 1);
    cameraCheckBox->setEventListener(this);
    intersectionsCheckBox = new coTUIToggleButton("intersections", earthTab->getID(), false);
    intersectionsCheckBox->setPos(3, 2);
    intersectionsCheckBox->setEventListener(this);
    rttCheckBox = new coTUIToggleButton("rtt", earthTab->getID(), false);
    rttCheckBox->setPos(3, 3);
    rttCheckBox->setEventListener(this);

    requestDump = new coTUIButton("requestDump", earthTab->getID());
    requestDump->setPos(3, 4);
    requestDump->setEventListener(this);
}

void EarthPlugin::toggle(osg::Group *g, const std::string &name, bool onoff)
{
    for (unsigned i = 0; i < g->getNumChildren(); ++i)
    {
        if (g->getChild(i)->getName() == name)
        {
            g->getChild(i)->setNodeMask(onoff ? ~0 : 0);
            break;
        }
    }
}

void EarthPlugin::tabletEvent(coTUIElement *e)
{
    if (e == requestDump && mapNode)
    {
        OverlayDecorator* od = osgEarth::Util::findTopMostNodeOfType<OverlayDecorator>(mapNode);
        if (od) od->requestDump();
    }
    if (e == cameraCheckBox && oldDump)
    {
        toggle(static_cast<osg::Group *>(oldDump), "camera", cameraCheckBox->getState());
    }
    if (e == intersectionsCheckBox && oldDump)
    {
        toggle(static_cast<osg::Group *>(oldDump), "intersection", intersectionsCheckBox->getState());
    }
    if (e == rttCheckBox && oldDump)
    {
        toggle(static_cast<osg::Group *>(oldDump), "rtt", rttCheckBox->getState());
    }
    if (e == DebugCheckBox && mapNode)
    {
        if (DebugCheckBox->getState())
        {

            OverlayDecorator* od = osgEarth::Util::findTopMostNodeOfType<OverlayDecorator>(mapNode);
            if (od)
            {
                osg::Node* dump = od->getDump();
                if (!dump)
                {
                    od->requestDump();
                }
                else
                {
                    dump->getOrCreateStateSet()->setAttributeAndModes(new osg::Depth(
                        osg::Depth::LEQUAL, 0, 1, false),
                        1 | osg::StateAttribute::OVERRIDE);
                    dump->getOrCreateStateSet()->setMode(GL_BLEND, 1);
                    dump->getOrCreateStateSet()->setAttributeAndModes(new osg::LineWidth(1.5f), 1);

                    cover->getObjectsRoot()->removeChild(oldDump);
                    cover->getObjectsRoot()->addChild(dump);
                    oldDump = dump;

                    toggle(static_cast<osg::Group*>(oldDump), "camera", cameraCheckBox->getState());
                    //toggle(_parent, "overlay", s_overlayCheck->getValue());
                    toggle(static_cast<osg::Group*>(oldDump), "intersection", intersectionsCheckBox->getState());
                    toggle(static_cast<osg::Group*>(oldDump), "rtt", rttCheckBox->getState());
                }
            }
        }
        else
        {
        }
    }
}

const osgEarth::SpatialReference *EarthPlugin::getSRS() const
{
    return mapNode->getMap()->getProfile()->getSRS();
}

int EarthPlugin::loadFile(const char *fn, osg::Group *parent)
{

    //Tell the database pager to not modify the unref settings
    VRViewer::instance()->getDatabasePager()->setUnrefImageDataAfterApplyPolicy(false, false);

    //Load the map
    loadedModel = osgDB::readNodeFile(fn);

    //Find the MapNode
    mapNode = MapNode::findMapNode(loadedModel);
    if (mapNode)
    {

        layerManager->setMap(mapNode->getMap());
        // look for external data:

        const Config &externals = mapNode->externalConfig();
        const ConfigSet children = externals.children("viewpoint");
        if (children.size() > 0)
        {
            for (ConfigSet::const_iterator i = children.begin(); i != children.end(); ++i)
                viewpointManager->addViewpoint(new Viewpoint(*i));
        }
        auto extensions = mapNode->getExtensions();
        for (const auto& extension : extensions)
        {
            Extension* e = extension;
          /*  osgEarth::Viewpoints::ViewpointsExtension* ve = dynamic_cast<osgEarth::Viewpoints::ViewpointsExtension*>(e);
            const ConfigSet children = ve->getConfig().children("viewpoint");
            if (children.size() > 0)
            {
                for (ConfigSet::const_iterator i = children.begin(); i != children.end(); ++i)
                    viewpointManager->addViewpoint(new Viewpoint(*i));
            }*/
        }
        mapNode->open(); // necessary to resolve the SRS on the next line
        std::string ext = mapNode->getMapSRS()->isGeographic() ? "sky_simple" : "sky_gl";
        mapNode->addExtension(Extension::create(ext, ConfigOptions()));
        /*if (mapNode->getMapSRS()->isGeocentric())
        {

            // Sky model.
            Config skyConf = externals.child("sky");
            if (!skyConf.empty())
                useSky = true;

            if (useSky)
            {
                double hours = skyConf.value("hours", 12.0);
#if OSGEARTH_VERSION_LESS_THAN(2,6,0)
                s_sky = new SkyNode(mapNode->getMap());
                s_sky->setDateTime(2011, 3, 6, hours);
#else
                s_sky = SkyNode::create(mapNode);
                s_sky->setDateTime(DateTime(2011, 3, 6, hours));
#endif
                s_sky->attach(VRViewer::instance());
                parent->addChild(s_sky);
            }

            // Ocean surface.
            if (externals.hasChild("ocean"))
                useOcean = true;

            // if ( useOcean )
		   //{
			//   s_ocean = new OceanSurfaceNode( mapNode, externals.child("ocean") );
			//   if ( s_ocean )
			//	   parent->addChild( s_ocean );
		   //}

            // The automatic clip plane generator will adjust the near/far clipping
            // planes based on your view of the horizon. This prevents near clipping issues
            // when you are very close to the ground. If your app never brings a user very
            // close to the ground, you may not need this.
            if (externals.hasChild("autoclip"))
            {
#if OSGEARTH_VERSION_GREATER_THAN(2,10,0)
                useAutoClip = externals.child("autoclip").valueAs(useAutoClip);
#else
                useAutoClip = externals.child("autoclip").boolValue(useAutoClip);
#endif
            }

            if (useSky || useAutoClip || useOcean)
            {
                VRViewer::instance()->getCamera()->addCullCallback(new AutoClipPlaneCullCallback(mapNode));
            }
        }
        */
        // Configure the de-cluttering engine for labels and annotations:
        //const Config &declutterConf = externals.child("decluttering");
        //if (!declutterConf.empty())
        //{
        //    osgEarth::Decluttering::setOptions(osgEarth::DeclutteringOptions(declutterConf));
        //}
    }

    if (loadedModel)
    {
        loadedModel->setNodeMask(loadedModel->getNodeMask() & ~(Isect::Walk | Isect::Intersection | Isect::Collision | Isect::Touch | Isect::Pick));
        // enable Intersection only after first rendering traversal, otherwise standard AutoTransform is not defined --> lots of NAN
        parent->addChild(loadedModel);
        return 0;
    }
    return -1;
}

int EarthPlugin::unloadFile(const char *fn)
{
    if (loadedModel)
        cover->getScene()->removeChild(loadedModel);
    loadedModel = NULL;
    return 1;
}

int EarthPlugin::loadKmlFile(const char *fn, osg::Group *parent)
{
    //Load the map
    loadedModel = osgDB::readNodeFile(fn);
    // Install a new Canvas for our UI controls, or use one that already exists.
    ControlCanvas* canvas = ControlCanvas::getOrCreate(VRViewer::instance());

    Container* mainContainer;
    
    {
        mainContainer = new VBox();
        mainContainer->setAbsorbEvents(false);
        mainContainer->setBackColor(Color(Color::Black, 0.8));
        mainContainer->setHorizAlign(Control::ALIGN_LEFT);
        mainContainer->setVertAlign(Control::ALIGN_BOTTOM);
    }
    canvas->addControl(mainContainer);
    //Find the MapNode
    mapNode = MapNode::findMapNode(loadedModel);
    if (mapNode)
    {
        KML::KMLOptions kmlo;
        kmlo.declutter() = true;
        bool kmlUI = true;
        osg::Node *kml = KML::load(URI(fn), mapNode, kmlo);
        if(kml)
        {
            if (kmlUI)
            {
                Control* c = AnnotationGraphControlFactory().create(kml, VRViewer::instance());
                if (c)
                {
                    c->setVertAlign(Control::ALIGN_TOP);
                    mainContainer->addControl(c);
                }
            }
            mapNode->addChild(kml);
        }
    }

    if (loadedModel)
    {
        parent->addChild(loadedModel);
        return 0;
    }
    return -1;
}

int EarthPlugin::unloadKmlFile(const char *fn)
{
    if (loadedModel)
        cover->getScene()->removeChild(loadedModel);
    loadedModel = NULL;
    return 1;
}

// this is called if the plugin is removed at runtime
EarthPlugin::~EarthPlugin()
{
    plugin = NULL;
}

bool
EarthPlugin::init()
{
    fprintf(stderr, "EarthPlugin::EarthPlugin\n");

    coVRFileManager::instance()->registerFileHandler(&fileHandler[0]);
    GLUtils::setGlobalDefaults(VRViewer::instance()->getCamera()->getOrCreateStateSet());
    GLUtils::setLighting(VRViewer::instance()->getCamera()->getOrCreateStateSet(), osg::StateAttribute::ON);
    /*
   // Create a Map and set it to Geocentric to display a globe
   Map* map = new Map();

   // Add an imagery layer (blue marble from a TMS source)

   {
   TMSOptions tms;
   tms.url() = "http://tilecache.osgeo.org/wms-c/Basic.py/1.0.0/satellite/";
   ImageLayer* layer = new ImageLayer( "NASA", tms );
   map->addImageLayer( layer );
   }

   // Add an elevationlayer (SRTM from a local GeoTiff file)

   {
   GDALOptions gdal;
   gdal.url() = "C:/src/cvsbase/osgearth.git/data/terrain/mt_everest_90m.tif";
   ElevationLayer* layer = new ElevationLayer( "SRTM", gdal );
   map->addElevationLayer( layer );
   }

   // Add an OpenStreetMap image source
   TMSOptions driverOpt;
   driverOpt.url() = "http://tile.openstreetmap.org/";
   driverOpt.tmsType() = "google";

   ImageLayerOptions layerOpt( "OSM", driverOpt );
   layerOpt.profile() = ProfileOptions( "global-mercator" );

   ImageLayer* osmLayer = new ImageLayer( layerOpt );
   map->addImageLayer( osmLayer );*/
    // Create a MapNode to render this map:
    /* MapNode* mapNode = new MapNode( map );


   cover->getObjectsRoot()->addChild(mapNode );*/

    return true;
}
void
EarthPlugin::preFrame()
{
}

COVERPLUGIN(EarthPlugin)
