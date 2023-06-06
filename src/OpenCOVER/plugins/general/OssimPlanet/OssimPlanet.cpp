/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// **************************************************************************
//
// Description:   OssimPlanet viewer
//
// Author:        Jurgen Schulze (jschulze@ucsd.edu)
//
// Creation Date: 2007-02-12
//
// **************************************************************************

// Covise:
#include <cover/coVRPluginSupport.h>
#include <cover/coVRSceneView.h>
#include <cover/VRViewer.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coPotiMenuItem.h>
#include <OpenVRUI/coPanel.h>
#include <OpenVRUI/coFlatPanelGeometry.h>
#include <OpenVRUI/coFlatButtonGeometry.h>
#include <OpenVRUI/coRectButtonGeometry.h>
#include <OpenVRUI/coValuePoti.h>
#include <OpenVRUI/coButton.h>
#include <OpenVRUI/coLabel.h>
#include <OpenVRUI/coFrame.h>
#include <config/CoviseConfig.h>
#include <string.h>

// OSG:
#include <osg/Node>
#include <osg/StateSet>
#include <osg/CullFace>
#include <osg/PolygonMode>
#include <osgDB/ReadFile>
#include <osgDB/WriteFile>
#include <osgUtil/SceneView>

// ossimPlanet:
#include <osgDB/Registry>
#include <osgGA/TerrainManipulator>
#include <ossimPlanet/ossimPlanet.h>
#include <ossimPlanet/ossimPlanetDatabasePager.h>
#include <ossimPlanet/ossimPlanetLand.h>
#include <ossimPlanet/ossimPlanetLatLonHud.h>
#include <ossimPlanet/ossimPlanetManipulator.h>
#include <ossimPlanet/ossimPlanetOssimImageLayer.h>
#include <ossimPlanet/ossimPlanetTextureLayerGroup.h>
#include <ossimPlanet/ossimPlanetTextureLayerRegistry.h>
#include <ossim/base/ossimArgumentParser.h>
#include <ossim/init/ossimInit.h>
#include <wms/wms.h>

// Local:
#include "OssimPlanet.h"

OssimPlanet *plugin = NULL;
using namespace osg;

/// Constructor
OssimPlanet::OssimPlanet()
: coVRPlugin(COVER_PLUGIN_NAME)
{
}

bool OssimPlanet::init()
{
    _planet = NULL;
    _databasePager = NULL;

    //create panel
    _panel = new coPanel(new coFlatPanelGeometry(coUIElement::BLACK));

    // Create main menu button
    _ossimPlanetMenuItem = new coSubMenuItem("OssimPlanet");
    _ossimPlanetMenuItem->setMenuListener(this);
    cover->getMenu()->add(_ossimPlanetMenuItem);

    _ossimPlanetMenu = new coRowMenu("OssimPlanet");
    _showPlanetItem = new coCheckboxMenuItem("Show Planet", false);
    _hudModeItem = new coCheckboxMenuItem("HUD", false);
    _wireframeModeItem = new coCheckboxMenuItem("Wireframe", false);
    _elevationDial = new coPotiMenuItem("Elevation", 0.1, 10.0, 3.0);
    _splitMetricDial = new coPotiMenuItem("Split Metric", 0.1, 20.0, 10.0);
    _flatlandModeItem = new coCheckboxMenuItem("Flatland", false);

    _layersMenu = new coRowMenu("Layers");
    _layersMenuItem = new coSubMenuItem("Layers");
    _layersMenuItem->setMenuListener(this);
    _layersMenuItem->setMenu(_layersMenu);
    _bordersItem = new coCheckboxMenuItem("Country Borders", false);

    _showPlanetItem->setMenuListener(this);
    _hudModeItem->setMenuListener(this);
    _wireframeModeItem->setMenuListener(this);
    _elevationDial->setMenuListener(this);
    _splitMetricDial->setMenuListener(this);
    _flatlandModeItem->setMenuListener(this);
    _ossimPlanetMenu->add(_showPlanetItem);
    _ossimPlanetMenu->add(_layersMenuItem);
    _ossimPlanetMenu->add(_hudModeItem);
    _ossimPlanetMenu->add(_wireframeModeItem);
    _ossimPlanetMenu->add(_elevationDial);
    _ossimPlanetMenu->add(_splitMetricDial);
    _ossimPlanetMenu->add(_flatlandModeItem);
    _ossimPlanetMenuItem->setMenu(_ossimPlanetMenu);
    _layersMenu->add(_bordersItem);

    return true;
}

void OssimPlanet::createPlanet()
{
    wmsInitialize();

    ossimInit::instance()->initialize();

    // create the database pager
    _databasePager = new ossimPlanetDatabasePager;
    VRViewer::instance()->getScene()->setDatabasePager(_databasePager);

    _planet = new ossimPlanet();
    if (!_planet.get())
    {
        cerr << "Error: OssimPlanet::createPlanet failed" << endl;
        return;
    }

    // Load the kwl (Key Word List) file which tells OSSIM what imagery to use
    ossimKeywordlist kwl;
    std::string configFile = coCoviseConfig::getEntry("COVER.Plugin.OssimPlanet.ConfigFile");
    if (!configFile.empty())
    {
        kwl.addFile(configFile.c_str());
        cerr << "OssimPlanet using config file " << configFile << endl;
    }
    else
    {
        cerr << "Error: COVER.Plugin.OssimPlanet.ConfigFile required for OssimPlanet" << endl;
        return;
    }

    osg::ref_ptr<ossimPlanetTextureLayer> layer = ossimPlanetTextureLayerRegistry::instance()->createLayer(kwl.toString());
    /*
  new ossimplanet TODO
  _planet->getLand()->setTextureLayer(layer.get(), 0);
*/

    // need to do this prior to setting viewer data
    _databasePager->setExpiryDelay(0);
    /*
  new ossimplanet TODO
  _databasePager->setUseFrameBlock(false); */

    // pass the loaded scene graph to the viewer.
    _planet->setCullingActive(false); // doesn't fix culling problem
    /*
  new ossimplanet TODO
  _planet->getLand()->setLandType(ossimPlanetLandType_NORMALIZED_ELLIPSOID);
  _planet->getLand()->setElevationEnabledFlag(true);
  _planet->getLand()->setHeightExag(_elevationDial->getValue());
  _planet->getLand()->setMaxLevelDetail(16);
  _planet->getLand()->setElevationPatchSize(32);
  _planet->getLand()->setSplitMetricRatio(_splitMetricDial->getValue());
  _planet->setEnableHudFlag(false);
*/

    cover->getObjectsRoot()->addChild(_planet.get());
    //  cover->screens[0].sv->setCullingMode(osgUtil::CullVisitor::NO_CULLING); // doesn't fix culling problem
}

/// Destructor
OssimPlanet::~OssimPlanet()
{
    // clean things up
    if (_databasePager)
    {
        _databasePager->setAcceptNewDatabaseRequests(false);
        _databasePager->cancel();
        cover->getObjectsRoot()->removeChild(_planet.get());
        wmsFinalize();
    }

    delete _flatlandModeItem;
    delete _splitMetricDial;
    delete _elevationDial;
    delete _wireframeModeItem;
    delete _showPlanetItem;
    delete _hudModeItem;
    delete _panel;
    delete _ossimPlanetMenuItem;
    delete _ossimPlanetMenu;

    delete _layersMenu;
    delete _layersMenuItem;
    delete _bordersItem;
}

void OssimPlanet::menuReleaseEvent(coMenuItem *item)
{
    if (item == _elevationDial)
    {
        cerr << "menuReleaseEvent" << endl;
        if (_planet.get())
        {
            //      _planet->getLand()->setHeightExag(_elevationDial->getValue());
            //    _planet->getLand()->resetGraph();
        }
    }
    else if (item == _splitMetricDial)
    {
        // new ossimplanet TODO    _planet->getLand()->setSplitMetricRatio(_splitMetricDial->getValue());
    }
}

void OssimPlanet::menuEvent(coMenuItem *item)
{
    if (item == _showPlanetItem)
    {
        if (_showPlanetItem->getState())
        {
            createPlanet();
        }
        else
        {
            if (_planet.get())
            {
                cover->getObjectsRoot()->removeChild(_planet.get());
            }
        }
    }
    else if (item == _bordersItem)
    {
    }
    else if (item == _hudModeItem)
    {
        // new ossimplanet TODO if (_planet.get()) _planet->setEnableHudFlag(_hudModeItem->getState());
    }
    else if (item == _wireframeModeItem)
    {
        setWireFrameMode(_wireframeModeItem->getState());
    }
    else if (item == _flatlandModeItem)
    {
        // new ossimplanet TODOif (_planet.get()) _planet->getLand()->setLandType((_flatlandModeItem->getState()) ? ossimPlanetLandType_FLAT : ossimPlanetLandType_NORMALIZED_ELLIPSOID);
    }
}

// need to define because abstract
void OssimPlanet::potiValueChanged(float, float, coValuePoti *, int)
{
}

// load new structure listener
void OssimPlanet::buttonEvent(coButton *)
{
}

/// Called before each frame
void OssimPlanet::preFrame()
{
    //  glDisable( GL_CULL_FACE );
    if (_planet.get())
    {
        Vec3d eyePos = _planet->getEyePositionLatLonHeight();
        // new ossimplanet TODO  double normalizationFactor = _planet->getLand()->getModel()->getNormalizationScale();
        //    cerr << "latitude = " << eyePos[0] << " longitude = " << eyePos[1] << " height = " << eyePos[2]*normalizationFactor << endl;
    }
}

void OssimPlanet::setWireFrameMode(bool newState)
{
    if (_planet.get())
    {
        osg::StateSet *planetState = _planet->getOrCreateStateSet();

        // Disable culling:
        ref_ptr<CullFace> cull_face = new CullFace;
        cull_face->setMode(CullFace::FRONT_AND_BACK);
        planetState->setAttribute(cull_face.get(), StateAttribute::ON | StateAttribute::OVERRIDE);
        planetState->setMode(GL_CULL_FACE, StateAttribute::OFF | StateAttribute::OVERRIDE);

        osg::PolygonMode *polyModeObj;
        polyModeObj = dynamic_cast<osg::PolygonMode *>(planetState->getAttribute(osg::StateAttribute::POLYGONMODE));
        if (!polyModeObj)
        {
            polyModeObj = new osg::PolygonMode;
            planetState->setAttribute(polyModeObj);
        }

        if (newState)
        {
            polyModeObj->setMode(osg::PolygonMode::FRONT_AND_BACK, osg::PolygonMode::LINE);
            cerr << "wireframe is now on" << endl;
        }
        else
        {
            polyModeObj->setMode(osg::PolygonMode::FRONT_AND_BACK, osg::PolygonMode::FILL);
            cerr << "wireframe is now off" << endl;
        }
    }
}

COVERPLUGIN(OssimPlanet)
