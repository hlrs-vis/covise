/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                            (C)2001 HLRS  **
 **                                                                          **
 ** Description: TrafficSimulation Plugin                                    **
 **                                                                          **
 **                                                                          **
 ** Author: Florian Seybold, U.Woessner		                                **
 **                                                                          **
 ** History:  								                                         **
 ** Nov-01  v1	    				       		                                   **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include "TrafficSimulationPlugin.h"

#include <TrafficSimulation/FindTrafficLightSwitch.h>
#include <TrafficSimulation/coTrafficSimulation.h>

#include <cover/coVRPluginSupport.h>
#include <cover/RenderObject.h>
#include <cover/coVRShader.h>
#include <cover/coVRFileManager.h>
#include <cover/coVRTui.h>
#include <cover/coVRMSController.h>

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

#include <xercesc/parsers/XercesDOMParser.hpp>
#include <TrafficSimulation/HumanVehicle.h>

#include <TrafficSimulation/PorscheFFZ.h>

using namespace covise;
using namespace opencover;
using namespace vehicleUtil;
using namespace TrafficSimulation;

TrafficSimulationPlugin::TrafficSimulationPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    plugin = this;

}

TrafficSimulationPlugin *TrafficSimulationPlugin::plugin = NULL;

static FileHandler handlers[] = {
    { NULL,
      TrafficSimulationPlugin::loadOD,
      NULL,
      "xodr" }
};

// this is called if the plugin is removed at runtime
TrafficSimulationPlugin::~TrafficSimulationPlugin()
{


    coVRFileManager::instance()->unregisterFileHandler(&handlers[0]);
    //coVRFileManager::instance()->unregisterFileHandler(&handlers[1]);

	coTrafficSimulation::freeInstance();
}


int TrafficSimulationPlugin::unloadOD(const char *, const char *key)
{
	coTrafficSimulation::instance()->deleteRoadSystem();
    return 1;
}

int TrafficSimulationPlugin::loadOD(const char *filename, osg::Group *, const char *key)
{
    if (coTrafficSimulation::instance()->loadRoadSystem(filename))
        return 1;
    else
        return 0;
}

bool TrafficSimulationPlugin::init()
{

    coVRFileManager::instance()->registerFileHandler(&handlers[0]);
    //coVRFileManager::instance()->registerFileHandler(&handlers[1]);
    cover->setScale(1000);
	coTrafficSimulation::useInstance();

    pluginTab = new coTUITab("Traffic Simulation", coVRTui::instance()->mainFolder->getID());
    pluginTab->setPos(0, 0);
    startButton = new coTUIButton("Continue", pluginTab->getID());
    startButton->setEventListener(this);
    startButton->setPos(0, 0);
    stopButton = new coTUIButton("Stop", pluginTab->getID());
    stopButton->setEventListener(this);
    stopButton->setPos(0, 1);
    saveButton = new coTUIFileBrowserButton("Save OpenDRIVE...", pluginTab->getID());
    saveButton->setEventListener(this);
    saveButton->setPos(1, 0);
    saveButton->setMode(coTUIFileBrowserButton::SAVE);
    saveButton->setFilterList("*.xodr");
    saveButton->setCurDir(".");
    openC4DXMLButton = new coTUIFileBrowserButton("Import Cinema4D XML...", pluginTab->getID());
    openC4DXMLButton->setEventListener(this);
    openC4DXMLButton->setPos(1, 1);
    openC4DXMLButton->setMode(coTUIFileBrowserButton::OPEN);
    openC4DXMLButton->setFilterList("*.xml");
    openLandXMLButton = new coTUIFileBrowserButton("Import LandXML...", pluginTab->getID());
    openLandXMLButton->setEventListener(this);
    openLandXMLButton->setPos(1, 2);
    openLandXMLButton->setMode(coTUIFileBrowserButton::OPEN);
    openLandXMLButton->setFilterList("*.xml");
    openIntermapRoadButton = new coTUIFileBrowserButton("Import Intermap road...", pluginTab->getID());
    openIntermapRoadButton->setEventListener(this);
    openIntermapRoadButton->setPos(1, 3);
    openIntermapRoadButton->setMode(coTUIFileBrowserButton::OPEN);
    openIntermapRoadButton->setFilterList("*.xyz");
    exportSceneGraphButton = new coTUIFileBrowserButton("Export scene graph...", pluginTab->getID());
    exportSceneGraphButton->setEventListener(this);
    exportSceneGraphButton->setPos(1, 4);
    exportSceneGraphButton->setMode(coTUIFileBrowserButton::SAVE);
    exportSceneGraphButton->setFilterList("*");
    loadTerrainButton = new coTUIFileBrowserButton("Load VPB terrain...", pluginTab->getID());
    loadTerrainButton->setEventListener(this);
    loadTerrainButton->setPos(1, 5);
    loadTerrainButton->setMode(coTUIFileBrowserButton::OPEN);
    loadTerrainButton->setFilterList("*.ive");

    // Remove Agents //
    //
    // button to remove agents that are slower than the specified velocity
    removeAgentsButton = new coTUIButton("Delete cars slower than:", pluginTab->getID());
    removeAgentsButton->setEventListener(this);
    removeAgentsButton->setPos(0, 10);
    removeAgentsVelocity_ = 1;
    removeAgentsSlider = new coTUISlider("Velocity", pluginTab->getID());
    removeAgentsSlider->setEventListener(this);
    removeAgentsSlider->setPos(1, 10);
    removeAgentsSlider->setMin(1);
    removeAgentsSlider->setMax(255);
    removeAgentsSlider->setValue(removeAgentsVelocity_);

    debugRoadButton = new coTUIToggleButton("DebugRoad", pluginTab->getID());
    debugRoadButton->setEventListener(this);
    debugRoadButton->setPos(0, 11);
    debugRoadButton->setState(false);

    //FFZ Tab
    pluginTab = new coTUITab("FFZ", coVRTui::instance()->mainFolder->getID());
    pluginTab->setPos(0, 0);
    pluginTab->setEventListener(this);
    int pos = 1;
    int frame = 1;

    //Carpool Frame
    carpoolFrame = new coTUIFrame("Carpool-Frame", pluginTab->getID());
    carpoolFrame->setPos(0, frame++);

    //Carpool Label
    carpoolLabel = new coTUILabel("Carpool\n", carpoolFrame->getID());
    carpoolLabel->setPos(0, pos++);

    //Carpool an/aus
    /*toggleCarpoolLabel = new coTUILabel("Toggle Carpool State", pluginTab->getID());
 *         toggleCarpoolLabel->setPos(0,pos);*/
    useCarpoolButton = new coTUIButton("Toggle Carpool State", carpoolFrame->getID());
    useCarpoolButton->setEventListener(this);
    useCarpoolButton->setPos(0, pos);
    useCarpool_ = 1;

    carpoolField = new coTUIEditField("ON", carpoolFrame->getID());
    carpoolField->setPos(1, pos);

    carpoolStateField = new coTUIEditField("", carpoolFrame->getID());
    carpoolStateField->setPos(2, pos++);
    carpoolStateField->setColor(Qt::darkGreen);

    /*carpoolStateField = new coTUIEditIntField("Carpool State", pluginTab->getID());
 *         carpoolStateField->setValue(useCarpool_);
 *                 carpoolStateField->setPos(1,pos++);*/

    pos = 0;
    //Create Frame
    createFrame = new coTUIFrame("Create-Frame", pluginTab->getID());
    createFrame->setPos(0, frame++);

    //Create Vehicles Label
    createLabel = new coTUILabel("Create Vehicles\n", createFrame->getID());
    createLabel->setPos(0, pos++);

    //Minimalabstand - Fahrzeuge erstellen
    createVehiclesAtMinLabel = new coTUILabel("Set min distance:", createFrame->getID());
    createVehiclesAtMinLabel->setPos(0, pos);

    /*createVehiclesAtMin_Button = new coTUIButton("Set min distance", pluginTab->getID());
 *         createVehiclesAtMin_Button->setEventListener(this);
 *                 createVehiclesAtMin_Button->setPos(0,pos);*/
    createVehiclesAtMin_ = 180;
    createVehiclesAtMin_Slider = new coTUISlider("min distance", createFrame->getID());
    createVehiclesAtMin_Slider->setEventListener(this);
    createVehiclesAtMin_Slider->setPos(1, pos++);
    createVehiclesAtMin_Slider->setMin(1);
    createVehiclesAtMin_Slider->setMax(2000);
    createVehiclesAtMin_Slider->setValue(createVehiclesAtMin_);

    //Geschwindigkeit bis zur welcher createVehiclesAtMin_ gilt
    minVelLabel = new coTUILabel("... at velocity [km/h]:", createFrame->getID());
    minVelLabel->setPos(0, pos);

    /*minVel_Button  = new coTUIButton("... at v:", pluginTab->getID());
 *         minVel_Button->setEventListener(this);
 *                 minVel_Button->setPos(0,pos);*/
    minVel_ = 50;
    minVel_Slider = new coTUISlider("min v", createFrame->getID());
    minVel_Slider->setEventListener(this);
    minVel_Slider->setPos(1, pos++);
    minVel_Slider->setMin(0);
    minVel_Slider->setMax(300);
    minVel_Slider->setValue(minVel_);

    //Maximalabstand - Fahrzeuge erstellen
    createVehiclesAtMaxLabel = new coTUILabel("Set max distance:", createFrame->getID());
    createVehiclesAtMaxLabel->setPos(0, pos);

    /*createVehiclesAtMax_Button = new coTUIButton("Set max distance", pluginTab->getID());
 *         createVehiclesAtMax_Button->setEventListener(this);
 *                 createVehiclesAtMax_Button->setPos(0,pos);*/
    createVehiclesAtMax_ = 800;
    createVehiclesAtMax_Slider = new coTUISlider("Create Vehicles", createFrame->getID());
    createVehiclesAtMax_Slider->setEventListener(this);
    createVehiclesAtMax_Slider->setPos(1, pos++);
    createVehiclesAtMax_Slider->setMin(1);
    createVehiclesAtMax_Slider->setMax(2000);
    createVehiclesAtMax_Slider->setValue(createVehiclesAtMax_);

    //Geschwindigkeit ab welcher createVehiclesAtMax gilt
    maxVelLabel = new coTUILabel("... at velocity [km/h]:", createFrame->getID());
    maxVelLabel->setPos(0, pos);

    /*maxVel_Button  = new coTUIButton("... at v:", pluginTab->getID());
 *         maxVel_Button->setEventListener(this);
 *                 maxVel_Button->setPos(0,pos);*/
    maxVel_ = 100;
    maxVel_Slider = new coTUISlider("max v", createFrame->getID());
    maxVel_Slider->setEventListener(this);
    maxVel_Slider->setPos(1, pos++);
    maxVel_Slider->setMin(0);
    maxVel_Slider->setMax(300);
    maxVel_Slider->setValue(maxVel_);

    //Timer für die FFZ-Erstellung
    createFreqLabel = new coTUILabel("Create Vehicles every x Frames:", createFrame->getID());
    createFreqLabel->setPos(0, pos);

    /*createFreqButton = new coTUIButton("Create Vehicles every x Frames:", pluginTab->getID());
 *         createFreqButton->setEventListener(this);
 *                 createFreqButton->setPos(0,pos);*/
    createFreq_ = coTrafficSimulation::instance()->createFreq;
    createFreqSlider = new coTUISlider("Create Vehicles", createFrame->getID());
    createFreqSlider->setEventListener(this);
    createFreqSlider->setPos(1, pos++);
    createFreqSlider->setMin(1);
    createFreqSlider->setMax(300);
    createFreqSlider->setValue(createFreq_);

    //Maximale Anzahl an FFZ bestimmen
    maxVehiclesLabel = new coTUILabel("Maximum amount of Vehicles [0: unlimited]:", createFrame->getID());
    maxVehiclesLabel->setPos(0, pos);
    maxVehicles_ = 0;
    maxVehiclesSlider = new coTUISlider("Max Vehicles", createFrame->getID());
    maxVehiclesSlider->setEventListener(this);
    maxVehiclesSlider->setPos(1, pos++);
    maxVehiclesSlider->setMin(0);
    maxVehiclesSlider->setMax(200);
    maxVehiclesSlider->setValue(maxVehicles_);

    pos = 0;
    //Remove Vehicles Frame
    removeFrame = new coTUIFrame("Remove-Frame", pluginTab->getID());
    removeFrame->setPos(0, frame++);

    //Remove Vehicles Label
    removeLabel = new coTUILabel("Remove Vehicles\n", removeFrame->getID());
    removeLabel->setPos(0, pos++);

    //Fahrzeuge löschen
    removeVehiclesAtLabel = new coTUILabel("Remove Vehicles at creation distance + x:", removeFrame->getID());
    removeVehiclesAtLabel->setPos(0, pos);

    /*removeVehiclesAtButton = new coTUIButton("Remove Vehicles at creation distance + x:", pluginTab->getID());
 *         removeVehiclesAtButton->setEventListener(this);
 *                 removeVehiclesAtButton->setPos(0,pos);*/
    removeVehiclesDelta_ = 50;
    removeVehiclesAtSlider = new coTUISlider("Remove Vehicles", removeFrame->getID());
    removeVehiclesAtSlider->setEventListener(this);
    removeVehiclesAtSlider->setPos(1, pos++);
    removeVehiclesAtSlider->setMin(1);
    removeVehiclesAtSlider->setMax(1000);
    removeVehiclesAtSlider->setValue(removeVehiclesDelta_);

    pos = 0;
    //Traffic Density Frame
    tdFrame = new coTUIFrame("Traffic Density-Frame", pluginTab->getID());
    tdFrame->setPos(0, frame++);

    //Traffic Density Label
    tdLabel = new coTUILabel("Traffic Density\n", tdFrame->getID());
    tdLabel->setPos(0, pos++);

    //Faktor für die Verkehrsdichte
    td_multLabel = new coTUILabel("Set traffic density multiplier:", tdFrame->getID());
    td_multLabel->setPos(0, pos);

    /*td_multButton = new coTUIButton("Set traffic density multiplier", pluginTab->getID());
 *         td_multButton->setEventListener(this);
 *                 td_multButton->setPos(0,pos);*/
    td_mult_ = 1.0;
    td_multSlider = new coTUIFloatSlider("TD multiplier", tdFrame->getID());
    td_multSlider->setEventListener(this);
    td_multSlider->setPos(1, pos);
    td_multSlider->setMin(0);
    td_multSlider->setMax(20);
    td_multSlider->setValue(td_mult_);

    multiField = new coTUIEditField("ON", tdFrame->getID());
    multiField->setPos(3, pos);

    tdMultField = new coTUIEditField("", tdFrame->getID());
    tdMultField->setPos(4, pos++);
    tdMultField->setColor(Qt::darkGreen);

    //Absolutwert der Verkehrsdichte
    td_valueLabel = new coTUILabel("Set placeholder [m]:", tdFrame->getID());
    td_valueLabel->setPos(0, pos);

    placeholder_ = -1;

    td_valueSlider = new coTUIFloatSlider("traffic density", tdFrame->getID());
    td_valueSlider->setEventListener(this);
    td_valueSlider->setPos(1, pos);
    td_valueSlider->setMin(0);
    td_valueSlider->setMax(100);
    td_valueSlider->setValue(15);

    tdField = new coTUIEditField("OFF", tdFrame->getID());
    tdField->setPos(3, pos);

    tdValueField = new coTUIEditField(" ", tdFrame->getID());
    tdValueField->setColor(Qt::red);
    tdValueField->setPos(4, pos++);

//
// Operator Map // TODO: SO FAR ONLY FOR TESTING!!!
//
// a tab with a map that shows all the cars
// 	std::string opMapFileString = coCoviseConfig::getEntry("image","COVER.Plugin.TrafficSimulation.OperatorMap.Map");
// 	if(!opMapFileString.empty()) {
// 		const char* opMapFileName = opMapFileString.c_str();
// 		operatorMapTab = new coTUITab("Operator Map", coVRTui::instance()->mainFolder->getID());
// 		operatorMap = new coTUIMap("OperatorMap", operatorMapTab->getID());
// 		if(coVRMSController::instance()->isMaster()) {
// 			operatorMap->addMap(opMapFileName, 600.0, 1620.0, 3072.0, 2048.0, 0.0); // TODO
// 		}
// 		// TODO: define in xodr?
// 		/*
// 		// TODO: doesn't work:
// 		coCoviseConfig::ScopeEntries e = coCoviseConfig::getScopeEntries("COVER.Plugin.TrafficSimulation.OperatorMap", "Map");
// 		const char** entries = e.getValue();
// 		while(entries && *entries) {
// 			std::cout << *entries << std::endl;
// 			++entries;
// 		}
// 		*/
// 	}


    sphereTransform = new osg::MatrixTransform();
    sphere = new osg::Sphere(osg::Vec3(0, 0, 0), 20);
    sphereGeode = new osg::Geode();
    osg::ShapeDrawable *sd = new osg::ShapeDrawable(sphere.get());
    sd->setUseDisplayList(false);
    sphereGeode->addDrawable(sd);
    sphereTransform->addChild(sphereGeode.get());

    sphereGeoState = sphereGeode->getOrCreateStateSet();
    redmtl = new osg::Material;
    //transpmtl->setColorMode(Material::AMBIENT_AND_DIFFUSE);
    redmtl->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(0.2f, 0.2f, 0.2f, 0.4));
    redmtl->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(1.0f, 0.0f, 0.0f, 0.4));
    redmtl->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(1.0f, 1.0f, 1.0f, 0.4));
    redmtl->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4(0.0f, 0.0f, 0.0f, 0.4));
    redmtl->setShininess(osg::Material::FRONT_AND_BACK, 5.0f);
    redmtl->setTransparency(osg::Material::FRONT_AND_BACK, 0.4);

    greenmtl = new osg::Material;
    greenmtl->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(0.2f, 0.2f, 0.2f, 0.4));
    greenmtl->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(1.0f, 0.0f, 0.0f, 0.4));
    greenmtl->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(1.0f, 1.0f, 1.0f, 0.4));
    greenmtl->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4(0.0f, 0.0f, 0.0f, 0.4));
    greenmtl->setShininess(osg::Material::FRONT_AND_BACK, 5.0f);
    greenmtl->setTransparency(osg::Material::FRONT_AND_BACK, 0.4);

    sphereGeoState->setAttributeAndModes(redmtl.get(), osg::StateAttribute::ON);
    sphereGeoState->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
    sphereGeoState->setMode(GL_BLEND, osg::StateAttribute::ON);
    sphereGeoState->setNestRenderBins(false);

    return coTrafficSimulation::instance()->init();
}

void
TrafficSimulationPlugin::preFrame()
{
    if (debugRoadButton->getState())
    {
        osg::Vec3d hp = cover->getPointerMat().getTrans();
        osg::Vec3d spherePos = hp;

        sphereGeoState->setAttributeAndModes(redmtl.get(), osg::StateAttribute::ON);
        double xr = hp[0];
        double yr = hp[1];
        if (RoadSystem::Instance())
        {
            Vector2D v_c(std::numeric_limits<float>::signaling_NaN(), std::numeric_limits<float>::signaling_NaN());
            static vehicleUtil::Road *currentRoad = NULL;
            static double currentLongPos = -1;
            Vector3D v_w(xr, yr, 0);
            if (currentRoad)
            {
                v_c = RoadSystem::Instance()->searchPositionFollowingRoad(v_w, currentRoad, currentLongPos);
                if (!v_c.isNaV())
                {
                    if (currentRoad->isOnRoad(v_c))
                    {
                        RoadPoint point = currentRoad->getRoadPoint(v_c.u(), v_c.v());
                        spherePos[2] = point.z();

                        sphereGeoState->setAttributeAndModes(greenmtl.get(), osg::StateAttribute::ON);
                    }
                    else
                    {
                        currentRoad = NULL;
                    }
                }
                else if (currentLongPos < -0.1 || currentLongPos > currentRoad->getLength() + 0.1)
                {
                    // left road searching for the next road over all roads in the system
                    v_c = RoadSystem::Instance()->searchPosition(v_w, currentRoad, currentLongPos);
                    if (!v_c.isNaV())
                    {
                        if (currentRoad->isOnRoad(v_c))
                        {
                            RoadPoint point = currentRoad->getRoadPoint(v_c.u(), v_c.v());
                            spherePos[2] = point.z();
                            sphereGeoState->setAttributeAndModes(greenmtl.get(), osg::StateAttribute::ON);
                        }
                        else
                        {
                            RoadPoint point = currentRoad->getRoadPoint(v_c.u(), v_c.v());
                            spherePos[2] = point.z();
                            currentRoad = NULL;
                        }
                    }
                    else
                    {
                        currentRoad = NULL;
                    }
                }
            }
            else
            {
                // left road searching for the next road over all roads in the system
                v_c = RoadSystem::Instance()->searchPosition(v_w, currentRoad, currentLongPos);
                if (!v_c.isNaV())
                {
                    if (currentRoad->isOnRoad(v_c))
                    {
                        RoadPoint point = currentRoad->getRoadPoint(v_c.u(), v_c.v());
                        spherePos[2] = point.z();
                        sphereGeoState->setAttributeAndModes(greenmtl.get(), osg::StateAttribute::ON);
                    }
                    else
                    {
                        RoadPoint point = currentRoad->getRoadPoint(v_c.u(), v_c.v());
                        spherePos[2] = point.z();
                        currentRoad = NULL;
                    }
                }
                else
                {
                    currentRoad = NULL;
                }
            }
        }
    }
	coTrafficSimulation::instance()->preFrame();
}

void TrafficSimulationPlugin::tabletEvent(coTUIElement *tUIItem)
{
    if (tUIItem == saveButton)
    {
        //std::cout << "Save Button pressed: Selected path: " << saveButton->getSelectedPath() << ", filename: " << saveButton->getFilename("") << std::endl;
        std::string filename = saveButton->getSelectedPath();
        size_t spos = filename.find("file://");
        if (spos != std::string::npos)
        {
            filename.erase(spos, 7);
        }

		coTrafficSimulation::instance()->system->writeOpenDrive(filename);
    }

    else if (tUIItem == openC4DXMLButton || tUIItem == openLandXMLButton || tUIItem == openIntermapRoadButton)
    {
        if (coTrafficSimulation::instance()->system == NULL)
        {
			coTrafficSimulation::instance()->system = RoadSystem::Instance();
        }

        std::string filename;
        if (tUIItem == openC4DXMLButton)
        {
            filename = openC4DXMLButton->getSelectedPath();
            size_t spos = filename.find("file://");
            if (spos != std::string::npos)
            {
                filename.erase(spos, 7);
            }

			coTrafficSimulation::instance()->system->parseCinema4dXml(filename);
            std::cout << "Information about road system: " << std::endl << coTrafficSimulation::instance()->system;
        }
        else if (tUIItem == openLandXMLButton)
        {
            filename = openLandXMLButton->getSelectedPath();
            size_t spos = filename.find("file://");
            if (spos != std::string::npos)
            {
                filename.erase(spos, 7);
            }

			coTrafficSimulation::instance()->system->parseLandXml(filename);
            std::cout << "Information about road system: " << std::endl << coTrafficSimulation::instance()->system;
        }
        else if (tUIItem == openIntermapRoadButton)
        {
            filename = openIntermapRoadButton->getSelectedPath();
            size_t spos = filename.find("file://");
            if (spos != std::string::npos)
            {
                filename.erase(spos, 7);
            }

			coTrafficSimulation::instance()->system->parseIntermapRoad(filename, "+proj=latlong +datum=WGS84", "+proj=merc +x_0=-1008832.89 +y_0=-6179385.47");
            //std::cout << "Information about road system: " << std::endl << coTrafficSimulation::instance()->system;
        }

        //roadGroup = new osg::Group;
        //roadGroup->setName(std::string("RoadSystem_")+filename);
		coTrafficSimulation::instance()->roadGroup = new osg::PositionAttitudeTransform;
		coTrafficSimulation::instance()->roadGroup->setName(std::string("RoadSystem_") + filename);
        //roadGroup->setPosition(osg::Vec3d(5.0500000000000000e+05, 5.3950000000000000e+06, 0.0));

        int numRoads = coTrafficSimulation::instance()->system->getNumRoads();
        for (int i = 0; i < numRoads; ++i)
        {
            vehicleUtil::Road *road = coTrafficSimulation::instance()->system->getRoad(i);
            if (true)
            {
                fprintf(stderr, "2tessellateBatters %d\n", coTrafficSimulation::instance()->tessellateBatters);
                osg::Group *roadGroup = road->getRoadBatterGroup(coTrafficSimulation::instance()->tessellateBatters, coTrafficSimulation::instance()->tessellateObjects);
                if (roadGroup)
                {
                    osg::LOD *roadGeodeLOD = new osg::LOD();
                    roadGeodeLOD->addChild(roadGroup, 0.0, 5000.0);

                    roadGroup->addChild(roadGeodeLOD);
                }
            }
        }

        if (coTrafficSimulation::instance()->roadGroup->getNumChildren() > 0)
        {
            //std::cout << "Adding road group: " << roadGroup->getName() << std::endl;
            cover->getObjectsRoot()->addChild(coTrafficSimulation::instance()->roadGroup);
        }

    }

    else if (tUIItem == exportSceneGraphButton)
    {
        std::string filename = exportSceneGraphButton->getSelectedPath();
        size_t spos = filename.find("file://");
        if (spos != std::string::npos)
        {
            filename.erase(spos, 7);
        }
        if (filename.size() != 0)
        {
            osgDB::writeNodeFile(*coTrafficSimulation::instance()->roadGroup, filename);
        }
    }
    else if (tUIItem == loadTerrainButton)
    {
        std::string filename;
        filename = loadTerrainButton->getSelectedPath();
        size_t spos = filename.find("file://");
        if (spos != std::string::npos)
        {
            filename.erase(spos, 7);
        }
        osg::Node *terrain = osgDB::readNodeFile(filename);
        if (terrain)
        {
            osg::StateSet *terrainStateSet = terrain->getOrCreateStateSet();

            osg::PolygonOffset *offset = new osg::PolygonOffset(1.0, 1.0);
            //osg::PolygonOffset* offset = new osg::PolygonOffset(0.0, 0.0);

            terrainStateSet->setAttributeAndModes(offset, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);

            //osgTerrain::TerrainTile::setTileLoadedCallback(new RoadFootprintTileLoadedCallback());

            cover->getObjectsRoot()->addChild(terrain);
        }
    }
    else if (tUIItem == createVehiclesAtMin_Slider)
    {
        createVehiclesAtMin_ = createVehiclesAtMin_Slider->getValue();

        if (createVehiclesAtMin_ >= createVehiclesAtMax_)
        {
            createVehiclesAtMin_ = createVehiclesAtMax_;
        }
		coTrafficSimulation::instance()->min_distance_tui = createVehiclesAtMin_;
        createVehiclesAtMin_Slider->setValue(createVehiclesAtMin_);
    }
    else if (tUIItem == createVehiclesAtMax_Slider)
    {
        createVehiclesAtMax_ = createVehiclesAtMax_Slider->getValue();

        if (createVehiclesAtMax_ <= createVehiclesAtMin_)
        {
            createVehiclesAtMax_ = createVehiclesAtMin_;
        }
		coTrafficSimulation::instance()->max_distance_tui = createVehiclesAtMax_;
        //std::cout << std::endl << " >>> Slider TabletEvent - max_distance_tui: " << max_distance_tui << " <<<" << std::endl <<std::endl;
        createVehiclesAtMax_Slider->setValue(createVehiclesAtMax_);
    }
    else if (tUIItem == minVel_Slider)
    {
        minVel_ = minVel_Slider->getValue();

        if (minVel_ >= maxVel_)
            minVel_ = maxVel_;

        minVel_Slider->setValue(minVel_);
		coTrafficSimulation::instance()->minVel = minVel_;
    }
    else if (tUIItem == maxVel_Slider)
    {
        maxVel_ = maxVel_Slider->getValue();

        if (maxVel_ <= minVel_)
            maxVel_ = minVel_;

        maxVel_Slider->setValue(maxVel_);
		coTrafficSimulation::instance()->maxVel = maxVel_;
    }
    else if (tUIItem == removeVehiclesAtSlider)
    {
        removeVehiclesDelta_ = removeVehiclesAtSlider->getValue();
		coTrafficSimulation::instance()->delete_delta = removeVehiclesDelta_;
        removeVehiclesAtSlider->setValue(removeVehiclesDelta_);
        //std::cout << std::endl << " >>> delete_delta: " << delete_delta << " , delete Distance: " << (delete_delta+TrafficSimulationPlugin::min_distance) <<"m <<<" << std::endl <<std::endl;
    }
    else if (tUIItem == createFreqSlider)
    {
        createFreq_ = createFreqSlider->getValue();
		coTrafficSimulation::instance()->createFreq = createFreq_;
        createFreqSlider->setValue(createFreq_);
        //std::cout << std::endl << " >>> Create new vehicles every  " << createFreq << " Frames <<<" << std::endl <<std::endl;
    }
    else if (tUIItem == td_valueSlider)
    {
        placeholder_ = td_valueSlider->getValue();
		coTrafficSimulation::instance()->placeholder = placeholder_;
        //std::cout << std::endl << " >>> Traffic density on ALL roads: " << td_value << " vehicles/100m <<< placeholder_" << placeholder_ << std::endl <<std::endl;
        //td_valueSlider->setValue(td_value);
        tdField->setText("ON");
        tdValueField->setColor(Qt::darkGreen);
        multiField->setText("OFF");
        tdMultField->setColor(Qt::red);
        //int sliderPos = placeholder_+0.5;
        td_valueSlider->setValue(placeholder_);
    }
    else if (tUIItem == td_multSlider)
    {
        td_mult_ = td_multSlider->getValue();
		coTrafficSimulation::instance()->td_multiplier = td_mult_;
        //std::cout << std::endl << " >>> Traffic density multiplier: " << td_multiplier << " <<<" << std::endl <<std::endl;
        //td_multSlider->setValue(td_multiplier);
        tdField->setText("OFF");
        tdValueField->setColor(Qt::red);
        multiField->setText("ON");
        tdMultField->setColor(Qt::darkGreen);
        placeholder_ = -1;
		coTrafficSimulation::instance()->placeholder = -1;
        td_multSlider->setValue(td_mult_);
    }
    else if (tUIItem == maxVehiclesSlider)
    {
        maxVehicles_ = maxVehiclesSlider->getValue();
		coTrafficSimulation::instance()->maxVehicles = maxVehicles_;
        maxVehiclesSlider->setValue(maxVehicles_);
    }
}

void TrafficSimulationPlugin::tabletPressEvent(coTUIElement *tUIItem)
{
    if (tUIItem == startButton)
    {
		coTrafficSimulation::instance()->runSim = true;
    }
    else if (tUIItem == stopButton)
    {
		coTrafficSimulation::instance()->runSim = false;
    }
    else if (tUIItem == debugRoadButton)
    {
        if (debugRoadButton->getState())
        {

            cover->getObjectsRoot()->addChild(sphereTransform.get());
        }
        else
        {
            cover->getObjectsRoot()->removeChild(sphereTransform.get());
        }
    }

    else if (tUIItem == removeAgentsButton)
    {
        removeAgentsVelocity_ = removeAgentsSlider->getValue();
        VehicleManager::Instance()->removeAllAgents(removeAgentsVelocity_ / 3.6);
    }
    else if (tUIItem == removeAgentsSlider)
    {
        removeAgentsVelocity_ = removeAgentsSlider->getValue();
    }
    else if (tUIItem == useCarpoolButton)
    {

        if ((Carpool::Instance()->getPoolVector()).size() > 0)
        {
            if (useCarpool_ == 0)
            {
                useCarpool_ = 1;
				coTrafficSimulation::instance()->useCarpool = useCarpool_;
                //carpoolStateField->setValue(useCarpool_);
                std::cout << std::endl << " !! Bewegliche Fiddleyards aktiviert !!" << std::endl << std::endl;
                carpoolField->setText("ON");
                carpoolStateField->setColor(Qt::darkGreen);
                //carpoolField->setText("ON");
                //carpoolField->setSize(10,10)
            }
            else
            {
                useCarpool_ = 0;
				coTrafficSimulation::instance()->useCarpool = useCarpool_;
                //carpoolStateField->setValue(useCarpool_);
                std::cout << std::endl << " !! Bewegliche Fiddleyards deaktiviert !!" << std::endl << std::endl;
                carpoolStateField->setColor(Qt::red);
                carpoolField->setText("OFF");
            }
        }
        else
        {
            useCarpool_ = 0;
			coTrafficSimulation::instance()->useCarpool = useCarpool_;
            std::cout << std::endl << " !! Es wurde kein Carpool definiert !!" << std::endl << std::endl;
            carpoolStateField->setColor(Qt::red);
            carpoolField->setText("OFF");
        }
    }
    else if (tUIItem == pluginTab)
    {
        if (useCarpool_ == 0)
            carpoolStateField->setColor(Qt::red);
        else
            carpoolStateField->setColor(Qt::darkGreen);
        if (placeholder_ != -1)
        {
            tdValueField->setColor(Qt::darkGreen);
            tdMultField->setColor(Qt::red);
        }
        else
        {
            tdValueField->setColor(Qt::red);
            tdMultField->setColor(Qt::darkGreen);
        }
        if ((Carpool::Instance()->getPoolVector()).size() == 0)
        {
            carpoolField->setText("OFF");
            carpoolStateField->setColor(Qt::red);
            useCarpool_ = 0;
            coTrafficSimulation::useCarpool = 0;
        }
        else if (useCarpool_ == 1)
        {
            carpoolField->setText("ON");
            carpoolStateField->setColor(Qt::darkGreen);
        }
        else
        {
            carpoolField->setText("OFF");
            carpoolStateField->setColor(Qt::red);
        }
    }
}

void TrafficSimulationPlugin::key(int type, int keySym, int mod)
{
    if (type == osgGA::GUIEventAdapter::KEYDOWN)
    {
        if (keySym == 108 && mod == 0)
        {
			VehicleManager::Instance()->switchToNextCamera();
        }
        else if (keySym == 76 && mod == 2)
        {
			VehicleManager::Instance()->switchToPreviousCamera();
        }
        else if (keySym == 12 && mod == 8)
        {
			VehicleManager::Instance()->unbindCamera();
        }
        //else if (keySym == 101 && mod == 0)
        //{
        //    UDPBroadcast::errorStatus_TS();
        //}

        /*else if(keySym==98 && mod==0) {
         VehicleManager::Instance()->brakeCameraVehicle();
      }*/
    }
}

COVERPLUGIN(TrafficSimulationPlugin)
