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

#include "TrafficSimulation.h"

#include "FindTrafficLightSwitch.h"

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
#include "HumanVehicle.h"

#include "PorscheFFZ.h"
#include <RoadTerrain/RoadTerrainLoader.h>

#include <functional>

using namespace covise;
using namespace opencover;
using namespace vehicleUtil;

int TrafficSimulation::TrafficSimulation::counter = 0;
int TrafficSimulation::TrafficSimulation::createFreq = 20;
double TrafficSimulation::TrafficSimulation::min_distance = 800;
int TrafficSimulation::TrafficSimulation::delete_delta = 50;
double TrafficSimulation::TrafficSimulation::delete_at = 230;
//int TrafficSimulation::TrafficSimulation::min_distance_50 = 180;
//int TrafficSimulation::TrafficSimulation::min_distance_100 = 800;
int TrafficSimulation::TrafficSimulation::useCarpool = 1;
float TrafficSimulation::TrafficSimulation::td_multiplier = 1.0;
float TrafficSimulation::TrafficSimulation::placeholder = -1;
int TrafficSimulation::TrafficSimulation::minVel = 50;
int TrafficSimulation::TrafficSimulation::maxVel = 100;
int TrafficSimulation::TrafficSimulation::min_distance_tui = 180;
int TrafficSimulation::TrafficSimulation::max_distance_tui = 800;
int TrafficSimulation::TrafficSimulation::maxVehicles = 0;

TrafficSimulation::TrafficSimulation::TrafficSimulation()
    : coVRPlugin(COVER_PLUGIN_NAME)
    , system(NULL)
    , manager(NULL)
    , pedestrianManager(NULL)
    , factory(NULL)
    , roadGroup(NULL)
    , rootElement(NULL)
    ,
    //operatorMapTab(NULL),
    //operatorMap(NULL),
    ffzBroadcaster(NULL)
    , runSim(true)
    , tessellateRoads(true)
    , tessellatePaths(true)
    , tessellateBatters(false)
    , tessellateObjects(false)
    , terrain(NULL)
    , mersenneTwisterEngine((int)cover->frameTime() * 1000)
{
    //srand ( (int)(cover->frameTime()*1000) );
    
        manager = VehicleManager::Instance();

        pedestrianManager = PedestrianManager::Instance();
}

TrafficSimulation::TrafficSimulation *TrafficSimulation::TrafficSimulation::instance()
{
    static TrafficSimulation *singleton = NULL;
    if (!singleton)
        singleton = new TrafficSimulation;
    return singleton;
}


void TrafficSimulation::TrafficSimulation::runSimulation()
{
    runSim = true;
}

void TrafficSimulation::TrafficSimulation::haltSimulation()
{
    runSim = false;
}

unsigned long TrafficSimulation::TrafficSimulation::getIntegerRandomNumber()
{
    return mersenneTwisterEngine();
}

double TrafficSimulation::TrafficSimulation::getZeroOneRandomNumber()
{
    auto r = std::bind(uniformDist, mersenneTwisterEngine);
    return r();
}

TrafficSimulation::VehicleManager *TrafficSimulation::TrafficSimulation::getVehicleManager()
{
    return manager;
}

TrafficSimulation::PedestrianManager *TrafficSimulation::TrafficSimulation::getPedestrianManager()
{
    return pedestrianManager;
}

xercesc::DOMElement *TrafficSimulation::TrafficSimulation::getOpenDriveRootElement(std::string filename)
{
    try
    {
        xercesc::XMLPlatformUtils::Initialize();
    }
    catch (const xercesc::XMLException &toCatch)
    {
        char *message = xercesc::XMLString::transcode(toCatch.getMessage());
        std::cout << "Error during initialization! :\n" << message << std::endl;
        xercesc::XMLString::release(&message);
        return NULL;
    }

    xercesc::XercesDOMParser *parser = new xercesc::XercesDOMParser();
    parser->setValidationScheme(xercesc::XercesDOMParser::Val_Never);

    try
    {
        parser->parse(filename.c_str());
    }
    catch (...)
    {
        std::cerr << "Couldn't parse OpenDRIVE XML-file " << filename << "!" << std::endl;
    }

    xercesc::DOMDocument *xmlDoc = parser->getDocument();
    if (xmlDoc)
    {
        rootElement = xmlDoc->getDocumentElement();
    }

    return rootElement;
}

void TrafficSimulation::TrafficSimulation::parseOpenDrive(xercesc::DOMElement *rootElement)
{
	XMLCh *t1 = NULL, *t2 = NULL, *t3 = NULL, *t4 = NULL, *t5 = NULL;
	char *ch;
    xercesc::DOMNodeList *documentChildrenList = rootElement->getChildNodes();

    for (int childIndex = 0; childIndex < documentChildrenList->getLength(); ++childIndex)
    {
        xercesc::DOMElement *sceneryElement = dynamic_cast<xercesc::DOMElement *>(documentChildrenList->item(childIndex));
        if (sceneryElement && xercesc::XMLString::compareIString(sceneryElement->getTagName(), t1 = xercesc::XMLString::transcode("scenery")) == 0)
        {
            std::string fileString = ch = xercesc::XMLString::transcode(sceneryElement->getAttribute(t2 = xercesc::XMLString::transcode("file"))); xercesc::XMLString::release(&t2); xercesc::XMLString::release(&ch);
            std::string vpbString = ch = xercesc::XMLString::transcode(sceneryElement->getAttribute(t2 = xercesc::XMLString::transcode("vpb"))); xercesc::XMLString::release(&t2); xercesc::XMLString::release(&ch);

            std::vector<BoundingArea> voidBoundingAreaVector;
            std::vector<std::string> shapeFileNameVector;

            xercesc::DOMNodeList *sceneryChildrenList = sceneryElement->getChildNodes();
            xercesc::DOMElement *sceneryChildElement;
            for (unsigned int childIndex = 0; childIndex < sceneryChildrenList->getLength(); ++childIndex)
            {
                sceneryChildElement = dynamic_cast<xercesc::DOMElement *>(sceneryChildrenList->item(childIndex));
                if (!sceneryChildElement)
                    continue;

                if (xercesc::XMLString::compareIString(sceneryChildElement->getTagName(), t4 = xercesc::XMLString::transcode("void")) == 0)
                {
                    double xMin = atof(ch = xercesc::XMLString::transcode(sceneryChildElement->getAttribute(t3 = xercesc::XMLString::transcode("xMin")))); xercesc::XMLString::release(&ch); xercesc::XMLString::release(&t3);
                    double yMin = atof(ch = xercesc::XMLString::transcode(sceneryChildElement->getAttribute(t3 = xercesc::XMLString::transcode("yMin")))); xercesc::XMLString::release(&ch); xercesc::XMLString::release(&t3);
                    double xMax = atof(ch = xercesc::XMLString::transcode(sceneryChildElement->getAttribute(t3 = xercesc::XMLString::transcode("xMax")))); xercesc::XMLString::release(&ch); xercesc::XMLString::release(&t3);
                    double yMax = atof(ch = xercesc::XMLString::transcode(sceneryChildElement->getAttribute(t3 = xercesc::XMLString::transcode("yMax")))); xercesc::XMLString::release(&ch); xercesc::XMLString::release(&t3);

                    voidBoundingAreaVector.push_back(BoundingArea(osg::Vec2(xMin, yMin), osg::Vec2(xMax, yMax)));
                    //voidBoundingAreaVector.push_back(BoundingArea(osg::Vec2(506426.839,5398055.357),osg::Vec2(508461.865,5399852.0)));
                }
                else if (xercesc::XMLString::compareIString(sceneryChildElement->getTagName(), t5 = xercesc::XMLString::transcode("shape")) == 0)
                {
                    std::string fileString = ch = xercesc::XMLString::transcode(sceneryChildElement->getAttribute(t3 = xercesc::XMLString::transcode("file"))); xercesc::XMLString::release(&ch); xercesc::XMLString::release(&t3);
                    shapeFileNameVector.push_back(fileString);
                }
				xercesc::XMLString::release(&t4);
				xercesc::XMLString::release(&t5);
            }

            if (!fileString.empty())
            {
                if (!coVRFileManager::instance()->fileExist((xodrDirectory + "/" + fileString).c_str()))
                {
                    std::cerr << "\n#\n# file not found: this may lead to a crash! \n#" << endl;
                }
                coVRFileManager::instance()->loadFile((xodrDirectory + "/" + fileString).c_str());
            }

            if (!vpbString.empty())
            {
                fprintf(stderr, "loading %s\n", vpbString.c_str());
                if (RoadTerrainLoader::instance())
                {
                    osg::Vec3d offset(0, 0, 0);
                    const RoadSystemHeader &header = RoadSystem::Instance()->getHeader();
                    offset.set(header.xoffset, header.yoffset, 0.0);
                    fprintf(stderr, "loading %s offset: %f %f\n", (xodrDirectory + "/" + vpbString).c_str(), offset[0], offset[1]);
                    RoadTerrainLoader::instance()->loadTerrain(xodrDirectory + "/" + vpbString, offset, voidBoundingAreaVector, shapeFileNameVector);
                }
            }
        }
        else if (sceneryElement && xercesc::XMLString::compareIString(sceneryElement->getTagName(), t2 = xercesc::XMLString::transcode("environment")) == 0)
        {
            std::string tessellateRoadsString = ch = xercesc::XMLString::transcode(sceneryElement->getAttribute(t3 = xercesc::XMLString::transcode("tessellateRoads"))); xercesc::XMLString::release(&ch); xercesc::XMLString::release(&t3);
            if (tessellateRoadsString == "false" || tessellateRoadsString == "0")
            {
                tessellateRoads = false;
            }
            else
            {
                tessellateRoads = true;
            }

            std::string tessellatePathsString = ch = xercesc::XMLString::transcode(sceneryElement->getAttribute(t3 = xercesc::XMLString::transcode("tessellatePaths"))); xercesc::XMLString::release(&ch); xercesc::XMLString::release(&t3);
            if (tessellatePathsString == "false" || tessellatePathsString == "0")
            {
                tessellatePaths = false;
            }
            else
            {
                tessellatePaths = true;
            }

            std::string tessellateBattersString = ch = xercesc::XMLString::transcode(sceneryElement->getAttribute(t3 = xercesc::XMLString::transcode("tessellateBatters"))); xercesc::XMLString::release(&ch); xercesc::XMLString::release(&t3);
            if (tessellateBattersString == "true")
            {
                tessellateBatters = true;
            }
            else
            {
                tessellateBatters = false;
            }

            std::string tessellateObjectsString = ch = xercesc::XMLString::transcode(sceneryElement->getAttribute(t3 = xercesc::XMLString::transcode("tessellateObjects"))); xercesc::XMLString::release(&ch); xercesc::XMLString::release(&t3);
            if (tessellateObjectsString == "true")
            {
                tessellateObjects = true;
            }
            else
            {
                tessellateObjects = false;
            }
        }
		xercesc::XMLString::release(&t1);
		xercesc::XMLString::release(&t2);
    }
}

bool TrafficSimulation::TrafficSimulation::loadRoadSystem(const char *filename_chars)
{
    std::string filename(filename_chars);
    std::cerr << "Loading road system!" << std::endl;
    if (system == NULL)
    {
        //Building directory string to xodr file
        xodrDirectory.clear();
        if (filename[0] != '/' && filename[0] != '\\' && (!(filename[1] == ':' && (filename[2] == '/' || filename[2] == '\\'))))
        { // / or backslash or c:/
            char *workingDir = getcwd(NULL, 0);
            xodrDirectory.assign(workingDir);
            free(workingDir);
        }
        size_t lastSlashPos = filename.find_last_of('/');
        size_t lastSlashPos2 = filename.find_last_of('\\');
        if (lastSlashPos != filename.npos && (lastSlashPos2 == filename.npos || lastSlashPos2 < lastSlashPos))
        {
            if (!xodrDirectory.empty())
                xodrDirectory += "/";
            xodrDirectory.append(filename, 0, lastSlashPos);
        }
        if (lastSlashPos2 != filename.npos && (lastSlashPos == filename.npos || lastSlashPos < lastSlashPos2))
        {
            if (!xodrDirectory.empty())
                xodrDirectory += "\\";
            xodrDirectory.append(filename, 0, lastSlashPos2);
        }

        system = RoadSystem::Instance();

        xercesc::DOMElement *openDriveElement = getOpenDriveRootElement(filename);
        if (!openDriveElement)
        {
            std::cerr << "No regular xodr file " << filename << " at: " + xodrDirectory << std::endl;
            return false;
        }

        system->parseOpenDrive(openDriveElement);
        this->parseOpenDrive(rootElement);

        factory = VehicleFactory::Instance();
        factory->parseOpenDrive(openDriveElement, xodrDirectory);

        pedestrianFactory = PedestrianFactory::Instance();
        pedestrianFactory->parseOpenDrive(openDriveElement, xodrDirectory);

        //system->parseOpenDrive(filename);
        //std::cout << "Information about road system: " << std::endl << system;

        //roadGroup = new osg::Group;
        roadGroup = new osg::PositionAttitudeTransform;
        roadGroup->setName("RoadSystem");
		roadGroup->setNodeMask(roadGroup->getNodeMask() & ~opencover::Isect::Update); // don't use the update traversal
        //roadGroup->setPosition(osg::Vec3d(5.0500000000000000e+05, 5.3950000000000000e+06, 0.0));
        //roadGroup->setPosition(osg::Vec3d(960128.3125, 6158421.5, 0.0));

        //osg::Material* roadGroupMaterial = new osg::Material;
        //roadGroupMaterial->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
        //roadGroupState->setAttribute(roadGroupMaterial);

        int numRoads = system->getNumRoads();
        for (int i = 0; i < numRoads; ++i)
        {
            vehicleUtil::Road *road = system->getRoad(i);
            osg::LOD *roadGeodeLOD = new osg::LOD();

            // Tesselation //
            //
            if ((!road->isJunctionPath() && tessellateRoads == true) // normal road
                || (road->isJunctionPath() && tessellatePaths == true) // junction path
                )
            {
                fprintf(stderr, "1tessellateBatters %d\n", tessellateBatters);
                osg::Group *roadGroup = road->getRoadBatterGroup(tessellateBatters, tessellateObjects);
                if (roadGroup)
                {
                    roadGeodeLOD->addChild(roadGroup, 0.0, 5000.0);
                }
            }

            osg::Group *roadObjectsGroup = road->createObjectsGroup();
            if (roadObjectsGroup->getNumChildren() > 0)
            {
                roadGeodeLOD->addChild(roadObjectsGroup, 0.0, 5000.0);
            }

            osg::Geode *guardRailGeode = road->getGuardRailGeode();
            if (guardRailGeode)
            {
                roadGeodeLOD->addChild(guardRailGeode, 0.0, 5000.0);
            }

            if (roadGeodeLOD->getNumChildren() > 0)
            {
                roadGroup->addChild(roadGeodeLOD);
            }
        }

        /*if(tessellatePaths==true) 
      {
         unsigned int numJunctions = system->getNumJunctions();
         for(int i=0; i<numJunctions; ++i) {
            osg::Geode* junctionGeode = system->getJunction(i)->getJunctionGeode();
            if(junctionGeode) {
               roadGroup->addChild(junctionGeode);
            }
         }
      }*/

        if (roadGroup->getNumChildren() > 0)
        {
            cover->getObjectsRoot()->addChild(roadGroup);
        }

		trafficSignalGroup = new osg::Group;
        trafficSignalGroup->setName("TrafficSignals");
		trafficSignalGroup->setNodeMask(trafficSignalGroup->getNodeMask() & ~opencover::Isect::Update); // don't use the update traversal
        //Traffic control
        for (int i = 0; i < system->getNumRoadSignals(); ++i)
        {
            RoadSignal *signal = system->getRoadSignal(i);
            TrafficLightSignal *trafficLightSignal = dynamic_cast<TrafficLightSignal *>(signal);

            if (trafficLightSignal)
            {
                FindTrafficLightSwitch findSwitch(trafficLightSignal->getName());
                osg::PositionAttitudeTransform *trafficSignalNode = trafficLightSignal->getRoadSignalNode();
                if (trafficSignalNode)
                {
                    trafficSignalGroup->addChild(trafficSignalNode);

                    //findSwitch.traverse(*cover->getObjectsXform());
                    trafficSignalNode->accept(findSwitch);
                    TrafficLightSignalTurnCallback *callbackGreen
                        = new TrafficLightSignalTurnCallback(findSwitch.getMultiSwitchGreen());
                    trafficLightSignal->setSignalGreenCallback(callbackGreen);

                    TrafficLightSignalTurnCallback *callbackYellow
                        = new TrafficLightSignalTurnCallback(findSwitch.getMultiSwitchYellow());
                    trafficLightSignal->setSignalYellowCallback(callbackYellow);

                    TrafficLightSignalTurnCallback *callbackRed
                        = new TrafficLightSignalTurnCallback(findSwitch.getMultiSwitchRed());
                    trafficLightSignal->setSignalRedCallback(callbackRed);
                }
            }
            else
            {
                osg::PositionAttitudeTransform *roadSignalNode = signal->getRoadSignalNode();
                if (roadSignalNode)
                {
                    trafficSignalGroup->addChild(roadSignalNode);
                    osg::PositionAttitudeTransform *roadSignalPost = signal->getRoadSignalPost();
                    if (roadSignalPost)
                    {
                        trafficSignalGroup->addChild(roadSignalPost);
                    }
                    
                }
            }
        }

        if (trafficSignalGroup->getNumChildren() > 0)
        {
            cover->getObjectsRoot()->addChild(trafficSignalGroup);
        }

    }

    if ((Carpool::Instance()->getPoolVector()).size() > 0)
        system->scanStreets(); // 29.08.2011

    /*
         std::cout << " ----------------------- TrafficSimulationPlugin -----------------------" << std::endl;
         for (int i=0; i<RoadSystem::_tiles_y+1;i++) {
         for (int j=0; j<RoadSystem::_tiles_x+1;j++) {
             std::cout << (system->getRLS_List(j,i)).size();
         }
         std::cout << std::endl;
        }*/
    /*Carpool * carpool = Carpool::Instance();
        std::vector<Pool*> poolVector = carpool->getPoolVector();
      //std::cout << " ----------->> poolVector.size(): " << poolVector.size() << std::endl;
        std::vector<Pool*>::iterator it;
        for (it = poolVector.begin(); it != poolVector.end(); it++){
        std::cout << " ----------->> (*it)->getName(): " << (*it)->getName() << " (*it)->getId(): " << (*it)->getId() << " Anzahl FFZ: " << (*it)->getMapSize() << std::endl;
 //              std::cout << " ----------->> (*it)->getRepeatTime(): " << (*it)->getRepeatTime() << std::endl;
        }
        */

    if (coVRMSController::instance()->isMaster() && (Carpool::Instance()->getPoolVector()).size() > 0)
    {
        if (RoadSystem::_tiles_y <= 400 && RoadSystem::_tiles_x <= 400)
        {
            //for (int i=0; i<=_tiles_y;i++) {
            for (int i = RoadSystem::_tiles_y; i >= 0; i--)
            {
                for (int j = 0; j <= RoadSystem::_tiles_x; j++)
                {
                    if ((system->getRLS_List(j, i)).size() == 0)
                        std::cout << "-";
                    else
                        std::cout << (system->getRLS_List(j, i)).size();
                }
                std::cout << std::endl;
            }
        }
        /*std::cout << "... delta x: " << RoadSystem::delta_x << " | delta y: " << RoadSystem::delta_y << std::endl << " _tiles_x " << RoadSystem::_tiles_x  << " _tiles_y " << RoadSystem::_tiles_y << std::endl;*/
    }

    return true;
}

void TrafficSimulation::TrafficSimulation::deleteRoadSystem()
{

    system = NULL;
    RoadSystem::Destroy();
	PedestrianFactory::Destroy();
    if (roadGroup)
    {
        while (roadGroup->getNumParents())
        {
            roadGroup->getParent(0)->removeChild(roadGroup);
        }
    }
	if (trafficSignalGroup)
	{
		while (trafficSignalGroup->getNumParents())
		{
			trafficSignalGroup->getParent(0)->removeChild(trafficSignalGroup);
		}
	}
	if (terrain)
	{
		while (terrain->getNumParents())
		{
			terrain->getParent(0)->removeChild(terrain);
		}
	}
	
}

bool TrafficSimulation::TrafficSimulation::init()
{

    cover->setScale(1000);

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

    //Timer fuer die FFZ-Erstellung
    createFreqLabel = new coTUILabel("Create Vehicles every x Frames:", createFrame->getID());
    createFreqLabel->setPos(0, pos);

    /*createFreqButton = new coTUIButton("Create Vehicles every x Frames:", pluginTab->getID());
 *         createFreqButton->setEventListener(this);
 *                 createFreqButton->setPos(0,pos);*/
    createFreq_ = TrafficSimulation::TrafficSimulation::createFreq;
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

    //Fahrzeuge loeschen
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

    //Faktor fuer die Verkehrsdichte
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

// UDP Broadcast //
//
// sends positions of vehicles e.g. to Porsche dSPACE //
#if 1
    double sendFrequency = (double)coCoviseConfig::getFloat("sendFrequency", "COVER.Plugin.TrafficSimulation.PorscheFFZ", 60.0f);

    // dSPACE //
    //
    std::string dSpaceIp = coCoviseConfig::getEntry("destinationIP", "COVER.Plugin.TrafficSimulation.PorscheFFZ.DSPACE");
    int dSpacePort = coCoviseConfig::getInt("port", "COVER.Plugin.TrafficSimulation.PorscheFFZ.DSPACE", 52002);
    int dSpaceLocalPort = coCoviseConfig::getInt("localPort", "COVER.Plugin.TrafficSimulation.PorscheFFZ.DSPACE", 52002);

    // KMS //
    //
    std::string kmsIp = coCoviseConfig::getEntry("destinationIP", "COVER.Plugin.TrafficSimulation.PorscheFFZ.KMS");
    int kmsPort = coCoviseConfig::getInt("port", "COVER.Plugin.TrafficSimulation.PorscheFFZ.KMS", 52002);
    int kmsLocalPort = coCoviseConfig::getInt("localPort", "COVER.Plugin.TrafficSimulation.PorscheFFZ.KMS", 52002);

    if (!dSpaceIp.empty() && !kmsIp.empty() && sendFrequency != 0.0)
    {
        ffzBroadcaster = new PorscheFFZ(sendFrequency);
        if (coVRMSController::instance()->isMaster())
        {
            ffzBroadcaster->setupDSPACE(dSpaceIp, dSpacePort, dSpaceLocalPort);
            ffzBroadcaster->setupKMS(kmsIp, kmsPort, kmsLocalPort);
        }
    }

#endif
    sphereTransform = new osg::MatrixTransform();
	sphereTransform->setName("TSDebugSphere");
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

    return true;
}

void
TrafficSimulation::TrafficSimulation::preFrame()
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
    double dt = cover->frameDuration();
    VehicleList vehOverallList = VehicleManager::Instance()->getVehicleOverallList();
    osg::Vec2d incoming_pos = HumanVehicle::human_pos; // Robert
    double v = (double)RoadSystem::dSpace_v;
    //Loeschdistanz ermitteln
    if (v <= minVel)
        min_distance = min_distance_tui;
    else if (v >= maxVel)
        min_distance = max_distance_tui;
    else
        min_distance = (min_distance_tui + ((max_distance_tui - min_distance_tui) / (maxVel - minVel)) * (v - minVel));
    delete_at = min_distance + delete_delta;

    // Ermitteln in welcher Kachel sich das Eigenfahrzeug befinden wuerde
    if ((Carpool::Instance()->getPoolVector()).size() > 0 && useCarpool == 1)
    { //Ausfuehrung nur falls mindestens ein pool definiert wurde
        osg::Vec2d current_tile_y = system->get_tile(incoming_pos[0], incoming_pos[1]);
        std::list<Vehicle *>::iterator it;
        for (it = vehOverallList.begin(); it != vehOverallList.end(); ++it)
        {
            if (it != vehOverallList.begin())
            {
                double distance = (*it)->getSquaredDistanceTo(Vector3D(incoming_pos[0], incoming_pos[1], 0.0));
                distance = sqrt(distance);
                if (distance > delete_at)
                {
                    //if(coVRMSController::instance()->isMaster()) {
                    //      std::cout << "xxx Distance: " << distance << " Removed Vehicle " << (*it)->getVehicleID() << /*" on Road " << (*it)->getRoad()->getId() <<*/std::endl;
                    //}*/

                    TrafficSimulation::TrafficSimulation::instance()->getVehicleManager()->removeVehicle((*it), (*it)->getRoad());
                }
            }
        }
        if (counter >= createFreq)
        {
            if (maxVehicles == 0)
                factory->createTileVehicle(current_tile_y[0], current_tile_y[1], manager->acitve_fiddleyards);
            else if (((VehicleManager::Instance()->getVehicleOverallList()).size()) < maxVehicles)
            {
                factory->createTileVehicle(current_tile_y[0], current_tile_y[1], manager->acitve_fiddleyards);
            }
            counter = 0;
        }
        counter++;
    }

    if (runSim)
    {
        if (system)
        {
            system->update(dt);
        }

        if (manager)
        {
            manager->moveAllVehicles(dt);
            manager->updateFiddleyards(dt, incoming_pos);
            if (coVRMSController::instance()->isMaster())
            {
                //if(operatorMap) {
                //   // send positions of the vehicles to operator map
                //   manager->sendDataTo(operatorMap);
                //}
                if (ffzBroadcaster)
                {
                    // send positions of nearby vehicles via UDP Broadcast
                    manager->sendDataTo(ffzBroadcaster);
                }
            }
            if (ffzBroadcaster)
            {
                // receive and parse Data from UDP Broadcast
                manager->receiveDataFrom(ffzBroadcaster);
            }
        }
        if (pedestrianManager)
        {
            pedestrianManager->moveAllPedestrians(dt);
            pedestrianManager->updateFiddleyards(dt);
        }
    }
}

void TrafficSimulation::TrafficSimulation::tabletEvent(coTUIElement *tUIItem)
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

        system->writeOpenDrive(filename);
    }

    else if (tUIItem == openC4DXMLButton || tUIItem == openLandXMLButton || tUIItem == openIntermapRoadButton)
    {
        if (system == NULL)
        {
            system = RoadSystem::Instance();
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

            system->parseCinema4dXml(filename);
            std::cout << "Information about road system: " << std::endl << system;
        }
        else if (tUIItem == openLandXMLButton)
        {
            filename = openLandXMLButton->getSelectedPath();
            size_t spos = filename.find("file://");
            if (spos != std::string::npos)
            {
                filename.erase(spos, 7);
            }

            system->parseLandXml(filename);
            std::cout << "Information about road system: " << std::endl << system;
        }
        else if (tUIItem == openIntermapRoadButton)
        {
            filename = openIntermapRoadButton->getSelectedPath();
            size_t spos = filename.find("file://");
            if (spos != std::string::npos)
            {
                filename.erase(spos, 7);
            }

            system->parseIntermapRoad(filename, "+proj=latlong +datum=WGS84", "+proj=merc +x_0=-1008832.89 +y_0=-6179385.47");
            //std::cout << "Information about road system: " << std::endl << system;
        }

        //roadGroup = new osg::Group;
        //roadGroup->setName(std::string("RoadSystem_")+filename);
        roadGroup = new osg::PositionAttitudeTransform;
        roadGroup->setName(std::string("RoadSystem_") + filename);
        //roadGroup->setPosition(osg::Vec3d(5.0500000000000000e+05, 5.3950000000000000e+06, 0.0));

        int numRoads = system->getNumRoads();
        for (int i = 0; i < numRoads; ++i)
        {
            vehicleUtil::Road *road = system->getRoad(i);
            if (true)
            {
                fprintf(stderr, "2tessellateBatters %d\n", tessellateBatters);
                osg::Group *roadGroup = road->getRoadBatterGroup(tessellateBatters, tessellateObjects);
                if (roadGroup)
                {
                    osg::LOD *roadGeodeLOD = new osg::LOD();
                    roadGeodeLOD->addChild(roadGroup, 0.0, 5000.0);

                    roadGroup->addChild(roadGeodeLOD);
                }
            }
        }

        if (roadGroup->getNumChildren() > 0)
        {
            //std::cout << "Adding road group: " << roadGroup->getName() << std::endl;
            cover->getObjectsRoot()->addChild(roadGroup);
        }

        if (!manager)
        {
            manager = VehicleManager::Instance();
        }
        if (!pedestrianManager)
        {
            pedestrianManager = PedestrianManager::Instance();
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
            osgDB::writeNodeFile(*roadGroup, filename);
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
        terrain = osgDB::readNodeFile(filename);
        if (terrain)
        {
            osg::StateSet *terrainStateSet = terrain->getOrCreateStateSet();

            osg::PolygonOffset *offset = new osg::PolygonOffset(1.0, 1.0);
            //osg::PolygonOffset* offset = new osg::PolygonOffset(0.0, 0.0);

            terrainStateSet->setAttributeAndModes(offset, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);

            //osgTerrain::TerrainTile::setTileLoadedCallback(new RoadFootprintTileLoadedCallback());
			if (terrain->getName() == "")
			{
				terrain->setName("TS_Terrain");
			}
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
        TrafficSimulation::TrafficSimulation::min_distance_tui = createVehiclesAtMin_;
        createVehiclesAtMin_Slider->setValue(createVehiclesAtMin_);
    }
    else if (tUIItem == createVehiclesAtMax_Slider)
    {
        createVehiclesAtMax_ = createVehiclesAtMax_Slider->getValue();

        if (createVehiclesAtMax_ <= createVehiclesAtMin_)
        {
            createVehiclesAtMax_ = createVehiclesAtMin_;
        }
        TrafficSimulation::TrafficSimulation::max_distance_tui = createVehiclesAtMax_;
        //std::cout << std::endl << " >>> Slider TabletEvent - max_distance_tui: " << max_distance_tui << " <<<" << std::endl <<std::endl;
        createVehiclesAtMax_Slider->setValue(createVehiclesAtMax_);
    }
    else if (tUIItem == minVel_Slider)
    {
        minVel_ = minVel_Slider->getValue();

        if (minVel_ >= maxVel_)
            minVel_ = maxVel_;

        minVel_Slider->setValue(minVel_);
        TrafficSimulation::TrafficSimulation::minVel = minVel_;
    }
    else if (tUIItem == maxVel_Slider)
    {
        maxVel_ = maxVel_Slider->getValue();

        if (maxVel_ <= minVel_)
            maxVel_ = minVel_;

        maxVel_Slider->setValue(maxVel_);
        TrafficSimulation::TrafficSimulation::maxVel = maxVel_;
    }
    else if (tUIItem == removeVehiclesAtSlider)
    {
        removeVehiclesDelta_ = removeVehiclesAtSlider->getValue();
        TrafficSimulation::TrafficSimulation::delete_delta = removeVehiclesDelta_;
        removeVehiclesAtSlider->setValue(removeVehiclesDelta_);
        //std::cout << std::endl << " >>> delete_delta: " << delete_delta << " , delete Distance: " << (delete_delta+TrafficSimualtion::min_distance) <<"m <<<" << std::endl <<std::endl;
    }
    else if (tUIItem == createFreqSlider)
    {
        createFreq_ = createFreqSlider->getValue();
        TrafficSimulation::TrafficSimulation::createFreq = createFreq_;
        createFreqSlider->setValue(createFreq_);
        //std::cout << std::endl << " >>> Create new vehicles every  " << createFreq << " Frames <<<" << std::endl <<std::endl;
    }
    else if (tUIItem == td_valueSlider)
    {
        placeholder_ = td_valueSlider->getValue();
        TrafficSimulation::TrafficSimulation::placeholder = placeholder_;
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
        TrafficSimulation::TrafficSimulation::td_multiplier = td_mult_;
        //std::cout << std::endl << " >>> Traffic density multiplier: " << td_multiplier << " <<<" << std::endl <<std::endl;
        //td_multSlider->setValue(td_multiplier);
        tdField->setText("OFF");
        tdValueField->setColor(Qt::red);
        multiField->setText("ON");
        tdMultField->setColor(Qt::darkGreen);
        placeholder_ = -1;
        TrafficSimulation::TrafficSimulation::placeholder = -1;
        td_multSlider->setValue(td_mult_);
    }
    else if (tUIItem == maxVehiclesSlider)
    {
        maxVehicles_ = maxVehiclesSlider->getValue();
        TrafficSimulation::TrafficSimulation::maxVehicles = maxVehicles_;
        maxVehiclesSlider->setValue(maxVehicles_);
    }
}

void TrafficSimulation::TrafficSimulation::tabletPressEvent(coTUIElement *tUIItem)
{
    if (tUIItem == startButton)
    {
        runSim = true;
    }
    else if (tUIItem == stopButton)
    {
        runSim = false;
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
        if (manager)
            manager->removeAllAgents(removeAgentsVelocity_ / 3.6);
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
                TrafficSimulation::TrafficSimulation::useCarpool = useCarpool_;
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
                TrafficSimulation::TrafficSimulation::useCarpool = useCarpool_;
                //carpoolStateField->setValue(useCarpool_);
                std::cout << std::endl << " !! Bewegliche Fiddleyards deaktiviert !!" << std::endl << std::endl;
                carpoolStateField->setColor(Qt::red);
                carpoolField->setText("OFF");
            }
        }
        else
        {
            useCarpool_ = 0;
            TrafficSimulation::TrafficSimulation::useCarpool = useCarpool_;
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
            TrafficSimulation::TrafficSimulation::useCarpool = 0;
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

void TrafficSimulation::TrafficSimulation::key(int type, int keySym, int mod)
{
    if (type == osgGA::GUIEventAdapter::KEYDOWN)
    {
        if (keySym == 108 && mod == 0)
        {
            manager->switchToNextCamera();
        }
        else if (keySym == 76 && mod == 2)
        {
            manager->switchToPreviousCamera();
        }
        else if (keySym == 12 && mod == 8)
        {
            manager->unbindCamera();
        }
        else if (keySym == 101 && mod == 0)
        {
            UDPBroadcast::errorStatus_TS();
        }

        /*else if(keySym==98 && mod==0) {
         manager->brakeCameraVehicle();
      }*/
    }
}

