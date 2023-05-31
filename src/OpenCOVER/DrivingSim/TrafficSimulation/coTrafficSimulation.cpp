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

#include "coTrafficSimulation.h"

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
#ifdef WIN32
#include <direct.h>
#endif

using namespace covise;
using namespace opencover;
using namespace vehicleUtil;
using namespace TrafficSimulation;

int coTrafficSimulation::counter = 0;
int coTrafficSimulation::createFreq = 20;
double coTrafficSimulation::min_distance = 800;
int coTrafficSimulation::delete_delta = 50;
double coTrafficSimulation::delete_at = 230;
//int coTrafficSimulation::min_distance_50 = 180;
//int coTrafficSimulation::min_distance_100 = 800;
int coTrafficSimulation::useCarpool = 1;
float coTrafficSimulation::td_multiplier = 1.0;
float coTrafficSimulation::placeholder = -1;
int coTrafficSimulation::minVel = 50;
int coTrafficSimulation::maxVel = 100;
int coTrafficSimulation::min_distance_tui = 180;
int coTrafficSimulation::max_distance_tui = 800;
int coTrafficSimulation::maxVehicles = 0;
int coTrafficSimulation::numInstances = 0;

coTrafficSimulation *coTrafficSimulation::instance()
{
	if (myInstance == NULL)
	{
		myInstance = new coTrafficSimulation();
	}
	return myInstance;
}

void coTrafficSimulation::useInstance()
{
	numInstances++;
}
void coTrafficSimulation::freeInstance()
{
	numInstances--;
	if (numInstances == 0)
		delete myInstance;
	myInstance = 0;
}

coTrafficSimulation::coTrafficSimulation()
    : coVRPlugin(COVER_PLUGIN_NAME)
	, system(NULL)
	, factory(NULL)
	, roadGroup(NULL)
	, rootElement(NULL)
	, trafficSignalGroup(NULL)
	, terrain(NULL)
    ,
    //operatorMapTab(NULL),
    //operatorMap(NULL),
    ffzBroadcaster(NULL)
    , runSim(true)
    , tessellateRoads(true)
    , tessellatePaths(true)
    , tessellateBatters(false)
    , tessellateObjects(false)
    , mersenneTwisterEngine((int)cover->frameTime() * 1000)
{
    //srand ( (int)(cover->frameTime()*1000) );
}

coTrafficSimulation *coTrafficSimulation::myInstance = NULL;


// this is called if the plugin is removed at runtime
coTrafficSimulation::~coTrafficSimulation()
{
    deleteRoadSystem();

    VehicleManager::Destroy();
    PedestrianManager::Destroy();

    delete ffzBroadcaster;

}

void coTrafficSimulation::runSimulation()
{
    runSim = true;
}

void coTrafficSimulation::haltSimulation()
{
    runSim = false;
}

unsigned long coTrafficSimulation::getIntegerRandomNumber()
{
    return mersenneTwisterEngine();
}

double coTrafficSimulation::getZeroOneRandomNumber()
{
    return uniformDist(mersenneTwisterEngine);
}


VehicleManager *coTrafficSimulation::getVehicleManager()
{
    return VehicleManager::Instance();
}

PedestrianManager *coTrafficSimulation::getPedestrianManager()
{
    return PedestrianManager::Instance();
}

xercesc::DOMElement *coTrafficSimulation::getOpenDriveRootElement(std::string filename)
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

void coTrafficSimulation::parseOpenDrive(xercesc::DOMElement *rootElement)
{
	XMLCh *t1 = NULL, *t2 = NULL, *t3 = NULL;
	char *ch;
    xercesc::DOMNodeList *documentChildrenList = rootElement->getChildNodes();

    for (int childIndex = 0; childIndex < documentChildrenList->getLength(); ++childIndex)
    {
        xercesc::DOMElement *sceneryElement = dynamic_cast<xercesc::DOMElement *>(documentChildrenList->item(childIndex));
        if (sceneryElement && xercesc::XMLString::compareIString(sceneryElement->getTagName(), t1 = xercesc::XMLString::transcode("scenery")) == 0)
        {
            std::string fileString = xercesc::XMLString::transcode(sceneryElement->getAttribute(t2 = xercesc::XMLString::transcode("file"))); xercesc::XMLString::release(&t2);
            std::string vpbString = xercesc::XMLString::transcode(sceneryElement->getAttribute(t2 = xercesc::XMLString::transcode("vpb"))); xercesc::XMLString::release(&t2);

            //std::vector<BoundingArea> voidBoundingAreaVector;
            std::vector<std::string> shapeFileNameVector;

            xercesc::DOMNodeList *sceneryChildrenList = sceneryElement->getChildNodes();
            xercesc::DOMElement *sceneryChildElement;
            for (unsigned int childIndex = 0; childIndex < sceneryChildrenList->getLength(); ++childIndex)
            {
                sceneryChildElement = dynamic_cast<xercesc::DOMElement *>(sceneryChildrenList->item(childIndex));
                if (!sceneryChildElement)
                    continue;

              /*  if (xercesc::XMLString::compareIString(sceneryChildElement->getTagName(), xercesc::XMLString::transcode("void")) == 0)
                {
                    double xMin = atof(xercesc::XMLString::transcode(sceneryChildElement->getAttribute(xercesc::XMLString::transcode("xMin"))));
                    double yMin = atof(xercesc::XMLString::transcode(sceneryChildElement->getAttribute(xercesc::XMLString::transcode("yMin"))));
                    double xMax = atof(xercesc::XMLString::transcode(sceneryChildElement->getAttribute(xercesc::XMLString::transcode("xMax"))));
                    double yMax = atof(xercesc::XMLString::transcode(sceneryChildElement->getAttribute(xercesc::XMLString::transcode("yMax"))));

                    voidBoundingAreaVector.push_back(BoundingArea(osg::Vec2(xMin, yMin), osg::Vec2(xMax, yMax)));
                   
                }
                else */if (xercesc::XMLString::compareIString(sceneryChildElement->getTagName(), t1 = xercesc::XMLString::transcode("shape")) == 0)
                {
                    std::string fileString = ch = xercesc::XMLString::transcode(sceneryChildElement->getAttribute(t2 = xercesc::XMLString::transcode("file"))); xercesc::XMLString::release(&t2); xercesc::XMLString::release(&ch);
                    shapeFileNameVector.push_back(fileString);
                }
				xercesc::XMLString::release(&t1);
            }

            if (!fileString.empty())
            {
                if (!coVRFileManager::instance()->fileExist((xodrDirectory + "/" + fileString).c_str()))
                {
                    std::cerr << "\n#\n# file not found: this may lead to a crash! \n#" << endl;
                }
                coVRFileManager::instance()->loadFile((xodrDirectory + "/" + fileString).c_str());
            }

            /* if (!vpbString.empty())
            {
                coVRPlugin *roadTerrainPlugin = cover->addPlugin("RoadTerrain");
                fprintf(stderr, "loading %s\n", vpbString.c_str());
                if (RoadTerrainPlugin::plugin)
                {
                    osg::Vec3d offset(0, 0, 0);
                    const RoadSystemHeader &header = RoadSystem::Instance()->getHeader();
                    offset.set(header.xoffset, header.yoffset, 0.0);
                    fprintf(stderr, "loading %s offset: %f %f\n", (xodrDirectory + "/" + vpbString).c_str(), offset[0], offset[1]);
                    RoadTerrainPlugin::plugin->loadTerrain(xodrDirectory + "/" + vpbString, offset, voidBoundingAreaVector, shapeFileNameVector);
                }
            }*/
        }
        else if (sceneryElement && xercesc::XMLString::compareIString(sceneryElement->getTagName(), t2 = xercesc::XMLString::transcode("environment")) == 0)
        {
            std::string tessellateRoadsString = ch = xercesc::XMLString::transcode(sceneryElement->getAttribute(t3 = xercesc::XMLString::transcode("tessellateRoads"))); xercesc::XMLString::release(&t3); xercesc::XMLString::release(&ch);
            if (tessellateRoadsString == "false" || tessellateRoadsString == "0")
            {
                tessellateRoads = false;
            }
            else
            {
                tessellateRoads = true;
            }

            std::string tessellatePathsString = ch = xercesc::XMLString::transcode(sceneryElement->getAttribute(t3 = xercesc::XMLString::transcode("tessellatePaths"))); xercesc::XMLString::release(&t3); xercesc::XMLString::release(&ch);
            if (tessellatePathsString == "false" || tessellatePathsString == "0")
            {
                tessellatePaths = false;
            }
            else
            {
                tessellatePaths = true;
            }

            std::string tessellateBattersString = ch = xercesc::XMLString::transcode(sceneryElement->getAttribute(t3 = xercesc::XMLString::transcode("tessellateBatters"))); xercesc::XMLString::release(&t3); xercesc::XMLString::release(&ch);
            if (tessellateBattersString == "true")
            {
                tessellateBatters = true;
            }
            else
            {
                tessellateBatters = false;
            }

            std::string tessellateObjectsString = ch = xercesc::XMLString::transcode(sceneryElement->getAttribute(t3 = xercesc::XMLString::transcode("tessellateObjects"))); xercesc::XMLString::release(&t3); xercesc::XMLString::release(&ch);
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

bool coTrafficSimulation::loadRoadSystem(const char *filename_chars)
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
         std::cout << " ----------------------- coTrafficSimulation -----------------------" << std::endl;
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

void coTrafficSimulation::deleteRoadSystem()
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

bool coTrafficSimulation::init()
{// UDP Broadcast //
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
	return true;
}

void
coTrafficSimulation::preFrame()
{
    
    double dt = cover->frameDuration();
    VehicleList vehOverallList = VehicleManager::Instance()->getVehicleOverallList();
    osg::Vec2d incoming_pos = HumanVehicle::human_pos; // Robert
    double v = (double)RoadSystem::dSpace_v;
    //Löschdistanz ermitteln
    if (v <= minVel)
        min_distance = min_distance_tui;
    else if (v >= maxVel)
        min_distance = max_distance_tui;
    else
        min_distance = (min_distance_tui + ((max_distance_tui - min_distance_tui) / (maxVel - minVel)) * (v - minVel));
    delete_at = min_distance + delete_delta;

    // Ermitteln in welcher Kachel sich das Eigenfahrzeug befinden würde
    if ((Carpool::Instance()->getPoolVector()).size() > 0 && useCarpool == 1)
    { //Ausführung nur falls mindestens ein pool definiert wurde
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

                    coTrafficSimulation::myInstance->getVehicleManager()->removeVehicle((*it), (*it)->getRoad());
                }
            }
        }
        if (counter >= createFreq)
        {
            if (maxVehicles == 0)
                factory->createTileVehicle(current_tile_y[0], current_tile_y[1], VehicleManager::Instance()->acitve_fiddleyards);
            else if (((VehicleManager::Instance()->getVehicleOverallList()).size()) < maxVehicles)
            {
                factory->createTileVehicle(current_tile_y[0], current_tile_y[1], VehicleManager::Instance()->acitve_fiddleyards);
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

		VehicleManager::Instance()->moveAllVehicles(dt);
		VehicleManager::Instance()->updateFiddleyards(dt, incoming_pos);
		if (coVRMSController::instance()->isMaster())
		{
			//if(operatorMap) {
			//   // send positions of the vehicles to operator map
			//   VehicleManager::Instance()->sendDataTo(operatorMap);
			//}
			if (ffzBroadcaster)
			{
				// send positions of nearby vehicles via UDP Broadcast
				VehicleManager::Instance()->sendDataTo(ffzBroadcaster);
			}
		}
		if (ffzBroadcaster)
		{
			// receive and parse Data from UDP Broadcast
			VehicleManager::Instance()->receiveDataFrom(ffzBroadcaster);
		}
		PedestrianManager::Instance()->moveAllPedestrians(dt);
		PedestrianManager::Instance()->updateFiddleyards(dt);
    }
}
