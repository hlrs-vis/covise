/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                            (C)2001 HLRS  **
 **                                                                          **
 ** Description: Template Plugin (does nothing)                              **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner		                                                **
 **                                                                          **
 ** History:  								                                **
 ** Nov-01  v1	    				       		                            **
 **                                                                          **
 **                                                                          **
\****************************************************************************/


#include <cover/coVRPluginSupport.h>
#include <cover/coVRPlugin.h>
#include <cover/RenderObject.h>
#include <cover/coVRFileManager.h>
#include "OpenScenarioPlugin.h"
#include <cover/coVRMSController.h>
#include "../RoadTerrain/RoadTerrainPlugin.h"

#include "FindTrafficLightSwitch.h"

#include <DrivingSim/OpenScenario/OpenScenarioBase.h>
#include <DrivingSim/OpenScenario/schema/oscFileHeader.h>
using namespace OpenScenario; 
using namespace opencover;

OpenScenarioPlugin *OpenScenarioPlugin::plugin = NULL;

static FileHandler handlers[] = {
    { NULL,
      OpenScenarioPlugin::loadOSC,
      OpenScenarioPlugin::loadOSC,
      NULL,
      "xosc" }
};

OpenScenarioPlugin::OpenScenarioPlugin()
{
	plugin = this;
	
	rootElement=NULL;
	roadGroup=NULL;
	system=NULL;
    manager=NULL;
	pedestrianManager=NULL;
    factory=NULL;
	pedestrianFactory=NULL;
    tessellateRoads=true;
    tessellatePaths=true;
    tessellateBatters=false;
    tessellateObjects=true;

	osdb = new OpenScenario::OpenScenarioBase();
    fprintf(stderr, "OpenScenario::OpenScenario\n");
}

// this is called if the plugin is removed at runtime
OpenScenarioPlugin::~OpenScenarioPlugin()
{
    fprintf(stderr, "OpenScenarioPlugin::~OpenScenarioPlugin\n");
}


void OpenScenarioPlugin::preFrame(){}

COVERPLUGIN(OpenScenarioPlugin)

bool OpenScenarioPlugin::init()
{
    coVRFileManager::instance()->registerFileHandler(&handlers[0]);
    //coVRFileManager::instance()->registerFileHandler(&handlers[1]);
	return true;
}

int OpenScenarioPlugin::loadOSC(const char *filename, osg::Group *g, const char *key)
{
	return plugin->loadOSCFile(filename,g,key);
}

int OpenScenarioPlugin::loadOSCFile(const char *filename, osg::Group *, const char *key)
{
	if(osdb->loadFile(filename, "OpenSCENARIO", "OpenSCENARIO") == false)
    {
        std::cerr << std::endl;
        std::cerr << "failed to load OpenSCENARIO from file " << filename << std::endl;
        std::cerr << std::endl;
        delete osdb;
        return -1;
    }

	//load xodr 
	std::string xodrName_st = osdb->RoadNetwork->Logics->openDRIVE.getValue();
	const char * xodrName = xodrName_st.c_str();
	loadRoadSystem(xodrName);

	// look for sources and sinks and load them
	oscGlobalAction	ga;
	oscGlobalActionArrayMember *Global = &osdb->Storyboard->Init->Actions->Global;
	int fiddleyards=0;
	for (oscGlobalActionArrayMember::iterator it = Global->begin(); it != Global->end(); it++)
	{
		oscGlobalAction* action = ((oscGlobalAction*)(*it));
		if (action->Traffic.getObject()) // this is a Traffic action
		{
			if (oscSource *source = action->Traffic->Source.getObject())
			{
				if (oscRelativeRoad* position = source->Position->RelativeRoad.getObject())
				{
					// find road
					system = RoadSystem::Instance();
					Road* myRoad = system->getRoad(position->object.getValue());
					if (myRoad)
					{
						std::string name = "fiddleyard" + position->object.getValue();
						std::string id = name + std::to_string(fiddleyards);
						Fiddleyard *fiddleyard = new Fiddleyard(name, id);

						system->addFiddleyard(fiddleyard); //, "road", position->object.getValue(), "start");

						std::string sid = name + "_source"+ std::to_string(fiddleyards++);
						double s, t;
						s = position->ds.getValue();
						t = position->dt.getValue();
						LaneSection * ls = myRoad->getLaneSection(s);
						ls->getLane()
						VehicleSource *source = new VehicleSource(sid, lane, 2, 8, 13.667, 6.0);
						fiddleyard->addVehicleSource(source);
/*
						fiddleyard->setTarmacConnection(new TarmacConnection(tarmac, direction));

						std::string id(xercesc::XMLString::transcode(fiddleyardChildElement->getAttribute(xercesc::XMLString::transcode("id"))));
						int lane = atoi(xercesc::XMLString::transcode(fiddleyardChildElement->getAttribute(xercesc::XMLString::transcode("lane"))));
						double starttime = atof(xercesc::XMLString::transcode(fiddleyardChildElement->getAttribute(xercesc::XMLString::transcode("startTime"))));
						double repeattime = atof(xercesc::XMLString::transcode(fiddleyardChildElement->getAttribute(xercesc::XMLString::transcode("repeatTime"))));
						double vel = atof(xercesc::XMLString::transcode(fiddleyardChildElement->getAttribute(xercesc::XMLString::transcode("velocity"))));
						double velDev = atof(xercesc::XMLString::transcode(fiddleyardChildElement->getAttribute(xercesc::XMLString::transcode("velocityDeviance"))));
						
						
						xercesc::DOMNodeList *sourceChildrenList = fiddleyardChildElement->getChildNodes();
						xercesc::DOMElement *sourceChildElement;
						for (unsigned int childIndex = 0; childIndex < sourceChildrenList->getLength(); ++childIndex)
						{
							sourceChildElement = dynamic_cast<xercesc::DOMElement *>(sourceChildrenList->item(childIndex));
							if (sourceChildElement && xercesc::XMLString::compareIString(sourceChildElement->getTagName(), xercesc::XMLString::transcode("vehicle")) == 0)
							{
								std::string id(xercesc::XMLString::transcode(sourceChildElement->getAttribute(xercesc::XMLString::transcode("id"))));
								double numerator = atof(xercesc::XMLString::transcode(sourceChildElement->getAttribute(xercesc::XMLString::transcode("numerator"))));
								source->addVehicleRatio(id, numerator);
							}
						}*/
					}
					else
					{
						fprintf(stderr,"Road not found in RelativeRoad Position %s\n", position->object.getValue());
					}
				}
				else
				{
					fprintf(stderr, "only Sources Relative to a Road are supported by now\n");
				}
			}
		}

	}
	

return 0;
}


bool OpenScenarioPlugin::loadRoadSystem(const char *filename_chars)
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
        //roadGroup->setPosition(osg::Vec3d(5.0500000000000000e+05, 5.3950000000000000e+06, 0.0));
        //roadGroup->setPosition(osg::Vec3d(960128.3125, 6158421.5, 0.0));

        //osg::Material* roadGroupMaterial = new osg::Material;
        //roadGroupMaterial->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
        //roadGroupState->setAttribute(roadGroupMaterial);

        int numRoads = system->getNumRoads();
        for (int i = 0; i < numRoads; ++i)
        {
            Road *road = system->getRoad(i);
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

        if (roadGroup->getNumChildren() > 0)
        {
            cover->getObjectsRoot()->addChild(roadGroup);
        }

        osg::Group *trafficSignalGroup = new osg::Group;
        trafficSignalGroup->setName("TrafficSignals");
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

        manager = VehicleManager::Instance();

        pedestrianManager = PedestrianManager::Instance();
    }

    if ((Carpool::Instance()->getPoolVector()).size() > 0)
        system->scanStreets(); // 29.08.2011

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

    }

    return true;
}

xercesc::DOMElement *OpenScenarioPlugin::getOpenDriveRootElement(std::string filename)
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

void OpenScenarioPlugin::parseOpenDrive(xercesc::DOMElement *rootElement)
{
    xercesc::DOMNodeList *documentChildrenList = rootElement->getChildNodes();

    for (int childIndex = 0; childIndex < documentChildrenList->getLength(); ++childIndex)
    {
        xercesc::DOMElement *sceneryElement = dynamic_cast<xercesc::DOMElement *>(documentChildrenList->item(childIndex));
        if (sceneryElement && xercesc::XMLString::compareIString(sceneryElement->getTagName(), xercesc::XMLString::transcode("scenery")) == 0)
        {
            std::string fileString = xercesc::XMLString::transcode(sceneryElement->getAttribute(xercesc::XMLString::transcode("file")));
            std::string vpbString = xercesc::XMLString::transcode(sceneryElement->getAttribute(xercesc::XMLString::transcode("vpb")));

            std::vector<BoundingArea> voidBoundingAreaVector;
            std::vector<std::string> shapeFileNameVector;

            xercesc::DOMNodeList *sceneryChildrenList = sceneryElement->getChildNodes();
            xercesc::DOMElement *sceneryChildElement;
            for (unsigned int childIndex = 0; childIndex < sceneryChildrenList->getLength(); ++childIndex)
            {
                sceneryChildElement = dynamic_cast<xercesc::DOMElement *>(sceneryChildrenList->item(childIndex));
                if (!sceneryChildElement)
                    continue;

                if (xercesc::XMLString::compareIString(sceneryChildElement->getTagName(), xercesc::XMLString::transcode("void")) == 0)
                {
                    double xMin = atof(xercesc::XMLString::transcode(sceneryChildElement->getAttribute(xercesc::XMLString::transcode("xMin"))));
                    double yMin = atof(xercesc::XMLString::transcode(sceneryChildElement->getAttribute(xercesc::XMLString::transcode("yMin"))));
                    double xMax = atof(xercesc::XMLString::transcode(sceneryChildElement->getAttribute(xercesc::XMLString::transcode("xMax"))));
                    double yMax = atof(xercesc::XMLString::transcode(sceneryChildElement->getAttribute(xercesc::XMLString::transcode("yMax"))));

                    voidBoundingAreaVector.push_back(BoundingArea(osg::Vec2(xMin, yMin), osg::Vec2(xMax, yMax)));
                    //voidBoundingAreaVector.push_back(BoundingArea(osg::Vec2(506426.839,5398055.357),osg::Vec2(508461.865,5399852.0)));
                }
                else if (xercesc::XMLString::compareIString(sceneryChildElement->getTagName(), xercesc::XMLString::transcode("shape")) == 0)
                {
                    std::string fileString = xercesc::XMLString::transcode(sceneryChildElement->getAttribute(xercesc::XMLString::transcode("file")));
                    shapeFileNameVector.push_back(fileString);
                }
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
            }
        }
        else if (sceneryElement && xercesc::XMLString::compareIString(sceneryElement->getTagName(), xercesc::XMLString::transcode("environment")) == 0)
        {
            std::string tessellateRoadsString = xercesc::XMLString::transcode(sceneryElement->getAttribute(xercesc::XMLString::transcode("tessellateRoads")));
            if (tessellateRoadsString == "false" || tessellateRoadsString == "0")
            {
                tessellateRoads = false;
            }
            else
            {
                tessellateRoads = true;
            }

            std::string tessellatePathsString = xercesc::XMLString::transcode(sceneryElement->getAttribute(xercesc::XMLString::transcode("tessellatePaths")));
            if (tessellatePathsString == "false" || tessellatePathsString == "0")
            {
                tessellatePaths = false;
            }
            else
            {
                tessellatePaths = true;
            }

            std::string tessellateBattersString = xercesc::XMLString::transcode(sceneryElement->getAttribute(xercesc::XMLString::transcode("tessellateBatters")));
            if (tessellateBattersString == "true")
            {
                tessellateBatters = true;
            }
            else
            {
                tessellateBatters = false;
            }

            std::string tessellateObjectsString = xercesc::XMLString::transcode(sceneryElement->getAttribute(xercesc::XMLString::transcode("tessellateObjects")));
            if (tessellateObjectsString == "true")
            {
                tessellateObjects = true;
            }
            else
            {
                tessellateObjects = false;
            }
        }
    }
}
