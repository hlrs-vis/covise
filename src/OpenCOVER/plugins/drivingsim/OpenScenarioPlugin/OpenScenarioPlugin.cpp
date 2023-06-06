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

#include <OpenVRUI/osg/mathUtils.h>
#include <config/CoviseConfig.h>
#include <net/covise_connect.h>
#include <net/covise_socket.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRPlugin.h>
#include <cover/RenderObject.h>
#include <cover/coVRFileManager.h>
#include "OpenScenarioPlugin.h"
#include <cover/coVRMSController.h>
#include <cover/OpenCOVER.h>
#include <cover/coVRPluginList.h>
#include <cover/coVRConfig.h>

#include "../RoadTerrain/RoadTerrainPlugin.h"
#include "ScenarioManager.h"
#include "Trajectory.h"
#include <TrafficSimulation/AgentVehicle.h>
#include <TrafficSimulation/CarGeometry.h>
#include <algorithm>

#include <TrafficSimulation/FindTrafficLightSwitch.h>

#include <OpenScenario/OpenScenarioBase.h>
#include <OpenScenario/schema/oscFileHeader.h>
#include "myFactory.h"
#include "CameraSensor.h"
#include <config/CoviseConfig.h>
#include "Position.h"
#include "Spline.h"
#include "Entity.h"
#include "ReferencePosition.h"
#include "Action.h"
#include "Sequence.h"
#include "Event.h"
#include "Condition.h"
#include "LaneChange.h"
#include <config/CoviseConfig.h>
#include <cover/coVRConfig.h>
#include <osgDB/WriteFile>
#include <chrono>
#include <thread>

using namespace OpenScenario;
using namespace opencover;
using namespace vehicleUtil;
using namespace TrafficSimulation;

OpenScenarioPlugin *OpenScenarioPlugin::plugin = NULL;

static FileHandler handlers[] = {
	{ NULL,
	  OpenScenarioPlugin::loadOSC,
	  NULL,
	  "xosc" }
};

OpenScenarioPlugin::OpenScenarioPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
	plugin = this;

	rootElement = NULL;
	roadGroup = NULL;
	system = NULL;
	manager = NULL;
	pedestrianManager = NULL;
	factory = NULL;
	pedestrianFactory = NULL;
	tessellateRoads = true;
	tessellatePaths = true;
	tessellateBatters = false;
	tessellateObjects = true;

	frameCounter = 0;

    waitOnStart = false;

#ifdef _MSC_VER
	GL_fmt = GL_BGR_EXT;
#else
	GL_fmt = GL_BGRA;
#endif
	doWait = false;
	frameRate = covise::coCoviseConfig::getInt("COVER.Plugin.OpenScenario.FrameRate", 0);
	writeRate = covise::coCoviseConfig::getInt("COVER.Plugin.OpenScenario.WriteRate", 0);
    minSimulationStep = covise::coCoviseConfig::getFloat("COVER.Plugin.OpenScenario.MinSimulationStep", minSimulationStep);
	doExit = covise::coCoviseConfig::isOn("COVER.Plugin.OpenScenario.ExitOnScenarioEnd", false);
    if(frameRate > 0)
	coVRConfig::instance()->setFrameRate(frameRate);

	scenarioManager = new ScenarioManager();

	osdb = new OpenScenario::OpenScenarioBase();
	// set our own object factory so that our own classes are created and not the bas osc* classes
	OpenScenario::oscFactories::instance()->setObjectFactory(new myFactory());
	osdb->setFullReadCatalogs(true);

	port = covise::coCoviseConfig::getInt("port", "COVER.Plugin.OpenScenario.Server", 11021);
	toClientConn = NULL;
	serverConn = NULL;
	
	//todo coTrafficSimulation::useInstance();
	fprintf(stderr, "OpenScenario::OpenScenario\n");
}

// this is called if the plugin is removed at runtime
OpenScenarioPlugin::~OpenScenarioPlugin()
{

	// coTrafficSimulation::freeInstance();
	fprintf(stderr, "OpenScenarioPlugin::~OpenScenarioPlugin\n");

	cover->getObjectsRoot()->removeChild(trafficSignalGroup);
	cover->getObjectsRoot()->removeChild(roadGroup);
}
void OpenScenarioPlugin::checkAndHandleMessages(bool blocking)
{
	if (serverConn && serverConn->is_connected() && serverConn->check_for_input()) // we have a server and received a connect
	{
		toClientConn = serverConn->spawnSimpleConnection();
		if (toClientConn)
		{
			if (toClientConn->is_connected())
			{

			}
			else
			{
				toClientConn.reset(nullptr);
			}
		}
	}

	int size = 0;
	if (coVRMSController::instance()->isMaster())
	{
		while (toClientConn && (blocking || toClientConn->check_for_input()))
		{
			blocking = false;
			if (readTCPData(&size, sizeof(int)) == false)
			{
				toClientConn.reset(nullptr);
			}
			byteSwap(size);
			if (size > 0)
			{
				char *buf = new char[size + 1];
				readTCPData(buf, size);
				buf[size] = '\0';
				coVRMSController::instance()->sendSlaves(&size, sizeof(size));
				coVRMSController::instance()->sendSlaves(buf, size);
				handleMessage(buf);
				delete[] buf;
			}
			size = 0;
		}
		coVRMSController::instance()->sendSlaves(&size, sizeof(size));

	}
	else
	{
		do {
			coVRMSController::instance()->readMaster(&size, sizeof(size));
			if (size > 0)
			{
				char *buf = new char[size + 1];
				coVRMSController::instance()->readMaster(&buf, size);
				buf[size] = '\0';
				handleMessage(buf);
				delete[] buf;
			}
		} while (size > 0);
	}
}

bool OpenScenarioPlugin::update()
{
	checkAndHandleMessages();
	return true;
}


void OpenScenarioPlugin::preSwapBuffers(int windowNumber)
{

	auto &coco = *coVRConfig::instance();


	if (writeRate > 0 && coVRMSController::instance()->isMaster())
	{
		if ((frameCounter % writeRate) == 0)
		{
			// fprintf(stderr,"glRead...\n");
			glPixelStorei(GL_PACK_ALIGNMENT, 1);
			if (coco.windows[windowNumber].doublebuffer)
				glReadBuffer(GL_BACK);
			// for depth to work, it might be necessary to read from GL_FRONT (change this, if it does not work like
			// this)
			if (image.get() == NULL)
			{

				image = new osg::Image();

				image.get()->allocateImage(coco.windows[windowNumber].sx, coco.windows[windowNumber].sy, 1, GL_fmt, GL_UNSIGNED_BYTE);
			}
			glReadPixels(0, 0, coco.windows[windowNumber].sx, coco.windows[windowNumber].sy, GL_fmt, GL_UNSIGNED_BYTE, image->data());

			char filename[100];
			sprintf(filename, "test%d.png", frameCounter);


			if (osgDB::writeImageFile(*(image.get()), filename))
			{
			}
			else
			{
			}
		}
	}
	frameCounter++;
}
void
OpenScenarioPlugin::handleMessage(const char *buf)
{
	fprintf(stderr, "%s\n", buf);
	if (strcmp(buf, "restart") == 0)
	{
		blockingWait = false;
        frameCounter = 0;
		scenarioManager->restart();
	}
	if (strncmp(buf, "set ", 4) == 0)
	{
		std::string assignment = buf + 4;
		std::string variable, value;
		std::string::size_type  pos = 0;

		if ((pos = assignment.find('=', 0)) != std::string::npos)
		{
			variable = assignment.substr(0, pos);
			value = assignment.substr(pos + 1, std::string::npos);
			osdb->setParameterValue(variable, value);
		}

	}
	if (strcmp(buf, "exit") == 0)
	{
		blockingWait = false;
		OpenCOVER::instance()->setExitFlag(true);
		coVRPluginList::instance()->requestQuit(true);
	}
}

bool
OpenScenarioPlugin::readTCPData(void *buf, unsigned int numBytes)
{
	unsigned int stillToRead = numBytes;
	unsigned int alreadyRead = 0;
	int readBytes = 0;
	while (alreadyRead < numBytes)
	{
		readBytes = toClientConn->getSocket()->Read(((unsigned char *)buf) + alreadyRead, stillToRead);
		if (readBytes < 0)
		{
			std::cerr << "Error error while reading data from socket in OpenScenarioPlugin::readTCPData" << std::endl;
			return false;
		}
		alreadyRead += readBytes;
		stillToRead = numBytes - alreadyRead;
	}
	return true;
}

bool OpenScenarioPlugin::advanceTime(double step)
{
    scenarioManager->simulationStep = step;
    scenarioManager->simulationTime += step;
    //cout << "TIME: " << scenarioManager->simulationTime << endl;
    list<Entity*> usedEntity;
    list<Entity*> unusedEntity = scenarioManager->entityList;
    list<Entity*> entityList_temp = scenarioManager->entityList;
    entityList_temp.sort();unusedEntity.sort();
    scenarioManager->conditionManager();

    if (!scenarioManager->scenarioCondition)
        return false;

    {
        for(list<Act*>::iterator act_iter = scenarioManager->actList.begin(); act_iter != scenarioManager->actList.end(); act_iter++)
        {
            Act* currentAct = (*act_iter);
            //check act start conditions
            if (currentAct->isRunning())
            {
                for(list<Sequence*>::iterator sequence_iter = currentAct->sequenceList.begin(); sequence_iter != currentAct->sequenceList.end(); sequence_iter++)
                {
                    Sequence* currentSequence = (*sequence_iter);
                    Maneuver* currentManeuver = currentSequence->activeManeuver;
                    if(currentManeuver != NULL)
                    {
                        Event* currentEvent = currentManeuver->activeEvent;
                        if(currentEvent != NULL)
                        {
                            for(list<Entity*>::iterator entity_iter = currentSequence->actorList.begin(); entity_iter != currentSequence->actorList.end(); entity_iter++)
                            {
                                Entity* currentEntity = (*entity_iter);
                                for(auto action_iter = currentEvent->Action.begin(); action_iter != currentEvent->Action.end(); action_iter++)
                                {
                                    Action* currentAction = dynamic_cast<Action *>(*action_iter);
                                    cout << "Entity Action: " << currentAction->name.getValue() << endl;
                                    if(currentAction->Private.exists())
                                    {
                                        if (currentAction->Private->Routing.exists())
                                        {
                                            if (currentAction->Private->Routing.exists())
                                            {
                                                if(currentAction->Private->Routing->FollowTrajectory.exists())
                                                {
                                                    Trajectory* currentTrajectory = currentAction->actionTrajectory;

													currentEntity->followTrajectory(currentEvent);
													//cout << "Entity new Position: " << currentEntity->refPos->xyz[0] << ", " << currentEntity->refPos->xyz[1] << ", "<< currentEntity->refPos->xyz[2] << endl;

													unusedEntity.remove(currentEntity);
													usedEntity.push_back(currentEntity);
													usedEntity.sort(); usedEntity.unique();
												}
											}
										}
										else if (currentAction->Private->Longitudinal.exists())
										{
											if (currentAction->Private->Longitudinal->Speed.exists())
											{
												double targetspeed = currentAction->Private->Longitudinal->Speed->Target->Absolute->value.getValue();
												int shape = currentAction->Private->Longitudinal->Speed->Dynamics->shape.getValue();

												currentEntity->longitudinalSpeedAction(currentEvent, targetspeed, shape);


											}
										}
										else if (currentAction->Private->Lateral.exists())
										{
											if (currentAction->Private->Lateral->LaneChange.exists())
											{
												LaneChange* lc = dynamic_cast<LaneChange*>(currentAction->Private->Lateral->LaneChange.getObject());
												if (lc != NULL)
												{
													//lc->doLaneChange(currentEntity);
													unusedEntity.remove(currentEntity);
													usedEntity.push_back(currentEntity);
													usedEntity.sort(); usedEntity.unique();
													
													currentEntity->doLaneChange(lc,currentEvent); 
													
													
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}



		for (list<Entity*>::iterator activeEntity = unusedEntity.begin(); activeEntity != unusedEntity.end(); activeEntity++)
		{
			Entity* currentEntity = (*activeEntity);
			currentEntity->moveLongitudinal();
		}
		usedEntity.clear();
		scenarioManager->resetReferencePositionStatus();

    }

    return true;
}
void OpenScenarioPlugin::writeString(const std::string &msg)
{
	int len = msg.length();

	byteSwap(len);
	toClientConn->send(&len, sizeof(int));
	toClientConn->send(msg.c_str(), msg.length());
}

void OpenScenarioPlugin::preFrame()
{
    double step = cover->frameDuration();
    int nsteps = 1;
    if (minSimulationStep > 0)
        nsteps = step/minSimulationStep;
    if (nsteps < 1)
        nsteps = 1;

    double s = step/nsteps;
    for (int i=0; i<nsteps; ++i)
    {
        if (!advanceTime(s))
        {
            // Scenario end

            if (doExit)
            {
                OpenCOVER::instance()->setExitFlag(true);
                coVRPluginList::instance()->requestQuit(true);
            }
			else
			{
                blockingWait = true;
                if (toClientConn != NULL)
                {
                    writeString("szenarioEnd\n");
                }
                while (blockingWait)
                {
                    if (toClientConn != NULL)
                    {
                        checkAndHandleMessages(true);
                    }
                    else
                    {
                        checkAndHandleMessages(false);
                        std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    }
                }
			}
            //fprintf(stderr, "END\n");

            break;
        }
    }

	if (currentCamera != NULL)
		currentCamera->updateView();
}

COVERPLUGIN(OpenScenarioPlugin)

bool OpenScenarioPlugin::init()
{
	coVRFileManager::instance()->registerFileHandler(&handlers[0]);
	//coVRFileManager::instance()->registerFileHandler(&handlers[1]);
	return true;
}

int OpenScenarioPlugin::loadOSC(const char *filename, osg::Group *g, const char *key)
{
	return plugin->loadOSCFile(filename, g, key);
}

int OpenScenarioPlugin::loadOSCFile(const char *file, osg::Group *, const char *key)
{
	osdb->setValidation(false); // don't validate, we might be on the road where we don't have axxess to the schema files
	if (osdb->loadFile(file, "OpenSCENARIO", "OpenSCENARIO") == false)
	{
		std::cerr << std::endl;
		std::cerr << "failed to load OpenSCENARIO from file " << file << std::endl;
		std::cerr << std::endl;
		delete osdb;
		osdb = nullptr;

		return -1;
	}
	std::string filename(file);
	xoscDirectory.clear();
	if (filename[0] != '/' && filename[0] != '\\' && (!(filename[1] == ':' && (filename[2] == '/' || filename[2] == '\\'))))
	{ // / or backslash or c:/
		char* workingDir = getcwd(NULL, 0);
		xoscDirectory.assign(workingDir);
		free(workingDir);
	}
	size_t lastSlashPos = filename.find_last_of('/');
	size_t lastSlashPos2 = filename.find_last_of('\\');
	if (lastSlashPos != filename.npos && (lastSlashPos2 == filename.npos || lastSlashPos2 < lastSlashPos))
	{
		if (!xoscDirectory.empty())
			xoscDirectory += "/";
		xoscDirectory.append(filename, 0, lastSlashPos);
	}
	if (lastSlashPos2 != filename.npos && (lastSlashPos == filename.npos || lastSlashPos < lastSlashPos2))
	{
		if (!xoscDirectory.empty())
			xoscDirectory += "\\";
		xoscDirectory.append(filename, 0, lastSlashPos2);
	}
	if (osdb->FileHeader.exists())
	{
		for (auto it = osdb->FileHeader->UserData.begin(); it != osdb->FileHeader->UserData.end(); it++)
		{

			oscUserData* userdata = ((oscUserData*)(*it));
			if (userdata->code.getValue() == "FrameRate")
			{
				frameRate = std::stoi(userdata->value.getValue());
                if (frameRate > 0)
                    coVRConfig::instance()->setFrameRate(frameRate);
			}
			else if (userdata->code.getValue() == "WriteRate")
			{
				writeRate = std::stoi(userdata->value.getValue());
			}
            else if (userdata->code.getValue() == "WaitOnStart")
            {
                if (userdata->value.getValue() == "True")
                {
                    waitOnStart = true;
                }
            }
			else if (userdata->code.getValue() == "MinSimulationStep")
			{
				minSimulationStep = std::stoi(userdata->value.getValue());
			}
            else if (userdata->code.getValue() == "ScenarioEnd")
            {
                if (userdata->value.getValue() == "Exit")
                    doExit = true;
                if (userdata->value.getValue() == "Wait")
                {
                    doWait = true;
                    doExit = false;
                }
            }
            else if (userdata->code.getValue() == "Port")
			{
				port = std::stoi(userdata->value.getValue());
			}
		}
	}
	// open controll socket
	if (coVRMSController::instance()->isMaster())
	{
		serverConn = new covise::ServerConnection(port, 1234, 0);
		if (!serverConn->getSocket())
		{
			cout << "tried to open server Port " << port << endl;
			cout << "Creation of server failed!" << endl;
			cout << "Port-Binding failed! Port already bound?" << endl;
			delete serverConn;
			serverConn = NULL;
		}

		struct linger linger;
		linger.l_onoff = 0;
		linger.l_linger = 0;
		if (serverConn)
		{
			setsockopt(serverConn->get_id(NULL), SOL_SOCKET, SO_LINGER, (char *)&linger, sizeof(linger));

			cout << "Set server to listen mode..." << endl;
			serverConn->listen();
			if (!serverConn->is_connected()) // could not open server port
			{
				fprintf(stderr, "Could not open server port %d\n", port);
				delete serverConn;
				serverConn = NULL;
			}
		}
	}
    if(waitOnStart)
    {
        blockingWait = true;
        while (blockingWait)
        {
            if (toClientConn != NULL)
            {
                checkAndHandleMessages(true);
            }
            else
            {
                checkAndHandleMessages(false);
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
    }

	//load xodr
	if (osdb->RoadNetwork.getObject() != NULL)
	{
        if (osdb->RoadNetwork->Logics.exists())
        {
            std::string xodrName_st = osdb->RoadNetwork->Logics->filepath.getValue();
            const char * xodrName = xodrName_st.c_str();
            loadRoadSystem(xodrName);
        }
        if(osdb->RoadNetwork->SceneGraph.exists())
        {
            std::string geometryFile = osdb->RoadNetwork->SceneGraph->filepath.getValue();
            //load ScenGraph
            coVRFileManager::instance()->loadFile(geometryFile.c_str());
        }
	}
	//if(osdb->getBase()->)

	if (osdb->Storyboard.getObject() == NULL)
	{
		return -1;
	}
	else
	{
		// look for sources and sinks and load them
		oscGlobalAction	ga;
		oscGlobalActionArrayMember *Global = &osdb->Storyboard->Init->Actions->Global;
		int fiddleyards = 0;
		for (oscGlobalActionArrayMember::iterator it = Global->begin(); it != Global->end(); it++)
		{
			oscGlobalAction* action = ((oscGlobalAction*)(*it));
			if (action->Traffic.getObject()) // this is a Traffic action
			{
				if (oscSource *source = action->Traffic->Source.getObject())
				{
					if (oscRelativeRoad* position = source->Position->RelativeRoad.getObject())
					{
						if (oscTrafficDefinition* traffic = source->TrafficDefinition.getObject())
						{
							oscVehicleDistribution* vd = traffic->VehicleDistribution.getObject();
							oscDriverDistribution* dd = traffic->DriverDistribution.getObject();
							if (vd != NULL && dd != NULL)
							{
								// find road
								system = vehicleUtil::RoadSystem::Instance();
								vehicleUtil::Road* myRoad = system->getRoad(position->object.getValue());
								if (myRoad)
								{
									std::string name = "fiddleyard" + position->object.getValue();
									std::string id = name + std::to_string(fiddleyards);
									Fiddleyard *fiddleyard = new Fiddleyard(name, id);

									system->addFiddleyard(fiddleyard); //, "road", position->object.getValue(), "start");

									std::string sid = name + "_source" + std::to_string(fiddleyards++);
									double s, t;
									s = position->ds.getValue();
									t = position->dt.getValue();

									int direction = 1;
									if (t < 0)
										direction = -1;
									fiddleyard->setTarmacConnection(new TarmacConnection(myRoad, direction));
									VehicleSource *vs = new VehicleSource(sid, myRoad->getLaneNumber(s, t), 0, 5, 5.0, 5.0);
									fiddleyard->addVehicleSource(vs);

									//for (oscVehicleArrayMember::iterator it = vd->Vehicle.begin(); it != vd->Vehicle.end(); it++);
									for (int i = 0; i < vd->Vehicle.size(); i++)
									{
										oscVehicle *vehicle = vd->Vehicle[i];
										//vs->addVehicleRatio(vehicle->name.getValue(), vd->percentage.getValue());
										vs->addVehicleRatio(vehicle->name.getValue(), 50.0);
									}
									//ls->getLane();
									//VehicleSource *source = new VehicleSource(sid, lane, 2, 8, 13.667, 6.0);
									//fiddleyard->addVehicleSource(source);
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
									fprintf(stderr, "Road not found in RelativeRoad Position %s\n", position->object.getValue().c_str());
								}
							}
						}
						else
						{
							fprintf(stderr, "no Traffic definition \n");
						}
					}
					else
					{
						fprintf(stderr, "only Sources Relative to a Road are supported by now\n");
					}
				}
			}
		}

		//initialize entities
		for (oscObjectArrayMember::iterator it = osdb->Entities->Object.begin(); it != osdb->Entities->Object.end(); it++)
		{
			oscObject* entity = ((oscObject*)(*it));
			scenarioManager->entityList.push_back(new Entity(entity));
			cout << "Entity: " << entity->name.getValue() << " initialized" << endl;
		}

		//create Cameras
		for (list<Entity*>::iterator entity_iter = scenarioManager->entityList.begin(); entity_iter != scenarioManager->entityList.end(); entity_iter++)
		{
			Entity *currentEntity = (*entity_iter);
			oscVehicle* vehicle = currentEntity->getVehicle();
			if (vehicle && vehicle->ParameterDeclaration.exists())
			{
				for (int i = 0; i < vehicle->ParameterDeclaration->Parameter.size(); i++)
				{
					if (vehicle->ParameterDeclaration->Parameter[i]->name.getValue() == "CameraX")
					{
						double X = 0.0, Y = 0.0, Z = 0.0, H = 0.0, P = 0.0, R = 0.0, FOV = 0.0;
						for (int n = i; n < vehicle->ParameterDeclaration->Parameter.size(); n++)
						{
							if (vehicle->ParameterDeclaration->Parameter[n]->name.getValue() == "CameraX")
							{
								X = atof(vehicle->ParameterDeclaration->Parameter[n]->value.getValue().c_str());
							}
							else if (vehicle->ParameterDeclaration->Parameter[n]->name.getValue() == "CameraY")
							{
								Y = atof(vehicle->ParameterDeclaration->Parameter[n]->value.getValue().c_str());
							}
							else if (vehicle->ParameterDeclaration->Parameter[n]->name.getValue() == "CameraZ")
							{
								Z = atof(vehicle->ParameterDeclaration->Parameter[n]->value.getValue().c_str());
							}
							else if (vehicle->ParameterDeclaration->Parameter[n]->name.getValue() == "CameraH")
							{
								H = atof(vehicle->ParameterDeclaration->Parameter[n]->value.getValue().c_str());
							}
							else if (vehicle->ParameterDeclaration->Parameter[n]->name.getValue() == "CameraP")
							{
								P = atof(vehicle->ParameterDeclaration->Parameter[n]->value.getValue().c_str());
							}
							else if (vehicle->ParameterDeclaration->Parameter[n]->name.getValue() == "CameraR")
							{
								R = atof(vehicle->ParameterDeclaration->Parameter[n]->value.getValue().c_str());
							}
							else if (vehicle->ParameterDeclaration->Parameter[n]->name.getValue() == "CameraFOV")
							{
								FOV = atof(vehicle->ParameterDeclaration->Parameter[n]->value.getValue().c_str());
							}
						}
						coCoord coord;
						coord.hpr[0] = H;
						coord.hpr[1] = P;
						coord.hpr[2] = R;
						coord.xyz[0] = X;
						coord.xyz[1] = Y;
						coord.xyz[2] = Z;
						osg::Matrix cameraMat;
						coord.makeMat(cameraMat);

						osg::Matrix rotMat;
						rotMat.makeRotate(M_PI / 2.0, 0.0, 0.0, 1.0);

						CameraSensor* camera = new CameraSensor(currentEntity, vehicle, rotMat * cameraMat, FOV);
						cameras.push_back(camera);
						if (currentCamera == NULL)
							currentCamera = camera;
						break;
					}
				}
			}
		}

		scenarioManager->initializeEntities();


		//initialize acts
		for (oscStoryArrayMember::iterator it = osdb->Storyboard->Story.begin(); it != osdb->Storyboard->Story.end(); it++)
		{
			oscStory* story = ((oscStory*)(*it));
			for (oscActArrayMember::iterator it = story->Act.begin(); it != story->Act.end(); it++)
			{
				Act* act = ((Act*)(*it));// these are not oscAct instances any more but our own Act
				scenarioManager->actList.push_back(act);

				for (oscActorsArrayMember::iterator it = act->Sequence.begin(); it != act->Sequence.end(); it++)
				{
					Sequence* currentSeq = ((Sequence*)(*it));
					list<Entity*> activeEntityList_temp;
					for (oscActorsArrayMember::iterator it = currentSeq->Actors->Entity.begin(); it != currentSeq->Actors->Entity.end(); it++)
					{
						oscEntity* namedEntity = ((oscEntity*)(*it));
						if (namedEntity->name.getValue() != "$owner")
						{
							activeEntityList_temp.push_back(scenarioManager->getEntityByName(namedEntity->name.getValue()));
						}
						else
						{
							activeEntityList_temp.push_back(scenarioManager->getEntityByName(story->owner.getValue()));
						}
						cout << "Entity: " << story->owner.getValue() << " allocated to " << act->getName() << endl;
					}

					list<Maneuver*> maneuverList_temp;
					for (oscManeuverArrayMember::iterator it = currentSeq->Maneuver.begin(); it != currentSeq->Maneuver.end(); it++)
					{
						Maneuver* maneuver = ((Maneuver*)(*it)); // these are not oscManeuver instances any more but our own Maneuver
						for (oscManeuverArrayMember::iterator it = maneuver->Event.begin(); it != maneuver->Event.end(); it++)
						{
							Event* event = ((Event*)(*it));
							maneuver->initialize(event);
						}
						maneuverList_temp.push_back(maneuver);
						cout << "Manuever: " << maneuver->getName() << " created" << endl;
					}
					cout << "Act: " << act->getName() << " initialized" << endl;
					currentSeq->initialize(activeEntityList_temp, maneuverList_temp);
					act->initialize(currentSeq);

					maneuverList_temp.clear();
					activeEntityList_temp.clear();
				}
			}
		}
		//get Conditions
		if (osdb->Storyboard->EndConditions.exists())
		{
			for (oscConditionArrayMember::iterator it = osdb->Storyboard->EndConditions->ConditionGroup.begin(); it != osdb->Storyboard->EndConditions->ConditionGroup.end(); it++)
			{
				oscConditionGroup* conditionGroup = ((oscConditionGroup*)(*it));
				for (oscConditionArrayMember::iterator it = conditionGroup->Condition.begin(); it != conditionGroup->Condition.end(); it++)
				{
					Condition* condition = ((Condition*)(*it));
					scenarioManager->initializeCondition(condition);
					scenarioManager->addCondition(condition);
				}
			}
		}
		for (list<Act*>::iterator act_iter = scenarioManager->actList.begin(); act_iter != scenarioManager->actList.end(); act_iter++)
		{
			Act* currentAct = (*act_iter);
			if (currentAct->Conditions.exists())
			{
				//Act Start Condition
				for (oscConditionArrayMember::iterator it = currentAct->Conditions->Start->ConditionGroup.begin(); it != currentAct->Conditions->Start->ConditionGroup.end(); it++)
				{
					oscConditionGroup* conditionGroup = (oscConditionGroup*)(*it);
					for (oscConditionArrayMember::iterator it = conditionGroup->Condition.begin(); it != conditionGroup->Condition.end(); it++)
					{
						Condition* condition = ((Condition*)(*it));
						scenarioManager->initializeCondition(condition);
						currentAct->addStartCondition(condition);
					}
				}
				//Act End Condition
				for (oscConditionArrayMember::iterator it = currentAct->Conditions->End->ConditionGroup.begin(); it != currentAct->Conditions->End->ConditionGroup.end(); it++)
				{
					oscConditionGroup* conditionGroup = ((oscConditionGroup*)(*it));
					for (oscConditionArrayMember::iterator it = conditionGroup->Condition.begin(); it != conditionGroup->Condition.end(); it++)
					{
						Condition* condition = ((Condition*)(*it));
						scenarioManager->initializeCondition(condition);
						currentAct->addEndCondition(condition);
					}
				}

				// get Maneuver StartConditions
				for (list<Sequence*>::iterator sequence_iter = currentAct->sequenceList.begin(); sequence_iter != currentAct->sequenceList.end(); sequence_iter++)
				{
					Sequence* currentSequence = (*sequence_iter);
					for (list<Maneuver*>::iterator maneuver_iter = currentSequence->maneuverList.begin(); maneuver_iter != currentSequence->maneuverList.end(); maneuver_iter++)
					{
						Maneuver* currentManeuver = (*maneuver_iter);
						for (list<Event*>::iterator event_iter = currentManeuver->eventList.begin(); event_iter != currentManeuver->eventList.end(); event_iter++)
						{
							Event* currentEvent = (*event_iter);
							//currentEvent->initialize();
							if (currentEvent->StartConditions.exists())
							{
								for (oscStartConditionsArrayMember::iterator it = currentEvent->StartConditions->ConditionGroup.begin(); it != currentEvent->StartConditions->ConditionGroup.end(); it++)
								{
									oscConditionGroup* conditionGroup = (oscConditionGroup*)(*it);
									for (oscConditionArrayMember::iterator it = conditionGroup->Condition.begin(); it != conditionGroup->Condition.end(); it++)
									{
										Condition* condition = ((Condition*)(*it));
										scenarioManager->initializeCondition(condition);
										currentEvent->addCondition(condition);
									}
								}
							}
							//get trajectoryCatalogReference
							for (oscActionArrayMember::iterator it = currentEvent->Action.begin(); it != currentEvent->Action.end(); it++)
							{
								Action* currentAction = ((Action*)(*it));
								if (currentAction->Private->Routing.exists())
								{
									if (currentAction->Private->Routing->FollowTrajectory.exists())
									{
										currentAction->trajectoryCatalogReference = currentAction->Private->Routing->FollowTrajectory->CatalogReference->entryName.getValue();

										oscObjectBase* trajectoryClass = osdb->getCatalogObjectByCatalogReference("TrajectoryCatalog", currentAction->trajectoryCatalogReference);
										Trajectory* traj = ((Trajectory*)(trajectoryClass));
										if (traj == NULL)
										{
											fprintf(stderr, "Trajectory %s not found in TrajectoryCatalog\n", currentAction->trajectoryCatalogReference.c_str());
										}
										currentAction->setTrajectory(traj);


										//currentEvent->actionList.push_back(currentAction);
									}
									else if (currentAction->Private->Routing->FollowRoute.exists())
									{
										currentAction->routeCatalogReference = currentAction->Private->Routing->FollowTrajectory->CatalogReference->entryName.getValue();
										//currentEvent->actionList.push_back(currentAction);
									}
								}
								if (currentAction->Private->Longitudinal.exists())
								{

									//currentEvent->actionList.push_back(currentAction);
								}
							}
						}
					}
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

		system = vehicleUtil::RoadSystem::Instance();

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
			vehicleUtil::Road *road = system->getRoad(i);
			osg::LOD *roadGeodeLOD = new osg::LOD();

			// Tesselation //
			//
			if ((!road->isJunctionPath() && tessellateRoads == true) // normal road
				|| (road->isJunctionPath() && tessellatePaths == true) // junction path
				)
			{
				//fprintf(stderr, "1tessellateBatters %d\n", tessellateBatters);
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

		trafficSignalGroup = new osg::Group;
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
		if (vehicleUtil::RoadSystem::_tiles_y <= 400 && vehicleUtil::RoadSystem::_tiles_x <= 400)
		{
			//for (int i=0; i<=_tiles_y;i++) {
			for (int i = vehicleUtil::RoadSystem::_tiles_y; i >= 0; i--)
			{
				for (int j = 0; j <= vehicleUtil::RoadSystem::_tiles_x; j++)
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
	XMLCh *t1 = NULL, *t2 = NULL, *t3 = NULL, *t4 = NULL, *t5 = NULL;
	char *cs = NULL;
	xercesc::DOMNodeList *documentChildrenList = rootElement->getChildNodes();

	for (int childIndex = 0; childIndex < documentChildrenList->getLength(); ++childIndex)
	{
		xercesc::DOMElement *sceneryElement = dynamic_cast<xercesc::DOMElement *>(documentChildrenList->item(childIndex));
		if (sceneryElement && xercesc::XMLString::compareIString(sceneryElement->getTagName(), t1 = xercesc::XMLString::transcode("scenery")) == 0)
		{
			std::string fileString = cs = xercesc::XMLString::transcode(sceneryElement->getAttribute(t3 = xercesc::XMLString::transcode("file"))); xercesc::XMLString::release(&t3); xercesc::XMLString::release(&cs);
			std::string vpbString = cs = xercesc::XMLString::transcode(sceneryElement->getAttribute(t3 = xercesc::XMLString::transcode("vpb"))); xercesc::XMLString::release(&t3); xercesc::XMLString::release(&cs);

			std::vector<BoundingArea> voidBoundingAreaVector;
			std::vector<std::string> shapeFileNameVector;

			xercesc::DOMNodeList *sceneryChildrenList = sceneryElement->getChildNodes();
			xercesc::DOMElement *sceneryChildElement;
			for (unsigned int childIndex = 0; childIndex < sceneryChildrenList->getLength(); ++childIndex)
			{
				sceneryChildElement = dynamic_cast<xercesc::DOMElement *>(sceneryChildrenList->item(childIndex));
				if (!sceneryChildElement)
					continue;

				if (xercesc::XMLString::compareIString(sceneryChildElement->getTagName(), t3 = xercesc::XMLString::transcode("void")) == 0)
				{
					double xMin = atof(cs = xercesc::XMLString::transcode(sceneryChildElement->getAttribute(t4 = xercesc::XMLString::transcode("xMin")))); xercesc::XMLString::release(&t4); xercesc::XMLString::release(&cs);
					double yMin = atof(cs = xercesc::XMLString::transcode(sceneryChildElement->getAttribute(t4 = xercesc::XMLString::transcode("yMin")))); xercesc::XMLString::release(&t4); xercesc::XMLString::release(&cs);
					double xMax = atof(cs = xercesc::XMLString::transcode(sceneryChildElement->getAttribute(t4 = xercesc::XMLString::transcode("xMax")))); xercesc::XMLString::release(&t4); xercesc::XMLString::release(&cs);
					double yMax = atof(cs = xercesc::XMLString::transcode(sceneryChildElement->getAttribute(t4 = xercesc::XMLString::transcode("yMax")))); xercesc::XMLString::release(&t4); xercesc::XMLString::release(&cs);

					voidBoundingAreaVector.push_back(BoundingArea(osg::Vec2(xMin, yMin), osg::Vec2(xMax, yMax)));
					//voidBoundingAreaVector.push_back(BoundingArea(osg::Vec2(506426.839,5398055.357),osg::Vec2(508461.865,5399852.0)));
				}
				else if (xercesc::XMLString::compareIString(sceneryChildElement->getTagName(), t4 = xercesc::XMLString::transcode("shape")) == 0)
				{
					std::string fileString = cs = xercesc::XMLString::transcode(sceneryChildElement->getAttribute(t5 = xercesc::XMLString::transcode("file"))); xercesc::XMLString::release(&t5); xercesc::XMLString::release(&cs);
					shapeFileNameVector.push_back(fileString);
				}
				xercesc::XMLString::release(&t3); xercesc::XMLString::release(&t4);
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
					const RoadSystemHeader &header = vehicleUtil::RoadSystem::Instance()->getHeader();
					offset.set(header.xoffset, header.yoffset, 0.0);
					fprintf(stderr, "loading %s offset: %f %f\n", (xodrDirectory + "/" + vpbString).c_str(), offset[0], offset[1]);
					RoadTerrainLoader::instance()->loadTerrain(xodrDirectory + "/" + vpbString, offset, voidBoundingAreaVector, shapeFileNameVector);
				}
			}
		}
		else if (sceneryElement && xercesc::XMLString::compareIString(sceneryElement->getTagName(), t2 = xercesc::XMLString::transcode("environment")) == 0)
		{
			std::string tessellateRoadsString = cs = xercesc::XMLString::transcode(sceneryElement->getAttribute(t3 = xercesc::XMLString::transcode("tessellateRoads"))); xercesc::XMLString::release(&t3); xercesc::XMLString::release(&cs);
			if (tessellateRoadsString == "false" || tessellateRoadsString == "0")
			{
				tessellateRoads = false;
			}
			else
			{
				tessellateRoads = true;
			}

			std::string tessellatePathsString = cs = xercesc::XMLString::transcode(sceneryElement->getAttribute(t3 = xercesc::XMLString::transcode("tessellatePaths"))); xercesc::XMLString::release(&t3); xercesc::XMLString::release(&cs);
			if (tessellatePathsString == "false" || tessellatePathsString == "0")
			{
				tessellatePaths = false;
			}
			else
			{
				tessellatePaths = true;
			}

			std::string tessellateBattersString = cs = xercesc::XMLString::transcode(sceneryElement->getAttribute(t3 = xercesc::XMLString::transcode("tessellateBatters"))); xercesc::XMLString::release(&t3); xercesc::XMLString::release(&cs);
			if (tessellateBattersString == "true")
			{
				tessellateBatters = true;
			}
			else
			{
				tessellateBatters = false;
			}

			std::string tessellateObjectsString = cs = xercesc::XMLString::transcode(sceneryElement->getAttribute(t3 = xercesc::XMLString::transcode("tessellateObjects"))); xercesc::XMLString::release(&t3); xercesc::XMLString::release(&cs);
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
