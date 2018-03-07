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
#include <cover/coVRPluginSupport.h>
#include <cover/coVRPlugin.h>
#include <cover/RenderObject.h>
#include <cover/coVRFileManager.h>
#include "OpenScenarioPlugin.h"
#include <cover/coVRMSController.h>
#include <cover/OpenCOVER.h>
#include <cover/coVRPluginList.h>

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

    scenarioManager = new ScenarioManager();

    osdb = new OpenScenario::OpenScenarioBase();
    // set our own object factory so that our own classes are created and not the bas osc* classes
    OpenScenario::oscFactories::instance()->setObjectFactory(new myFactory());
    osdb->setFullReadCatalogs(true);
    //todo coTrafficSimulation::useInstance();
    fprintf(stderr, "OpenScenario::OpenScenario\n");
}

// this is called if the plugin is removed at runtime
OpenScenarioPlugin::~OpenScenarioPlugin()
{

    // coTrafficSimulation::freeInstance();
    fprintf(stderr, "OpenScenarioPlugin::~OpenScenarioPlugin\n");
}

bool OpenScenarioPlugin::update()
{
    return true;
}
void OpenScenarioPlugin::preFrame()
{
    scenarioManager->simulationTime = scenarioManager->simulationTime + opencover::cover->frameDuration();
    //cout << "TIME: " << scenarioManager->simulationTime << endl;
    list<Entity*> usedEntity;
    list<Entity*> unusedEntity = scenarioManager->entityList;
    list<Entity*> entityList_temp = scenarioManager->entityList;
    entityList_temp.sort();unusedEntity.sort();
    scenarioManager->conditionManager();

    if (scenarioManager->scenarioCondition)
    {
        for(list<Act*>::iterator act_iter = scenarioManager->actList.begin(); act_iter != scenarioManager->actList.end(); act_iter++)
        {
            Act* currrentAct = (*act_iter);
            //check act start conditions
            if ((*act_iter)->actCondition)
            {
                for(list<Maneuver*>::iterator maneuver_iter = (*act_iter)->maneuverList.begin(); maneuver_iter != (*act_iter)->maneuverList.end(); maneuver_iter++)
                {
                    Maneuver* currentManeuver = (*maneuver_iter);
                    //check maneuver start conditions
                    if ((*maneuver_iter)->maneuverCondition)
                    {
                        if((*maneuver_iter)->maneuverType == "followTrajectory")
                        {
                            for(list<Trajectory*>::iterator trajectory_iter = (*maneuver_iter)->trajectoryList.begin(); trajectory_iter != (*maneuver_iter)->trajectoryList.end(); trajectory_iter++)
                            {
                                Trajectory* currentTrajectory = (*trajectory_iter);
                                for(list<Entity*>::iterator activeEntity = (*maneuver_iter)->activeEntityList.begin(); activeEntity != (*maneuver_iter)->activeEntityList.end(); activeEntity++)
                                {
                                    Position* currentPos;
                                    Entity* currentEntity = (*activeEntity);
                                    // check if Trajectory is about to start or Entity arrived at vertice
                                    if((*activeEntity)->totalDistance == 0)
                                    {
                                        (*activeEntity)->setRefPos();
                                        currentPos = ((Position*)((*trajectory_iter)->Vertex[(*activeEntity)->visitedVertices]->Position.getObject()));

                                        osg::Vec3 nextTargetPos0 = currentPos->getAbsolutePosition((*activeEntity),system, scenarioManager->entityList);
                                        oscShape* currentShape = (*trajectory_iter)->Vertex[(*activeEntity)->visitedVertices]->Shape.getObject();

                                        if(currentShape->Spline.exists())
                                        {
                                            (*activeEntity)->spline = new Spline();
                                            (*activeEntity)->spline->poly3Spline(currentPos);
                                            (*activeEntity)->setActiveShape("Spline");
                                        }

                                        (*activeEntity)->setTrajectoryDirection(nextTargetPos0);

                                        if((*trajectory_iter)->domain.getValue() == 0)
                                        { //if domain is set to "time"
                                            // calculate speed from trajectory vertices
                                            (*activeEntity)->getTrajSpeed(0.01);
                                        }
                                    }
                                    if((*activeEntity)->activeShape == "Spline")
                                    {
                                        osg::Vec3 test =(*activeEntity)->spline->getSplinePos((*activeEntity)->visitedSplineVertices);
                                        (*activeEntity)->setSplinePos(test);
                                        (*activeEntity)->followSpline();
                                    }
                                    else
                                    {
                                        (*activeEntity)->followTrajectory((*trajectory_iter)->verticesCounter,(*maneuver_iter)->finishedEntityList);
                                    }



                                    unusedEntity.remove(*activeEntity);

                                    usedEntity.push_back((*activeEntity));
                                    usedEntity.sort();usedEntity.unique();
                                }
                                //set_difference(entityList_temp.begin(), entityList_temp.end(), usedEntity.begin(), usedEntity.end(), inserter(unusedEntity, unusedEntity.begin()));
                            }
                        }
                        if((*maneuver_iter)->maneuverType == "break")
                        {
                            for(list<Entity*>::iterator activeEntity = (*act_iter)->activeEntityList.begin(); activeEntity != (*act_iter)->activeEntityList.end(); activeEntity++)
                            {
                                (*maneuver_iter)->changeSpeedOfEntity((*activeEntity),opencover::cover->frameDuration());
                            }
                        }
                    }
                }
            }
        }
        for(list<Entity*>::iterator entity_iter = unusedEntity.begin(); entity_iter != unusedEntity.end(); entity_iter++)
        {
            Entity* currentEntity = (*entity_iter);
            (*entity_iter)->moveLongitudinal();
            (*entity_iter)->entityGeometry->setPosition((*entity_iter)->entityPosition,(*entity_iter)->directionVector);
            //(*activeEntity)->entityGeometry->move(opencover::cover->frameDuration());
            usedEntity.clear();
        }
        scenarioManager->endTrajectoryCheck();
    }
    else
    {
        // Scenario end

        bool doExit = covise::coCoviseConfig::isOn("COVER.Plugin.OpenScenario.ExitOnScenarioEnd", true);
        if (doExit)
        {
            OpenCOVER::instance()->setExitFlag(true);
            coVRPluginList::instance()->requestQuit(true);
        }
        //fprintf(stderr, "END\n");
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
    return plugin->loadOSCFile(filename,g,key);
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

        //neu
        std::string filename(file);
        xoscDirectory.clear();
        if (filename[0] != '/' && filename[0] != '\\' && (!(filename[1] == ':' && (filename[2] == '/' || filename[2] == '\\'))))
        { // / or backslash or c:/
            char *workingDir = getcwd(NULL, 0);
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
        //neu
        return -1;
    }

    //load xodr
    if (osdb->RoadNetwork.getObject() != NULL)
    {
        std::string xodrName_st = osdb->RoadNetwork->Logics->filepath.getValue();
        const char * xodrName = xodrName_st.c_str();
        loadRoadSystem(xodrName);
    }

    //load ScenGraph
    std::string geometryFile = osdb->RoadNetwork->SceneGraph->filepath.getValue();
    coVRFileManager::instance()->loadFile(geometryFile.c_str());

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
                    if (oscTrafficDefinition* traffic = source->TrafficDefinition.getObject())
                    {
                        oscVehicleDistribution* vd = traffic->VehicleDistribution.getObject();
                        oscDriverDistribution* dd = traffic->DriverDistribution.getObject();
                        if (vd != NULL && dd != NULL)
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
                                for(int i=0;i<vd->Vehicle.size();i++)
                                {
                                    oscVehicle *vehicle = vd->Vehicle[i];
                                    vs->addVehicleRatio(vehicle->name.getValue(), vd->percentage.getValue());
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
        scenarioManager->entityList.push_back(new Entity(entity->name.getValue(), entity->CatalogReference->entryName.getValue()));
        cout << "Entity: " <<  entity->name.getValue() << " initialized" << endl;
    }

    //get initial position and speed of entities
    for (list<Entity*>::iterator entity_iter = scenarioManager->entityList.begin(); entity_iter != scenarioManager->entityList.end(); entity_iter++)
    {
        Entity *currentTentity = (*entity_iter);
        oscObjectBase *vehicleCatalog = osdb->getCatalogObjectByCatalogReference("VehicleCatalog", (*entity_iter)->catalogReferenceName);
        oscVehicle* vehicle = ((oscVehicle*)(vehicleCatalog));
        for(int i=0;i<vehicle->ParameterDeclaration->Parameter.size();i++)
        {
            if (vehicle->ParameterDeclaration->Parameter[i]->name.getValue() == "CameraX")
            {
                double X=0.0, Y=0.0, Z=0.0, H=0.0, P=0.0, R=0.0, FOV=0.0;
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

                CameraSensor *camera = new CameraSensor(currentTentity, vehicle, rotMat*cameraMat, FOV);
                cameras.push_back(camera);
                if (currentCamera == NULL)
                    currentCamera = camera;
                break;
            }
        }

        for (oscFileArrayMember::iterator it = vehicle->Properties->File.begin(); it !=  vehicle->Properties->File.end(); it++)
        {
            oscFile* file = ((oscFile*)(*it));
            currentTentity->filepath = file->filepath.getValue();
        }
        for (oscPrivateArrayMember::iterator it = osdb->Storyboard->Init->Actions->Private.begin(); it != osdb->Storyboard->Init->Actions->Private.end(); it++)
        {
            oscPrivate* actions_private = ((oscPrivate*)(*it));
            if(currentTentity->getName() == actions_private->object.getValue())
            {
                for (oscPrivateActionArrayMember::iterator it2 = actions_private->Action.begin(); it2 != actions_private->Action.end(); it2++)
                {
                    oscPrivateAction* action = ((oscPrivateAction*)(*it2));
                    if(action->Longitudinal.exists())
                    {
                        currentTentity->setSpeed(action->Longitudinal->Speed->Target->Absolute->value.getValue());
                    }
                    if(action->Position.exists())
                    {
                        if (action->Position->Lane.exists())
                        {

                            currentTentity->roadId = action->Position->Lane->roadId.getValue();
                            currentTentity->laneId = action->Position->Lane->laneId.getValue();
                            currentTentity->inits = action->Position->Lane->s.getValue();

                            ReferencePosition* refPos = new ReferencePosition();
                            refPos->initFromLane(action->Position->Lane->roadId.getValue(),action->Position->Lane->laneId.getValue(),action->Position->Lane->s.getValue(),system);
                            currentTentity->refPos = refPos;

                            int roadId = atoi(currentTentity->roadId.c_str());
                            currentTentity->setInitEntityPosition(system->getRoad(roadId));
                        }
                        else if(action->Position->World.exists())
                        {
                            Position* pos = (Position*)(action->Position.getObject());
                            osg::Vec3 initPosition = pos->getAbsoluteWorld();
                            osg::Vec3 initDirVec = pos->hpr2directionVector();

                            currentTentity->setInitEntityPosition(initPosition, initDirVec);
                            currentTentity->setPosition(initPosition);
                        }
                    }
                }

            }
        }

    }

    //initialize acts
    for (oscStoryArrayMember::iterator it = osdb->Storyboard->Story.begin(); it != osdb->Storyboard->Story.end(); it++)
    {
        oscStory* story = ((oscStory*)(*it));
        for (oscActArrayMember::iterator it = story->Act.begin(); it != story->Act.end(); it++)
        {
            Act* act = ((Act*)(*it));// these are not oscAct instances any more but our own Act

            list<Entity*> activeEntityList_temp;
            for (oscActorsArrayMember::iterator it = act->Sequence->Actors->Entity.begin(); it != act->Sequence->Actors->Entity.end(); it++)
            {
                oscEntity* namedEntity = ((oscEntity*)(*it));
                if (namedEntity->name.getValue() != "$owner")
                {
                    activeEntityList_temp.push_back(scenarioManager->getEntityByName(namedEntity->name.getValue()));
                }
                else{activeEntityList_temp.push_back(scenarioManager->getEntityByName(story->owner.getValue()));
                }
                cout << "Entity: " << story->owner.getValue() << " allocated to " << act->getName() << endl;
            }
            list<Maneuver*> maneuverList_temp;
            for (oscManeuverArrayMember::iterator it = act->Sequence->Maneuver.begin(); it != act->Sequence->Maneuver.end(); it++)
            {
                Maneuver* maneuver = ((Maneuver*)(*it)); // these are not oscManeuver instances any more but our own Maneuver
                maneuver->initialize(activeEntityList_temp);
                maneuverList_temp.push_back(maneuver);
                cout << "Manuever: " << maneuver->getName() << " created" << endl;
            }
            act->initialize(act->Sequence->numberOfExecutions.getValue(), maneuverList_temp, activeEntityList_temp);
            scenarioManager->actList.push_back(act);
            cout << "Act: " <<  act->getName() << " initialized" << endl;
            maneuverList_temp.clear();
            activeEntityList_temp.clear();
        }
    }

    //get Conditions
    for (oscConditionArrayMember::iterator it = osdb->Storyboard->End->ConditionGroup.begin(); it != osdb->Storyboard->End->ConditionGroup.end(); it++)
    {
        oscConditionGroup* conditionGroup = ((oscConditionGroup*)(*it));
        for (oscConditionArrayMember::iterator it = conditionGroup->Condition.begin(); it != conditionGroup->Condition.end(); it++)
        {
            oscCondition* condition = ((oscCondition*)(*it));
            if(condition->ByValue.exists())
            {
                scenarioManager->endConditionType = "time";
                scenarioManager->endTime = condition->ByValue->SimulationTime->value.getValue();
            }
        }
    }
    for (list<Act*>::iterator act_iter = scenarioManager->actList.begin(); act_iter != scenarioManager->actList.end(); act_iter++)
    {
        for (list<Maneuver*>::iterator maneuver_iter = (*act_iter)->maneuverList.begin(); maneuver_iter != (*act_iter)->maneuverList.end(); maneuver_iter++)
        {
            for (oscStoryArrayMember::iterator it = osdb->Storyboard->Story.begin(); it != osdb->Storyboard->Story.end(); it++)
            {
                oscStory* story = ((oscStory*)(*it));
                for (oscActArrayMember::iterator it = story->Act.begin(); it != story->Act.end(); it++)
                {
                    oscAct* act = ((oscAct*)(*it));
                    //Act Start Condition
                    for (oscConditionArrayMember::iterator it = act->Conditions->Start->ConditionGroup->Condition.begin(); it != act->Conditions->Start->ConditionGroup->Condition.end(); it++)
                    {
                        oscCondition* condition = ((oscCondition*)(*it));
                        if(condition->ByValue.exists())
                        {
                            if ((*act_iter)->getName() == act->name.getValue())
                            {
                                (*act_iter)->startConditionType = "time";
                                (*act_iter)->startTime = condition->ByValue->SimulationTime->value.getValue();
                            }
                        }
                    }
                    //Act End Condition
                    for (oscConditionArrayMember::iterator it = act->Conditions->End->ConditionGroup.begin(); it != act->Conditions->End->ConditionGroup.end(); it++)
                    {
                        oscConditionGroup* conditionGroup = ((oscConditionGroup*)(*it));
                        for (oscConditionArrayMember::iterator it = conditionGroup->Condition.begin(); it != conditionGroup->Condition.end(); it++)
                        {
                            oscCondition* condition = ((oscCondition*)(*it));
                            if(condition->ByValue.exists())
                            {
                                if ((*act_iter)->getName() == act->name.getValue())
                                {
                                    (*act_iter)->endConditionType = "time";
                                    (*act_iter)->endTime = condition->ByValue->SimulationTime->value.getValue();
                                }
                            }
                        }
                    }
                    for (oscManeuverArrayMember::iterator it = act->Sequence->Maneuver.begin(); it != act->Sequence->Maneuver.end(); it++)
                    {
                        oscManeuver* maneuver = ((oscManeuver*)(*it));
                        for (oscEventArrayMember::iterator it = maneuver->Event.begin(); it != maneuver->Event.end(); it++)
                        {
                            oscEvent* event = ((oscEvent*)(*it));
                            for (oscConditionsArrayMember::iterator it = event->Conditions.begin(); it != event->Conditions.end(); it++)
                            {
                                oscConditions* conditions = ((oscConditions*)(*it));

                                //get Maneuver conditions
                                if(conditions->Start.exists())
                                {
                                    for (oscConditionArrayMember::iterator it = conditions->Start->ConditionGroup->Condition.begin(); it != conditions->Start->ConditionGroup->Condition.end(); it++)
                                    {
                                        oscCondition* condition = ((oscCondition*)(*it));
                                        if(condition->ByValue.exists())
                                        {
                                            if ((*maneuver_iter)->getName() == maneuver->name.getValue())
                                            {
                                                (*maneuver_iter)->startConditionType="time";
                                                (*maneuver_iter)->startTime = condition->ByValue->SimulationTime->value.getValue();
                                            }
                                        }
                                        if(condition->ByState.exists())
                                        {
                                            if ((*maneuver_iter)->getName() == maneuver->name.getValue())
                                            {
                                                (*maneuver_iter)->startConditionType="termination";
                                                (*maneuver_iter)->startAfterManeuver = condition->ByState->AfterTermination->name.getValue();
                                            }
                                        }
                                        if(condition->ByEntity.exists())
                                        {
                                            if ((*maneuver_iter)->getName() == maneuver->name.getValue())
                                            {
                                                (*maneuver_iter)->startConditionType="distance";
                                                (*maneuver_iter)->passiveCarName = condition->ByEntity->EntityCondition->RelativeDistance->entity.getValue();
                                                (*maneuver_iter)->relativeDistance = condition->ByEntity->EntityCondition->RelativeDistance->value.getValue();

                                                for (oscEntityArrayMember::iterator it = condition->ByEntity->TriggeringEntities->Entity.begin(); it != condition->ByEntity->TriggeringEntities->Entity.end(); it++)
                                                {
                                                    oscEntity* entity = ((oscEntity*)(*it));
                                                    (*maneuver_iter)->activeCarName = entity->name.getValue();
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            //get trajectoryCatalogReference
                            for (oscActionArrayMember::iterator it = event->Action.begin(); it != event->Action.end(); it++)
                            {
                                oscAction* action = ((oscAction*)(*it));

                                if(action->Private->Routing.exists())
                                {
                                    if ((*maneuver_iter)->getName() == maneuver->name.getValue())
                                    {
                                        if (action->Private->Routing->FollowTrajectory.exists())
                                        {
                                            (*maneuver_iter)->maneuverType = "followTrajectory";
                                            (*maneuver_iter)->trajectoryCatalogReference = action->Private->Routing->FollowTrajectory->CatalogReference->entryName.getValue();
                                        }
                                        else if (action->Private->Routing->FollowRoute.exists())
                                        {
                                            (*maneuver_iter)->maneuverType = "FollowRoute";
                                            (*maneuver_iter)->routeCatalogReference = action->Private->Routing->FollowTrajectory->CatalogReference->entryName.getValue();
                                        }
                                    }
                                }
                                if(action->Private->Longitudinal.exists())
                                {
                                    if ((*maneuver_iter)->getName() == maneuver->name.getValue())
                                    {
                                        (*maneuver_iter)->maneuverType="break";
                                        (*maneuver_iter)->targetSpeed = action->Private->Longitudinal->Speed->Target->Absolute->value.getValue();
                                    }
                                }
                            }


                        }
                    }
                }
            }
        }
    }

    //acess maneuvers in acts
    for(list<Act*>::iterator act_iter = scenarioManager->actList.begin(); act_iter != scenarioManager->actList.end(); act_iter++)
    {
        for(list<Maneuver*>::iterator maneuver_iter = (*act_iter)->maneuverList.begin(); maneuver_iter != (*act_iter)->maneuverList.end(); maneuver_iter++)
        {
            if((*maneuver_iter)->trajectoryCatalogReference != "")
            {
                oscObjectBase *trajectoryClass = osdb->getCatalogObjectByCatalogReference("TrajectoryCatalog", (*maneuver_iter)->trajectoryCatalogReference);
                Trajectory* traj = ((Trajectory*)(trajectoryClass));
                (*maneuver_iter)->trajectoryList.push_back(traj);

                int verticesCounter = traj->Vertex.size();
                traj->initialize(verticesCounter);

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
