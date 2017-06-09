/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "VehicleFactory.h"

#include "cover/coVRMSController.h"
#include "RoadSystem/RoadSystem.h"
#include "AgentVehicle.h"
#include "HumanVehicle.h"
#include "CarGeometry.h"

#include "VehicleManager.h"
#include "TrafficSimulation.h"

VehicleFactory *VehicleFactory::__instance = NULL;

VehicleFactory *VehicleFactory::Instance()
{
    if (__instance == NULL)
    {
        __instance = new VehicleFactory();
    }
    return __instance;
}

void VehicleFactory::Destroy()
{
    delete __instance;
    __instance = NULL;
}

VehicleFactory::VehicleFactory()
{
}

void VehicleFactory::deleteRoadVehicle(Vehicle *veh)
{
    std::set<Vehicle *>::iterator vehIt = blueprintVehicleSet.find(veh);
    if (vehIt == blueprintVehicleSet.end())
    {
        delete veh;
    }
    else
    {
        AgentVehicle *agent = dynamic_cast<AgentVehicle *>(veh);
        if (agent)
        {
            agent->getCarGeometry()->removeFromSceneGraph();
        }
    }
}

Vehicle *VehicleFactory::cloneRoadVehicle(std::string id, std::string name, Road *road, double pos, int lane, double vel, int dir)
{
    std::map<std::string, Vehicle *>::iterator vehIt = roadVehicleMap.find(id);
    if (vehIt != roadVehicleMap.end())
    {
        AgentVehicle *veh = dynamic_cast<AgentVehicle *>(vehIt->second);
        if (veh)
        {
            return (new AgentVehicle(veh, name, veh->getVehicleParameters(), road, pos, lane, vel, dir));
        }
        else
        {
            return NULL;
        }
    }
    else
    {
        return NULL;
    }
}

Vehicle *VehicleFactory::createRoadVehicle(std::string id, std::string name, RoadVehicleType, IntelligenceType intelType, std::string modelfile)
{
    Vehicle *veh = NULL;

    switch (intelType)
    {
    case INTELLIGENCE_AGENT:
        veh = new AgentVehicle(name, new CarGeometry(name, modelfile, false));
        break;
    case INTELLIGENCE_DONKEY:
        break;
    case INTELLIGENCE_SCRIPTING:
    case INTELLIGENCE_HUMAN:
        veh = new HumanVehicle(name);
        break;
    default:
        veh = new AgentVehicle(name, new CarGeometry(name, modelfile, false));
    }
    if (veh)
    {
        roadVehicleMap.insert(std::pair<std::string, Vehicle *>(id, veh));
        blueprintVehicleSet.insert(veh);
    }

    return veh;
}

Vehicle *VehicleFactory::createRoadVehicle(std::string id, std::string name, Road *road, double pos, int lane, int dir, double vel, RoadVehicleType, IntelligenceType intelType, std::string modelfile)
{
    Vehicle *veh = NULL;

    switch (intelType)
    {
    case INTELLIGENCE_AGENT:
        veh = new AgentVehicle(name, new CarGeometry(name, modelfile), VehicleParameters(), road, pos, lane, vel, dir);
        break;
    case INTELLIGENCE_DONKEY:
        break;
    case INTELLIGENCE_SCRIPTING:
    case INTELLIGENCE_HUMAN:
        veh = new HumanVehicle(name);
        break;
    default:
        veh = new AgentVehicle(name, new CarGeometry(name, modelfile), VehicleParameters(), road, pos, lane, vel, dir);
    }
    if (veh)
    {
        roadVehicleMap.insert(std::pair<std::string, Vehicle *>(id, veh));
        blueprintVehicleSet.insert(veh);
    }

    return veh;
}

Vehicle *VehicleFactory::createRoadVehicle(std::string id, std::string name, std::string vehTypeString, std::string intelTypeString, std::string modelfile)
{
    RoadVehicleType vehType = ROADVEHICLE_CAR;
    IntelligenceType intelType = INTELLIGENCE_AGENT;

    if (vehTypeString == "car")
    {
        vehType = ROADVEHICLE_CAR;
    }

    if (intelTypeString == "agent")
    {
        intelType = INTELLIGENCE_AGENT;
    }
    else if (intelTypeString == "donkey")
    {
        intelType = INTELLIGENCE_DONKEY;
    }
    else if (intelTypeString == "human")
    {
        intelType = INTELLIGENCE_HUMAN;
    }

    return createRoadVehicle(id, name, vehType, intelType, modelfile);
}

Vehicle *VehicleFactory::createRoadVehicle(std::string id, std::string name, std::string roadId, double pos, int lane, int dir, double vel, std::string vehTypeString, std::string intelTypeString, std::string modelfile)
{
    Road *road = RoadSystem::Instance()->getRoad(roadId);
    RoadVehicleType vehType = ROADVEHICLE_CAR;
    IntelligenceType intelType = INTELLIGENCE_AGENT;

    if (vehTypeString == "car")
    {
        vehType = ROADVEHICLE_CAR;
    }

    if (intelTypeString == "agent")
    {
        intelType = INTELLIGENCE_AGENT;
    }
    else if (intelTypeString == "donkey")
    {
        intelType = INTELLIGENCE_DONKEY;
    }
    else if (intelTypeString == "human")
    {
        intelType = INTELLIGENCE_HUMAN;
    }

    return createRoadVehicle(id, name, road, pos, lane, dir, vel, vehType, intelType, modelfile);
}

void VehicleFactory::parseOpenDrive(xercesc::DOMElement *rootElement, const std::string &xodrDirectory)
{
    xercesc::DOMNodeList *documentChildrenList = rootElement->getChildNodes();

    for (int childIndex = 0; childIndex < documentChildrenList->getLength(); ++childIndex)
    {
        xercesc::DOMElement *vehiclesElement = dynamic_cast<xercesc::DOMElement *>(documentChildrenList->item(childIndex));
        if (vehiclesElement && xercesc::XMLString::compareIString(vehiclesElement->getTagName(), xercesc::XMLString::transcode("vehicles")) == 0)
        {
            // LOD //
            double rangeLOD = 3.4e38;
            if (vehiclesElement->hasAttribute(xercesc::XMLString::transcode("rangeLOD")))
            {
                rangeLOD = atof(xercesc::XMLString::transcode(vehiclesElement->getAttribute(xercesc::XMLString::transcode("rangeLOD"))));
            }

            if (vehiclesElement->hasAttribute(xercesc::XMLString::transcode("maximumNumber")))
            {
                unsigned int maximumNumber = atoi(xercesc::XMLString::transcode(vehiclesElement->getAttribute(xercesc::XMLString::transcode("maximumNumber"))));
                VehicleManager::Instance()->setMaximumNumberOfVehicles(maximumNumber);
            }

            // NoPassThreshold determines the minimum amount of space needed in order to consider passing the vehicle in front
            double passThreshold = 0.5;
            if (vehiclesElement->hasAttribute(xercesc::XMLString::transcode("passThreshold")))
            {
                passThreshold = atof(xercesc::XMLString::transcode(vehiclesElement->getAttribute(xercesc::XMLString::transcode("passThreshold"))));
            }

            xercesc::DOMNodeList *vehiclesChildrenList = vehiclesElement->getChildNodes();
            xercesc::DOMElement *vehiclesChildElement;
            for (int childIndex = 0; childIndex < vehiclesChildrenList->getLength(); ++childIndex)
            {
                vehiclesChildElement = dynamic_cast<xercesc::DOMElement *>(vehiclesChildrenList->item(childIndex));
                if (vehiclesChildElement && xercesc::XMLString::compareIString(vehiclesChildElement->getTagName(), xercesc::XMLString::transcode("roadVehicle")) == 0)
                {
                    std::string idString = xercesc::XMLString::transcode(vehiclesChildElement->getAttribute(xercesc::XMLString::transcode("id")));
                    std::string nameString = xercesc::XMLString::transcode(vehiclesChildElement->getAttribute(xercesc::XMLString::transcode("name")));
                    std::string typeString = xercesc::XMLString::transcode(vehiclesChildElement->getAttribute(xercesc::XMLString::transcode("type")));
                    std::string roadIdString = "1";
                    double pos = 0.0;
                    int lane = -1;
                    int dir = 1;
                    double vel = 100;
                    VehicleParameters vehPars;
                    vehPars.rangeLOD = rangeLOD; // can be overruled later by individual cars
                    vehPars.passThreshold = passThreshold; // minimum space needed to pass another vehicle
                    std::string intelString = "agent";
                    //  std::string vehicleString = "car"; NEU 02-02-2011
                    std::string vehicleString = typeString;
                    vehPars.obstacleType = typeString;
                    //std::cout << " vehString: " << vehicleString << " typeString: " << typeString << std::endl ;
                    std::string modelFile = "cars/hotcar.osg";
                    bool initialState = false;

                    xercesc::DOMNodeList *roadVehicleChildrenList = vehiclesChildElement->getChildNodes();
                    xercesc::DOMElement *roadVehicleChildElement;
                    for (int childIndex = 0; childIndex < roadVehicleChildrenList->getLength(); ++childIndex)
                    {
                        roadVehicleChildElement = dynamic_cast<xercesc::DOMElement *>(roadVehicleChildrenList->item(childIndex));
                        if (roadVehicleChildElement && xercesc::XMLString::compareIString(roadVehicleChildElement->getTagName(), xercesc::XMLString::transcode("intelligence")) == 0)
                        {
                            intelString = xercesc::XMLString::transcode(roadVehicleChildElement->getAttribute(xercesc::XMLString::transcode("type")));
                        }
                        else if (roadVehicleChildElement && xercesc::XMLString::compareIString(roadVehicleChildElement->getTagName(), xercesc::XMLString::transcode("geometry")) == 0)
                        {
                            if (roadVehicleChildElement->hasAttribute(xercesc::XMLString::transcode("rangeLOD")))
                                vehPars.rangeLOD = atof(xercesc::XMLString::transcode(roadVehicleChildElement->getAttribute(xercesc::XMLString::transcode("rangeLOD"))));

                            modelFile = xercesc::XMLString::transcode(roadVehicleChildElement->getAttribute(xercesc::XMLString::transcode("modelFile")));
                            if (modelFile[0] != '/' && modelFile[0] != '\\' && modelFile[1] != ':')
                                modelFile = xodrDirectory + "/" + modelFile;
                        }
                        else if (roadVehicleChildElement && xercesc::XMLString::compareIString(roadVehicleChildElement->getTagName(), xercesc::XMLString::transcode("dynamics")) == 0)
                        {
                            xercesc::DOMNodeList *dynamicsChildrenList = roadVehicleChildElement->getChildNodes();
                            xercesc::DOMElement *dynamicsChildElement;
                            for (int childIndex = 0; childIndex < dynamicsChildrenList->getLength(); ++childIndex)
                            {
                                dynamicsChildElement = dynamic_cast<xercesc::DOMElement *>(dynamicsChildrenList->item(childIndex));
                                if (dynamicsChildElement && xercesc::XMLString::compareIString(dynamicsChildElement->getTagName(), xercesc::XMLString::transcode("maximumAcceleration")) == 0)
                                {
                                    vehPars.accMax = atof(xercesc::XMLString::transcode(dynamicsChildElement->getAttribute(xercesc::XMLString::transcode("value"))));
                                }
                                else if (dynamicsChildElement && xercesc::XMLString::compareIString(dynamicsChildElement->getTagName(), xercesc::XMLString::transcode("indicatoryVelocity")) == 0)
                                {
                                    vehPars.dUtarget = atof(xercesc::XMLString::transcode(dynamicsChildElement->getAttribute(xercesc::XMLString::transcode("value"))));
                                }
                                else if (dynamicsChildElement && xercesc::XMLString::compareIString(dynamicsChildElement->getTagName(), xercesc::XMLString::transcode("maximumCrossAcceleration")) == 0)
                                {
                                    vehPars.accCrossmax = atof(xercesc::XMLString::transcode(dynamicsChildElement->getAttribute(xercesc::XMLString::transcode("value"))));
                                }
                            }
                        }
                        else if (roadVehicleChildElement && xercesc::XMLString::compareIString(roadVehicleChildElement->getTagName(), xercesc::XMLString::transcode("behaviour")) == 0)
                        {
                            xercesc::DOMNodeList *behaviourChildrenList = roadVehicleChildElement->getChildNodes();
                            xercesc::DOMElement *behaviourChildElement;
                            for (int childIndex = 0; childIndex < behaviourChildrenList->getLength(); ++childIndex)
                            {
                                behaviourChildElement = dynamic_cast<xercesc::DOMElement *>(behaviourChildrenList->item(childIndex));
                                if (behaviourChildElement && xercesc::XMLString::compareIString(behaviourChildElement->getTagName(), xercesc::XMLString::transcode("minimumGap")) == 0)
                                {
                                    vehPars.deltaSmin = atof(xercesc::XMLString::transcode(behaviourChildElement->getAttribute(xercesc::XMLString::transcode("value"))));
                                }
                                else if (behaviourChildElement && xercesc::XMLString::compareIString(behaviourChildElement->getTagName(), xercesc::XMLString::transcode("pursueTime")) == 0)
                                {
                                    vehPars.respTime = atof(xercesc::XMLString::transcode(behaviourChildElement->getAttribute(xercesc::XMLString::transcode("value"))));
                                }
                                else if (behaviourChildElement && xercesc::XMLString::compareIString(behaviourChildElement->getTagName(), xercesc::XMLString::transcode("comfortableDeceleration")) == 0)
                                {
                                    vehPars.decComf = atof(xercesc::XMLString::transcode(behaviourChildElement->getAttribute(xercesc::XMLString::transcode("value"))));
                                }
                                else if (behaviourChildElement && xercesc::XMLString::compareIString(behaviourChildElement->getTagName(), xercesc::XMLString::transcode("saveDeceleration")) == 0)
                                {
                                    vehPars.decSave = atof(xercesc::XMLString::transcode(behaviourChildElement->getAttribute(xercesc::XMLString::transcode("value"))));
                                }
                                else if (behaviourChildElement && xercesc::XMLString::compareIString(behaviourChildElement->getTagName(), xercesc::XMLString::transcode("approachFactor")) == 0)
                                {
                                    vehPars.approachFactor = atof(xercesc::XMLString::transcode(behaviourChildElement->getAttribute(xercesc::XMLString::transcode("value"))));
                                }
                                else if (behaviourChildElement && xercesc::XMLString::compareIString(behaviourChildElement->getTagName(), xercesc::XMLString::transcode("laneChangeTreshold")) == 0)
                                {
                                    vehPars.lcTreshold = atof(xercesc::XMLString::transcode(behaviourChildElement->getAttribute(xercesc::XMLString::transcode("value"))));
                                }
                                else if (behaviourChildElement && xercesc::XMLString::compareIString(behaviourChildElement->getTagName(), xercesc::XMLString::transcode("politenessFactor")) == 0)
                                {
                                    vehPars.politeFactor = atof(xercesc::XMLString::transcode(behaviourChildElement->getAttribute(xercesc::XMLString::transcode("value"))));
                                }
                                else if (behaviourChildElement && xercesc::XMLString::compareIString(behaviourChildElement->getTagName(), xercesc::XMLString::transcode("panicDistance")) == 0)
                                {
                                    vehPars.panicDistance = atof(xercesc::XMLString::transcode(behaviourChildElement->getAttribute(xercesc::XMLString::transcode("value"))));
                                }
                            }
                        }
                        else if (roadVehicleChildElement && xercesc::XMLString::compareIString(roadVehicleChildElement->getTagName(), xercesc::XMLString::transcode("initialState")) == 0)
                        {
                            initialState = true;
                            roadIdString = xercesc::XMLString::transcode(roadVehicleChildElement->getAttribute(xercesc::XMLString::transcode("roadId")));
                            pos = atof(xercesc::XMLString::transcode(roadVehicleChildElement->getAttribute(xercesc::XMLString::transcode("position"))));
                            lane = atoi(xercesc::XMLString::transcode(roadVehicleChildElement->getAttribute(xercesc::XMLString::transcode("lane"))));
                            std::string dirString = xercesc::XMLString::transcode(roadVehicleChildElement->getAttribute(xercesc::XMLString::transcode("direction")));
                            if (dirString == "forward")
                            {
                                dir = 1;
                            }
                            else if (dirString == "backward")
                            {
                                dir = -1;
                            }
                            else
                            {
                                dir = 1;
                            }
                            vel = atof(xercesc::XMLString::transcode(roadVehicleChildElement->getAttribute(xercesc::XMLString::transcode("velocity"))));
                        }
                    }

                    if (intelString == "agent")
                    {
                        Vehicle *veh = NULL;
                        if (initialState)
                        {
                            veh = createRoadVehicle(idString, nameString, roadIdString, pos, lane, dir, vel, vehicleString, intelString, modelFile);
                            VehicleManager::Instance()->addVehicle(veh);

                            for (int childIndex = 0; childIndex < roadVehicleChildrenList->getLength(); ++childIndex)
                            {
                                roadVehicleChildElement = dynamic_cast<xercesc::DOMElement *>(roadVehicleChildrenList->item(childIndex));
                                if (roadVehicleChildElement && xercesc::XMLString::compareIString(roadVehicleChildElement->getTagName(), xercesc::XMLString::transcode("route")) == 0)
                                {
                                    std::string routeRepeatString = xercesc::XMLString::transcode(roadVehicleChildElement->getAttribute(xercesc::XMLString::transcode("repeat")));
                                    if (routeRepeatString == "true")
                                    {
                                        AgentVehicle *routeVeh = dynamic_cast<AgentVehicle *>(veh);
                                        if (routeVeh)
                                        {
                                            routeVeh->setRepeatRoute(true);
                                        }
                                    }

                                    xercesc::DOMNodeList *routeChildrenList = roadVehicleChildElement->getChildNodes();
                                    xercesc::DOMElement *routeChildElement;
                                    for (int childIndex = 0; childIndex < routeChildrenList->getLength(); ++childIndex)
                                    {
                                        routeChildElement = dynamic_cast<xercesc::DOMElement *>(routeChildrenList->item(childIndex));
                                        if (routeChildElement && xercesc::XMLString::compareIString(routeChildElement->getTagName(), xercesc::XMLString::transcode("road")) == 0)
                                        {
                                            std::string idString = xercesc::XMLString::transcode(routeChildElement->getAttribute(xercesc::XMLString::transcode("id")));
                                            Road *road = RoadSystem::Instance()->getRoad(idString);
                                            std::string dirString = xercesc::XMLString::transcode(routeChildElement->getAttribute(xercesc::XMLString::transcode("contactPoint")));
                                            int dir = (dirString == "end") ? -1 : 1;
                                            AgentVehicle *routeVeh = dynamic_cast<AgentVehicle *>(veh);
                                            if (routeVeh && road)
                                            {
                                                routeVeh->addRouteTransition(RoadTransition(road, dir));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        else
                        {
                            veh = createRoadVehicle(idString, nameString, vehicleString, intelString, modelFile);
                        }

                        AgentVehicle *agentVeh = dynamic_cast<AgentVehicle *>(veh);
                        if (agentVeh)
                        {
                            agentVeh->setVehicleParameters(vehPars);
                        }
                    }
                    else
                    {
                        Vehicle *veh = createRoadVehicle(idString, nameString, vehicleString, intelString, modelFile);
                        if (veh)
                        {
                            VehicleManager::Instance()->addVehicle(veh);
                        }
                    }
                    //std::cout << "RoadVehicle: " << idString << nameString << typeString << roadIdString << pos << lane << dir << vel << intelString << modelFile << std::endl;
                }
            }
        }
    }
}

//Methode erstellt Fahrzeuge innerhalb einer Kachel
void VehicleFactory::createTileVehicle(int hv_x, int hv_y, std::vector<Vector3D> active_fy)
{
    RoadSystem *system = RoadSystem::Instance();
    std::map<int, VehicleSource *>::const_iterator sourceIt;
    bool createVehicles = false;

    //Berechnung der Distanz der Kachel in der FFZ erstellt werden sollen (abhängig von der Geschwindigkeit)
    int tile_distance = TrafficSimulation::min_distance / _tile_width;

    // y-Richtung
    for (int i = -tile_distance; i <= tile_distance; i++)
    {
        // x-Richtung
        for (int j = -tile_distance; j <= tile_distance; j++)
        {
            int x = hv_x + j;
            int y = hv_y + i;
            //FFZ sollen nur innerhalb eines "Rahmens" des Quadrats mit der Länge 2 x tile_distance erstellt werden (Eigenfahrzeug ist Mittelpunkt des Quadrats)
            if ((i > -tile_distance && i < tile_distance) && (j > -tile_distance && j < tile_distance))
                createVehicles = false;
            else
                createVehicles = true;
            //Überprüfen ob auf der ausgesuchten Kachel ein Fiddleyard ist. Falls ja, sollen dort keine weiteren Fahrzeuge erstellt werden.
            if (active_fy.size() > 0)
            {
                for (int afy_i = 0; afy_i < active_fy.size(); afy_i++)
                {
                    if (x == system->get_tile(active_fy.at(afy_i)[0], active_fy.at(afy_i)[1])[0] && y == system->get_tile(active_fy.at(afy_i)[0], active_fy.at(afy_i)[1])[1])
                    {
                        createVehicles = false;
                        if (coVRMSController::instance()->isMaster())
                        {
                            std::cout << std::endl << "KACHEL AUF FIDDLEYARD" << std::endl << std::endl;
                        }
                    }
                }
            }
            if (createVehicles == true)
            {
                if (system->check_position(x, y)) // Test ob die Kachel existiert
                {
                    std::list<RoadLineSegment *> current_rls_list = system->getRLS_List(x, y);
                    std::list<RoadLineSegment *>::iterator it;
                    // RoadLineSegment-Liste durchgehen um Zugriff auf die Straßen innerhalb einer Kachel zu bekommen
                    for (it = current_rls_list.begin(); it != current_rls_list.end(); it++)
                    {
                        //Bestimmung der Verkehrsdichte
                        double traffic_density = 0;
                        if (TrafficSimulation::placeholder != -1)
                            traffic_density = TrafficSimulation::placeholder * 100;
                        else
                        {
                            std::string name = ((*it)->getRoad())->getName();
                            size_t findTD;
                            size_t findSep;
                            findTD = name.find("Placeholder");
                            findSep = name.find(";");
                            if (findSep == string::npos)
                                findSep = name.length();
                            //std::cout << "name: " << name << " findTD: " << (int)findTD << " findSep: " << (int)findSep << std::endl;
                            if (findTD != string::npos)
                            {
                                //std::cout << "Verkehrsdichte aus String: " << (name.substr(findTD+14,findSep-1)).c_str() << std::endl;
                                traffic_density = 100 * atof((name.substr(findTD + 11, findSep - 1)).c_str());
                            }
                            traffic_density *= TrafficSimulation::td_multiplier;
                        }

                        //double length = (*it)->getRoad()->getLength();
                        double targetPoolRatio = TrafficSimulation::instance()->getZeroOneRandomNumber() * (Carpool::Instance()->getOverallRatio());
                        std::string poolId = (Carpool::Instance())->getPoolIdByRatio(targetPoolRatio);
                        Pool *currentPool = Carpool::Instance()->getPoolById(poolId);

                        int leftLanes = (((*it)->getRoad())->getLaneSection((*it)->get_smax()))->getNumLanesLeft();
                        int rightLanes = (((*it)->getRoad())->getLaneSection((*it)->get_smax()))->getNumLanesRight();
                        double targetRatio = TrafficSimulation::instance()->getZeroOneRandomNumber() * currentPool->getOverallRatio();
                        std::string vehString = currentPool->getVehicleByRatio(targetRatio);
                        Vehicle *veh = NULL;
                        std::string nameString = currentPool->getId();
                        //Neu Andreas
                        double vel = VehicleFactory::generateStartVelocity(it, currentPool, 0);
                        if (currentPool->isRatioVehicleMapEmpty())
                        {
                            veh = VehicleFactory::Instance()->createRoadVehicle(nameString, nameString, (*it)->getRoad(), (*it)->get_smax(), sourceIt->second->getFiddleLane(), -1, vel);
                        }
                        else
                        {
                            //FFZ erstellen

                            ////Verkehrsdichte pro Straßenseite ////

                            int pos_lanes = 0;
                            int neg_lanes = 0;
                            int drivableLanes = 0;
                            VehicleList vehiclesOnRoad = VehicleManager::Instance()->getVehicleList((*it)->getRoad());
                            std::list<Vehicle *>::iterator v_it;
                            bool place_busy = false;
                            //Überprüfung ob sich Fahrzeuge in der Nähe der dynm. Quelle befinden
                            for (v_it = vehiclesOnRoad.begin(); v_it != vehiclesOnRoad.end(); v_it++)
                            {
                                double dist2veh = (*v_it)->getSquaredDistanceTo((*v_it)->getRoad()->getCenterLinePoint((*it)->get_smax()));
                                if (dist2veh < traffic_density)
                                {
                                    place_busy = true;
                                    break;
                                }
                                int current_lane = (*v_it)->getLane();
                                if (current_lane < 0)
                                    neg_lanes++;
                                else if (current_lane > 0)
                                    pos_lanes++;
                            }
                            //Fahrzeuge nur erstellen falls genügen Platz dafür ist
                            if (!place_busy)
                            {
                                if (TrafficSimulation::maxVehicles == 0)
                                {
                                    //Bedarf auf positiver Seite
                                    if ((neg_lanes > pos_lanes) && (leftLanes >= 1))
                                    {
                                        for (int check_lane = leftLanes; check_lane >= 0; check_lane--)
                                        {
                                            if ((((*it)->getRoad())->getLaneSection((*it)->get_smax()))->getLane(check_lane)->getLaneType() == Lane::DRIVING)
                                            {
                                                vel = VehicleFactory::generateStartVelocity(it, currentPool, check_lane);
                                                veh = VehicleFactory::Instance()->cloneRoadVehicle(vehString, nameString, (*it)->getRoad(), (*it)->get_smax(), check_lane, vel, -1);
                                                break;
                                            }
                                        }
                                    }
                                    //Bedarf auf negativer Seite
                                    else if ((neg_lanes < pos_lanes) && (rightLanes >= 1))
                                    {
                                        for (int check_lane = rightLanes; check_lane >= 0; check_lane--)
                                        {
                                            if ((((*it)->getRoad())->getLaneSection((*it)->get_smax()))->getLane(-check_lane)->getLaneType() == Lane::DRIVING)
                                            {
                                                vel = VehicleFactory::generateStartVelocity(it, currentPool, -check_lane);
                                                veh = VehicleFactory::Instance()->cloneRoadVehicle(vehString, nameString, (*it)->getRoad(), (*it)->get_smax(), -check_lane, vel, 1);
                                                break;
                                            }
                                        }
                                    }
                                    //Verkehrssituation ist ausgeglichen - Seitenwahl über Wert von "direction" der Klasse Pool
                                    else if (currentPool->getDir() < 0 && (rightLanes >= 1))
                                    {
                                        for (int check_lane = rightLanes; check_lane >= 0; check_lane--)
                                        {
                                            if ((((*it)->getRoad())->getLaneSection((*it)->get_smax()))->getLane(-check_lane)->getLaneType() == Lane::DRIVING)
                                            {
                                                vel = VehicleFactory::generateStartVelocity(it, currentPool, -check_lane);
                                                veh = VehicleFactory::Instance()->cloneRoadVehicle(vehString, nameString, (*it)->getRoad(), (*it)->get_smax(), -check_lane, vel, 1);
                                                (currentPool)->changeDir();
                                                break;
                                            }
                                        }
                                    }
                                    else if (currentPool->getDir() > 0 && (leftLanes >= 1))
                                    {
                                        for (int check_lane = leftLanes; check_lane >= 0; check_lane--)
                                        {
                                            if ((((*it)->getRoad())->getLaneSection((*it)->get_smax()))->getLane(check_lane)->getLaneType() == Lane::DRIVING)
                                            {
                                                vel = VehicleFactory::generateStartVelocity(it, currentPool, check_lane);
                                                veh = VehicleFactory::Instance()->cloneRoadVehicle(vehString, nameString, (*it)->getRoad(), (*it)->get_smax(), check_lane, vel, -1);
                                                currentPool->changeDir();
                                                break;
                                            }
                                        }
                                    }
                                    else if ((rightLanes == 0) && (leftLanes >= 1))
                                    {
                                        for (int check_lane = leftLanes; check_lane >= 0; check_lane--)
                                        {
                                            if ((((*it)->getRoad())->getLaneSection((*it)->get_smax()))->getLane(check_lane)->getLaneType() == Lane::DRIVING)
                                            {
                                                vel = VehicleFactory::generateStartVelocity(it, currentPool, check_lane);
                                                veh = VehicleFactory::Instance()->cloneRoadVehicle(vehString, nameString, (*it)->getRoad(), (*it)->get_smax(), check_lane, vel, -1);
                                                break;
                                            }
                                        }
                                    }
                                    else if ((leftLanes == 0) && (rightLanes >= 1))
                                    {
                                        for (int check_lane = rightLanes; check_lane >= 0; check_lane--)
                                        {
                                            if ((((*it)->getRoad())->getLaneSection((*it)->get_smax()))->getLane(-check_lane)->getLaneType() == Lane::DRIVING)
                                            {
                                                vel = VehicleFactory::generateStartVelocity(it, currentPool, -check_lane);
                                                veh = VehicleFactory::Instance()->cloneRoadVehicle(vehString, nameString, (*it)->getRoad(), (*it)->get_smax(), -check_lane, vel, 1);
                                                break;
                                            }
                                        }
                                    }
                                }
                                else if (((VehicleManager::Instance()->getVehicleOverallList()).size()) < TrafficSimulation::maxVehicles)
                                {
                                    //Bedarf auf positiver Seite
                                    if ((neg_lanes > pos_lanes) && (leftLanes >= 1))
                                    {
                                        for (int check_lane = leftLanes; check_lane >= 0; check_lane--)
                                        {
                                            if ((((*it)->getRoad())->getLaneSection((*it)->get_smax()))->getLane(check_lane)->getLaneType() == Lane::DRIVING)
                                            {
                                                vel = VehicleFactory::generateStartVelocity(it, currentPool, check_lane);
                                                veh = VehicleFactory::Instance()->cloneRoadVehicle(vehString, nameString, (*it)->getRoad(), (*it)->get_smax(), check_lane, vel, -1);
                                                break;
                                            }
                                        }
                                    }
                                    //Bedarf auf negativer Seite
                                    else if ((neg_lanes < pos_lanes) && (rightLanes >= 1))
                                    {
                                        for (int check_lane = rightLanes; check_lane >= 0; check_lane--)
                                        {
                                            if ((((*it)->getRoad())->getLaneSection((*it)->get_smax()))->getLane(-check_lane)->getLaneType() == Lane::DRIVING)
                                            {
                                                vel = VehicleFactory::generateStartVelocity(it, currentPool, -check_lane);
                                                veh = VehicleFactory::Instance()->cloneRoadVehicle(vehString, nameString, (*it)->getRoad(), (*it)->get_smax(), -check_lane, vel, 1);
                                                (currentPool)->changeDir();
                                                break;
                                            }
                                        }
                                    }
                                    //Verkehrssituation ist ausgeglichen - Seitenwahl über Wert von "direction" der Klasse Pool
                                    // Neu Andreas 27-11-2012: auskommentiert
                                    //else if (currentPool->getDir() < 0 && (rightLanes >= 1)){

                                    //	veh = VehicleFactory::Instance()->cloneRoadVehicle(vehString, nameString, (*it)->getRoad(), (*it)->get_smax(), currentPool->getDir(), vel, -1*currentPool->getDir());

                                    //	(currentPool)->changeDir(); // sorgt dafür, dass beim jeden Aufruf die anfängliche Fahrtrichtung geändert wird
                                    //}
                                    else if (currentPool->getDir() < 0 && (rightLanes >= 1))
                                    {
                                        for (int check_lane = leftLanes; check_lane >= 0; check_lane--)
                                        {
                                            if ((((*it)->getRoad())->getLaneSection((*it)->get_smax()))->getLane(check_lane)->getLaneType() == Lane::DRIVING)
                                            {
                                                vel = VehicleFactory::generateStartVelocity(it, currentPool, check_lane);
                                                veh = VehicleFactory::Instance()->cloneRoadVehicle(vehString, nameString, (*it)->getRoad(), (*it)->get_smax(), check_lane, vel, -1 * currentPool->getDir());

                                                // sorgt dafür, dass beim jeden Aufruf die anfängliche Fahrtrichtung geändert wird
                                                (currentPool)->changeDir();
                                                break;
                                            }
                                        }
                                    }
                                    else if (currentPool->getDir() > 0 && (leftLanes >= 1))
                                    {
                                        for (int check_lane = leftLanes; check_lane >= 0; check_lane--)
                                        {
                                            if ((((*it)->getRoad())->getLaneSection((*it)->get_smax()))->getLane(check_lane)->getLaneType() == Lane::DRIVING)
                                            {
                                                vel = VehicleFactory::generateStartVelocity(it, currentPool, check_lane);
                                                veh = VehicleFactory::Instance()->cloneRoadVehicle(vehString, nameString, (*it)->getRoad(), (*it)->get_smax(), check_lane, vel, currentPool->getDir());

                                                currentPool->changeDir();
                                                break;
                                            }
                                        }
                                    }
                                    else if ((rightLanes == 0) && (leftLanes >= 1))
                                    {
                                        for (int check_lane = leftLanes; check_lane >= 0; check_lane--)
                                        {
                                            if ((((*it)->getRoad())->getLaneSection((*it)->get_smax()))->getLane(check_lane)->getLaneType() == Lane::DRIVING)
                                            {
                                                vel = VehicleFactory::generateStartVelocity(it, currentPool, check_lane);
                                                veh = VehicleFactory::Instance()->cloneRoadVehicle(vehString, nameString, (*it)->getRoad(), (*it)->get_smax(), check_lane, vel, -1);
                                                break;
                                            }
                                        }
                                        veh = VehicleFactory::Instance()->cloneRoadVehicle(vehString, nameString, (*it)->getRoad(), (*it)->get_smax(), 1, vel, -1);
                                    }
                                    else if ((leftLanes == 0) && (rightLanes >= 1))
                                    {
                                        for (int check_lane = leftLanes; check_lane >= 0; check_lane--)
                                        {
                                            if ((((*it)->getRoad())->getLaneSection((*it)->get_smax()))->getLane(check_lane)->getLaneType() == Lane::DRIVING)
                                            {
                                                vel = VehicleFactory::generateStartVelocity(it, currentPool, check_lane);
                                                veh = VehicleFactory::Instance()->cloneRoadVehicle(vehString, nameString, (*it)->getRoad(), (*it)->get_smax(), check_lane, vel, -1);
                                                //currentPool->changeDir();
                                                break;
                                            }
                                        }
                                    }
                                }
                            }
                            if (veh)
                            {
                                TrafficSimulation::instance()->getVehicleManager()->addVehicle(veh);
                            }
                        }
                    }
                }
            }
        }
    }
}

//Neu Andreas:
double VehicleFactory::generateStartVelocity(std::list<RoadLineSegment *>::iterator it, Pool *currentPool, int lane)
{

    // Velocity Deviance added in Percent of speedLimit
    double vel_startDev = currentPool->getStartVelocityDeviation() / 100;
    double randomNum = TrafficSimulation::instance()->getZeroOneRandomNumber();
    // Maximum Speed
    double vel_max = currentPool->getStartVelocity() - 2 * vel_startDev * currentPool->getStartVelocity() * randomNum;
    double vel_speedLim = 0;
    double vel = 0;
    // Speedlimit
    if ((((*it)->getRoad())->getLaneSection((*it)->get_smax()))->getLane(lane)->getLaneType() == Lane::DRIVING)
    {
        vel_speedLim = vel_speedLim = ((*it)->getRoad())->getSpeedLimit((*it)->get_smax(), lane);
        //std::cout<<"Sollgeschwindigkeit"<<vel_speedLim*3.6<<"km/h"<<std::endl;
        double vel_speedDev = 0;
        if (currentPool->getName() == "sportscars")
        {
            vel_speedDev = abs(vel_startDev * vel_speedLim * 2 * (randomNum - 0.5));
        }
        else
        {
            vel_speedDev = vel_startDev * vel_speedLim * 2 * (randomNum - 0.5);
        }
        vel_speedLim = vel_speedLim + vel_speedDev;
        vel = std::min(vel_speedLim, vel_max);
        return vel;
    }
    else
    {
        return 0;
    }
}
