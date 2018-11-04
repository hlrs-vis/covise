/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "VehicleManager.h"

#include "TrafficSimulation.h"
#include <VehicleUtil/RoadSystem/RoadSystem.h>
#include "VehicleFactory.h"
#include "HumanVehicle.h"
#include "AgentVehicle.h"
#include <cover/coVRPluginSupport.h>
#include <osg/Matrix>
#include <osg/MatrixTransform>
#include <algorithm>
#include "PorscheFFZ.h"

VehicleManager *VehicleManager::__instance = NULL;

VehicleManager *VehicleManager::Instance()
{
    if (__instance == NULL)
    {
        __instance = new VehicleManager();
    }
    return __instance;
}

void VehicleManager::Destroy()
{
    delete __instance;
    __instance = NULL;
}

VehicleManager::VehicleManager()
    : cameraVehicleIt(vehicleOverallList.begin())
    , humanVehicle(NULL)
{
    system = RoadSystem::Instance();
}

void VehicleManager::setCameraVehicle(int vehNum)
{
    cameraVehicleIt = vehicleOverallList.begin();
    advance(cameraVehicleIt, vehNum);
}

void VehicleManager::addVehicle(Vehicle *veh)
{
    vehicleOverallList.push_back(veh);
    if (veh->getDu() < 0)
        roadVehicleListMap[veh->getRoad()].push_back(veh);
    else
        roadVehicleListMap[veh->getRoad()].push_front(veh);
    roadVehicleListMap[veh->getRoad()].sort(Vehicle::compare);

    vehicleDecisionDeque.push_back(veh);
}

void VehicleManager::removeVehicle(VehicleList::iterator vehIt, Road *road)
{
    VehicleDeque::iterator vehDecIt = find(vehicleDecisionDeque.begin(), vehicleDecisionDeque.end(), *vehIt);
    vehicleDecisionDeque.erase(vehDecIt);

    if (vehIt == cameraVehicleIt)
    {
        cameraVehicleIt = vehicleOverallList.end();
    }
    roadVehicleListMap[road].erase(vehIt);
    vehicleOverallList.erase(vehIt);

    //delete (*vehIt);
    VehicleFactory::Instance()->deleteRoadVehicle(*vehIt);
}

void VehicleManager::removeVehicle(Vehicle *veh, Road *road)
{
    VehicleDeque::iterator vehDecIt = find(vehicleDecisionDeque.begin(), vehicleDecisionDeque.end(), veh);
    vehicleDecisionDeque.erase(vehDecIt);

    if ((cameraVehicleIt != vehicleOverallList.end()) && (veh == (*cameraVehicleIt)))
    {
        cameraVehicleIt = vehicleOverallList.end();
    }
    roadVehicleListMap[road].remove(veh);
    vehicleOverallList.remove(veh);
    //delete veh;
    VehicleFactory::Instance()->deleteRoadVehicle(veh);
}

void
VehicleManager::
    removeAllAgents(double maxVel)
{
    // delete all vehicles slower than maxVel
    int count = 0;
    VehicleList::iterator vehIt = vehicleOverallList.begin();
    while (vehIt != vehicleOverallList.end())
    {
        AgentVehicle *veh = dynamic_cast<AgentVehicle *>(*vehIt);
        if (veh && fabs(veh->getDu()) <= maxVel)
        {
            ++vehIt;
            removeVehicle(veh, veh->getRoad());
            count++;
        }
        else
        {
            vehIt++; // go to next one in list
        }
    }
    std::cout << "Deleted " << count << " agent vehicles (slower than " << maxVel * 3.6 << " km/h)." << std::endl;
}

void VehicleManager::changeRoad(VehicleList::iterator vehIt, Road *from, Road *to, int dir)
{
    Vehicle *veh = (*vehIt);
    //Road* from = (*vehIt)->getRoad();
    roadVehicleListMap[from].erase(vehIt);
    //(*vehIt)->setRoad(to);
    if (dir < 0)
        insertVehicleAtBack(veh, to);
    else
        insertVehicleAtFront(veh, to);
}

void VehicleManager::changeRoad(Vehicle *veh, Road *from, Road *to, int dir)
{
    //Road* from = veh->getRoad();
    VehicleList::iterator vehIt = std::find(roadVehicleListMap[from].begin(), roadVehicleListMap[from].end(), veh);
    if (vehIt != roadVehicleListMap[from].end())
    {
        changeRoad(vehIt, from, to, dir);
    }
}

void VehicleManager::moveVehicle(VehicleList::iterator vehIt, int dir)
{
    if (dir < 0 && vehIt != roadVehicleListMap[(*vehIt)->getRoad()].begin())
    {
        moveVehicleBackward(vehIt);
    }
    else if (dir > 0 && vehIt != roadVehicleListMap[(*vehIt)->getRoad()].end())
    {
        moveVehicleForward(vehIt);
    }
}

void VehicleManager::moveVehicle(Vehicle *veh, int dir)
{
    Road *road = veh->getRoad();
    VehicleList::iterator vehIt = std::find(roadVehicleListMap[road].begin(), roadVehicleListMap[road].end(), veh);
    if (vehIt != roadVehicleListMap[road].end())
    {
        moveVehicle(vehIt, dir);
    }
    //showVehicleList(road);
}

Vehicle *VehicleManager::getNextVehicle(VehicleList::iterator vehIt, int dir)
{
    dir = (dir >= 0) ? 1 : -1;
    Road *road = (*vehIt)->getRoad();

    if (dir > 0 && (++vehIt) != roadVehicleListMap[road].end())
    {
        return (*vehIt);
    }
    else if (dir < 0 && vehIt != roadVehicleListMap[road].begin())
    {
        return (*(--vehIt));
    }
    else
    {
        return NULL;
    }
}

Vehicle *VehicleManager::getNextVehicle(Vehicle *veh, int dir)
{
    Road *road = veh->getRoad();
    VehicleList::iterator vehIt = std::find(roadVehicleListMap[road].begin(), roadVehicleListMap[road].end(), veh);
    if (vehIt != roadVehicleListMap[road].end())
    {
        return getNextVehicle(vehIt, dir);
    }
    else
    {
        std::cout << "Vehicle Manager: Vehicle " << veh->getName() << " not found in list of road " << road->getId() << " -> Alarm, Inconsistency!!!" << std::endl;
        return NULL;
    }
}

Vehicle *VehicleManager::getNextVehicle(VehicleList::iterator vehIt, int dir, int lane)
{
    dir = (dir >= 0) ? 1 : -1;
    Road *road = (*vehIt)->getRoad();

    if (dir > 0 && vehIt != roadVehicleListMap[road].end())
    {
        Vehicle *nextVeh = NULL;
        while ((++vehIt) != roadVehicleListMap[road].end())
        {
            //if((*vehIt)->getLane() == lane) {
            if ((*vehIt)->isOnLane(lane))
            {
                nextVeh = (*vehIt);
                break;
            }
        }
        return nextVeh;
    }
    else if (dir < 0 && (vehIt) != roadVehicleListMap[road].begin())
    {
        Vehicle *nextVeh = NULL;
        do
        {
            --vehIt;
            //if((*vehIt)->getLane() == lane) {
            if ((*vehIt)->isOnLane(lane))
            {
                nextVeh = (*vehIt);
                break;
            }
        } while ((vehIt) != roadVehicleListMap[road].begin());
        return nextVeh;
    }
    else
    {
        return NULL;
    }
}

Vehicle *VehicleManager::getNextVehicle(Vehicle *veh, int dir, int lane)
{
    Road *road = veh->getRoad();
    VehicleList::iterator vehIt = std::find(roadVehicleListMap[road].begin(), roadVehicleListMap[road].end(), veh);
    if (vehIt != roadVehicleListMap[road].end())
    {
        return getNextVehicle(vehIt, dir, lane);
    }
    else
    {
        std::cout << "Vehicle Manager: Vehicle " << veh->getName() << " not found in list of road " << road->getId() << " -> Alarm, Inconsistency!!!" << std::endl;
        return NULL;
    }
}

Vehicle *VehicleManager::getFirstVehicle(Road *road)
{
    VehicleList::iterator vehIt = roadVehicleListMap[road].begin();
    if (vehIt != roadVehicleListMap[road].end())
    {
        return (*vehIt);
    }
    else
    {
        return NULL;
    }
}

Vehicle *VehicleManager::getLastVehicle(Road *road)
{
    VehicleList::iterator vehIt = roadVehicleListMap[road].end();
    if (vehIt != roadVehicleListMap[road].begin())
    {
        return (*(--vehIt));
    }
    else
    {
        return NULL;
    }
}

Vehicle *VehicleManager::getFirstVehicle(Road *road, int lane)
{
    VehicleList::iterator vehIt = roadVehicleListMap[road].begin();
    Vehicle *nextVeh = NULL;
    while (vehIt != roadVehicleListMap[road].end())
    {
        //if((*vehIt)->getLane()==lane) {
        if ((*vehIt)->isOnLane(lane))
        {
            nextVeh = (*vehIt);
            break;
        }
        ++vehIt;
    }
    return nextVeh;
}

Vehicle *VehicleManager::getLastVehicle(Road *road, int lane)
{
    VehicleList::iterator vehIt = roadVehicleListMap[road].end();
    Vehicle *nextVeh = NULL;
    if (vehIt != roadVehicleListMap[road].begin())
    {
        do
        {
            --vehIt;
            //if((*vehIt)->getLane()==lane) {
            if ((*vehIt)->isOnLane(lane))
            {
                nextVeh = (*vehIt);
                break;
            }
        } while (vehIt != roadVehicleListMap[road].begin());
    }
    return nextVeh;
}

std::map<double, Vehicle *> VehicleManager::getSurroundingVehicles(Vehicle *veh)
{
    Road *road = veh->getRoad();
    VehicleList::iterator vehIt = std::find(roadVehicleListMap[road].begin(), roadVehicleListMap[road].end(), veh);
    if (vehIt != roadVehicleListMap[road].end())
    {
        return getSurroundingVehicles(vehIt);
    }
    else
    {
        std::cerr << "Vehicle Manager: Vehicle " << veh->getName() << " not found in list of road " << road->getId() << " -> Alarm, Inconsistency!!!" << std::endl;
        return std::map<double, Vehicle *>();
    }
}

std::map<double, Vehicle *> VehicleManager::getSurroundingVehicles(VehicleList::iterator vehIt)
{
    std::map<double, Vehicle *> vehMap;

    Road *road = (*vehIt)->getRoad();
    if (!road)
    {
        return vehMap;
    }

    //std::set<RoadTransition> transSet = road->getConnectingRoadTransitionSet((*vehIt)->getRoadTransition());

    /*std::cout << "Road " << road->getId() << ", direction: " << (*vehIt)->getRoadTransition().direction << " connecting to: " << std::endl;
   for(std::set<RoadTransition>::iterator transSetIt = transSet.begin(); transSetIt != transSet.end(); ++transSetIt) {
      std::cout << "\tRoad: " << transSetIt->road->getId() << ", direction: " << transSetIt->direction << std::endl;
   }*/

    std::vector<Road *> roadVector;
    roadVector.push_back(road);

    RoadTransition trans = (*vehIt)->getRoadTransition();
    std::set<RoadTransition> transSet = road->getConnectingRoadTransitionSet(trans);
    for (std::set<RoadTransition>::iterator transSetIt = transSet.begin(); transSetIt != transSet.end(); ++transSetIt)
    {
        roadVector.push_back(transSetIt->road);
    }

    trans.direction *= -1;
    transSet = road->getConnectingRoadTransitionSet(trans);
    for (std::set<RoadTransition>::iterator transSetIt = transSet.begin(); transSetIt != transSet.end(); ++transSetIt)
    {
        roadVector.push_back(transSetIt->road);
    }

    for (int i = 0; i < roadVector.size(); ++i)
    {
        std::map<Road *, VehicleList>::iterator mapIt = roadVehicleListMap.find(roadVector[i]);
        for (VehicleList::iterator listIt = mapIt->second.begin(); listIt != mapIt->second.end(); ++listIt)
        {
            if ((*listIt) == (*vehIt))
            {
                continue;
            }
            double dist = ((*listIt)->getVehicleTransform().v() - (*vehIt)->getVehicleTransform().v()).length();
            vehMap.insert(std::pair<double, Vehicle *>(dist, (*listIt)));
        }
    }

    return vehMap;
}

void VehicleManager::sortVehicleList(Road *road)
{
    std::map<Road *, VehicleList>::iterator mapIt = roadVehicleListMap.find(road);
    if (mapIt != roadVehicleListMap.end())
    {
        mapIt->second.sort(Vehicle::compare);
    }
}

const VehicleList &VehicleManager::getVehicleList(Road *road)
{
    return roadVehicleListMap[road];
}

void VehicleManager::insertVehicleAtFront(Vehicle *veh, Road *road)
{
    VehicleList::iterator vehIt = roadVehicleListMap[road].begin();
    for (; vehIt != roadVehicleListMap[road].end(); ++vehIt)
    {
        if (veh->getU() < (*vehIt)->getU())
        {
            break;
        }
    }
    roadVehicleListMap[road].insert(vehIt, veh);
}

void VehicleManager::insertVehicleAtBack(Vehicle *veh, Road *road)
{
    VehicleList::iterator vehIt = roadVehicleListMap[road].end();

    if (vehIt != roadVehicleListMap[road].end())
    {
        for (--vehIt; vehIt != roadVehicleListMap[road].begin(); --vehIt)
        {
            if (veh->getU() > (*vehIt)->getU())
            {
                ++vehIt;
                break;
            }
        }
    }

    roadVehicleListMap[road].insert(vehIt, veh);
}

void VehicleManager::moveVehicleForward(VehicleList::iterator vehIt)
{
    Road *road = (*vehIt)->getRoad();
    Vehicle *veh = (*vehIt);

    vehIt = roadVehicleListMap[road].erase(vehIt);

    for (; vehIt != roadVehicleListMap[road].end(); ++vehIt)
    {
        if (veh->getU() < (*vehIt)->getU())
        {
            break;
        }
    }

    roadVehicleListMap[road].insert(vehIt, veh);
}

void VehicleManager::moveVehicleBackward(VehicleList::iterator vehIt)
{
    Road *road = (*vehIt)->getRoad();
    Vehicle *veh = (*vehIt);

    vehIt = roadVehicleListMap[road].erase(vehIt);

    do
    {
        if (veh->getU() > (*(--vehIt))->getU())
        {
            ++vehIt;
            break;
        }
    } while (vehIt != roadVehicleListMap[road].begin());

    roadVehicleListMap[road].insert(vehIt, veh);
}

void VehicleManager::showVehicleList(Road *road)
{
    std::cout << "Vehicle list of road " << road->getId() << ":";
    for (VehicleList::iterator vehIt = roadVehicleListMap[road].begin(); vehIt != roadVehicleListMap[road].end(); ++vehIt)
    {
        std::cout << " \t" << (*vehIt)->getName() << " (" << (*vehIt) << ")";
    }
    std::cout << std::endl;
}

void VehicleManager::switchToNextCamera()
{
    if (cameraVehicleIt == vehicleOverallList.end())
    {
        cameraVehicleIt = vehicleOverallList.begin();
    }
    else
    {
        ++cameraVehicleIt;
    }

    if (cameraVehicleIt != vehicleOverallList.end() && dynamic_cast<HumanVehicle *>(*cameraVehicleIt))
    {
        ++cameraVehicleIt;
    }
}

void VehicleManager::switchToPreviousCamera()
{
    if (cameraVehicleIt == vehicleOverallList.begin())
    {
        cameraVehicleIt = vehicleOverallList.end();
    }
    else
    {
        --cameraVehicleIt;
    }

    if (cameraVehicleIt != vehicleOverallList.end() && dynamic_cast<HumanVehicle *>(*cameraVehicleIt))
    {
        if (cameraVehicleIt == vehicleOverallList.begin())
        {
            cameraVehicleIt = vehicleOverallList.end();
        }
        else
        {
            --cameraVehicleIt;
        }
    }
}

void VehicleManager::unbindCamera()
{
    cameraVehicleIt = vehicleOverallList.end();
}

/*void VehicleManager::brakeCameraVehicle()
{
   FollowTheLeaderVehicle* veh = dynamic_cast<FollowTheLeaderVehicle*>(cameraVehicle);
   if(veh) {
      veh->brake();
   }
}*/

bool VehicleManager::isJunctionEmpty(Junction *junction)
{
    if (!junction)
    {
        return true;
    }
    //std::cout << "isJunctionEmpty(): " << std::endl;
    std::map<Road *, PathConnectionSet> connSetMap = junction->getPathConnectionSetMap();
    for (std::map<Road *, PathConnectionSet>::iterator connSetMapIt = connSetMap.begin(); connSetMapIt != connSetMap.end(); ++connSetMapIt)
    {
        //std::cout << "\t looking at incoming road " << connSetMapIt->first->getId() << std::endl;
        PathConnectionSet connSet = connSetMapIt->second;
        for (PathConnectionSet::iterator connSetIt = connSet.begin(); connSetIt != connSet.end(); ++connSetIt)
        {
            //std::cout << "\t\t looking at connecting path " << (*connSetIt)->getConnectingPath()->getId() << std::endl;
            std::map<Road *, VehicleList>::iterator mapIt = roadVehicleListMap.find((*connSetIt)->getConnectingPath());
            if (mapIt != roadVehicleListMap.end())
            {
                //std::cout << "\t\t\tsize of road vehicle list: " << mapIt->second.size() << std::endl;
                if (!(mapIt->second.empty()))
                {
                    return false;
                }
            }
            //else {
            //   std::cout << "\t\t\tno road vehicle list for path connection " << (*connSetIt)->getId() << std::endl;
            //}
        }
    }

    return true;
}

void VehicleManager::moveAllVehicles(double dt)
{
    for (VehicleList::iterator vehIt = vehicleOverallList.begin(); vehIt != vehicleOverallList.end(); ++vehIt)
    {
        Vehicle *veh = (*vehIt);
        //std::cout << "Moving: " << (*vehIt)->getName() << std::endl;
        double lastVehU = veh->getU();

        veh->move(dt);

        double vehU = veh->getU();

        //Resorting
        //std::cout << ">> moveAllVehicles 1" << std::endl;
        std::map<Road *, VehicleList>::iterator listMapIt = roadVehicleListMap.find(veh->getRoad());
        //std::cout << ">> moveAllVehicles 2" << std::endl;
        if (listMapIt == roadVehicleListMap.end())
        {
            //std::cout << ">> moveAllVehicles 3" << std::endl;
            roadVehicleListMap[veh->getRoad()].push_back(veh);
            //std::cout << ">> moveAllVehicles 4" << std::endl;
        }
        else
        {
            //std::cout << ">> moveAllVehicles 5" << std::endl;
            VehicleList::iterator vehListIt = std::find(listMapIt->second.begin(), listMapIt->second.end(), veh);
            //std::cout << ">> moveAllVehicles 6" << std::endl;
            VehicleList::iterator nextVehListIt = vehListIt;
            //std::cout << ">> moveAllVehicles 7" << std::endl;
            //Forward search
            bool movedForward = false;
            ++nextVehListIt;
            //std::cout << ">> moveAllVehicles 8" << std::endl;
            while ((nextVehListIt != listMapIt->second.end()) && !(veh->getU() < (*nextVehListIt)->getU()))
            {
                ++nextVehListIt;
                movedForward = true;
                //std::cout << ">> moveAllVehicles 9" << std::endl;
            }
            if (movedForward)
            {
                listMapIt->second.erase(vehListIt);
                listMapIt->second.insert(nextVehListIt, veh);
                //std::cout << ">> moveAllVehicles 10" << std::endl;
            }
            else
            { //Backward search
                //std::cout << ">> moveAllVehicles 11" << std::endl;
                listMapIt->second.reverse();
                bool movedBackward = false;
                nextVehListIt = vehListIt;
                ++nextVehListIt;
                //std::cout << ">> moveAllVehicles 12" << std::endl;
                while ((nextVehListIt != listMapIt->second.end()) && !(veh->getU() > (*nextVehListIt)->getU()))
                {
                    ++nextVehListIt;
                    movedBackward = true;
                    //std::cout << ">> moveAllVehicles 13" << std::endl;
                }
                if (movedBackward)
                {
                    //std::cout << ">> moveAllVehicles 14" << std::endl;
                    listMapIt->second.erase(vehListIt);
                    listMapIt->second.insert(nextVehListIt, veh);
                    //std::cout << ">> moveAllVehicles 15" << std::endl;
                }
                listMapIt->second.reverse();
                //std::cout << ">> moveAllVehicles 16" << std::endl;
            }
        }

        //Trigger road sensors
        Road *vehRoad = veh->getRoadTransition().road;
        if (vehRoad)
        {
            std::map<double, RoadSensor *>::iterator roadSensorEntry = vehRoad->getRoadSensorMapEntry(vehU);

            if (roadSensorEntry != vehRoad->getRoadSensorMapEnd() && ((lastVehU < roadSensorEntry->first && vehU > roadSensorEntry->first) || (lastVehU > roadSensorEntry->first && vehU < roadSensorEntry->first)))
            {
                std::cout << "Vehicle from " << lastVehU << " to " << vehU << ": Vehicle " << veh->getName() << " triggering sensor " << roadSensorEntry->second->getId() << " at " << roadSensorEntry->first << ", (s=" << roadSensorEntry->second->getS() << ")" << std::endl;
                roadSensorEntry->second->trigger(veh->getName());
            }
        }
    }

    //Round Robin for vehicle decision making
    double decInt = 1.0; //Interval every vehicle can make a decision
    double frameDur = 1.0 / 60.0; //Standard frame duration
    int numVehDec = (int)(ceil(vehicleDecisionDeque.size() / decInt * frameDur));
    for (int decIt = 0; decIt < numVehDec; ++decIt)
    {
        Vehicle *veh = vehicleDecisionDeque.front();
        vehicleDecisionDeque.pop_front();
        veh->makeDecision();
        vehicleDecisionDeque.push_back(veh);
    }
    /*static double numVehDisplayTime = 0.0;
   if(numVehDisplayTime > 1.0) {
     // std::cout << "VehicleManager::moveAllVehicles(): Number of vehicles: " << vehicleOverallList.size() << ", make decision: " << numVehDec << std::endl;
      numVehDisplayTime -=1.0;
   }
   numVehDisplayTime += dt;*/

    /*
   std::cout << "Vehicle list on road " << roadVehicleListMap.begin()->first->getName() << ": ";
   for(VehicleList::iterator vehIt = roadVehicleListMap.begin()->second.begin(); vehIt != roadVehicleListMap.begin()->second.end(); ++vehIt) {
      std::cout << (*vehIt)->getName() << " ";
   }
   std::cout << std::endl;
   */
    //for(int roadIt = 0; roadIt < RoadSystem::Instance()->getNumRoads(); ++roadIt) {
    //   showVehicleList(RoadSystem::Instance()->getRoad(roadIt));
    //}

    //if(vehicleOverallList.begin()!=vehicleOverallList.end()) {
    //   cameraVehicle = *(vehicleOverallList.begin());
    //}
    if (cameraVehicleIt != vehicleOverallList.end())
    {
        osg::Matrix cameraTransform;
        cameraTransform.makeRotate(M_PI_2, osg::Vec3(0, 0, -1));
        cameraTransform *= osg::Matrix::translate(1.5, 0.5, 1.3);
        osg::Matrix invViewSpace = osg::Matrix::inverse(cameraTransform * (*cameraVehicleIt)->getVehicleGeometry()->getVehicleTransformMatrix() * cover->getObjectsScale()->getMatrix());
        //osg::Matrix invViewSpace = osg::Matrix::inverse(cameraVehicle->getVehicleTransformMatrix());
        osg::Matrix objSpace = invViewSpace * cover->getObjectsScale()->getMatrix();
        //osg::Matrix objSpace = invViewSpace;
        cover->setXformMat(objSpace);

        //std::cerr << "Camera vehicle: " << cameraVehicle->getName() << ", road: " << cameraVehicle->getRoad()->getId() << ", u: " << cameraVehicle->getU() << ", vel: " << cameraVehicle->getDu() << std::endl;
    }
}

// Neuerung: updateFiddleyards bekommt nun noch die aktuelle Position des Eigenfahrzeugs, sowie dessen Geschwindigkeit übergeben um darüber zu entscheiden, ob ein Fiddleyard de-/aktiviert ist.
void VehicleManager::updateFiddleyards(double dt, osg::Vec2d incoming_pos)
{
    if (system)
    {
        if ((Carpool::Instance()->getPoolVector()).size() > 0 && TrafficSimulation::useCarpool == 1)
        {
            acitve_fiddleyards.clear();
            for (int i = 0; i < system->getNumFiddleyards(); ++i)
            {
                FiddleRoad *fiddleroad = system->getFiddleyard(i)->getFiddleroad();

                int direction = system->getFiddleyard(i)->getTarmacConnection()->getConnectingTarmacDirection();
                Vector3D fiddleyard_pos(-1, -1, -1);
                //direction zeigt an, an welcher Stelle der Straße das Fiddleyard sein soll: -1 = Ende; 1 = Anfang (s=0.0)
                if (direction == -1)
                    fiddleyard_pos = RoadSystem::Instance()->getRoad(system->getFiddleyard(i)->getTarmacConnection()->getConnectingTarmac()->getId())->getCenterLinePoint(RoadSystem::Instance()->getRoad(system->getFiddleyard(i)->getTarmacConnection()->getConnectingTarmac()->getId())->getLength());
                else
                    fiddleyard_pos = RoadSystem::Instance()->getRoad(system->getFiddleyard(i)->getTarmacConnection()->getConnectingTarmac()->getId())->getCenterLinePoint(0.0);
                double distance = sqrt((incoming_pos[0] - fiddleyard_pos[0]) * (incoming_pos[0] - fiddleyard_pos[0]) + (incoming_pos[1] - fiddleyard_pos[1]) * ((incoming_pos[1] - fiddleyard_pos[1])));

                //double min_distance = velocity*4;
                //if (min_distance < 180)min_distance = 180; //Ein Mindestabstand soll auch bei geringen Geschwindigkeiten gewährleistet sein
                //double max_distance = velocity*8;
                //if (max_distance < 250) max_distance = 230;
                //std::cout << "DISTANCE TO FIDDLEYARD: " << distance << " >> min_distance: " << min_distance << ", max_distance: " << max_distance <<std::endl;

                //Senken sind immer aktiv
                const std::map<int, VehicleSink *> &sinkMap = system->getFiddleyard(i)->getVehicleSinkMap();
                for (std::map<int, VehicleSink *>::const_iterator sinkIt = sinkMap.begin(); sinkIt != sinkMap.end(); ++sinkIt)
                {
                    Vehicle *veh = TrafficSimulation::instance()->getVehicleManager()->getLastVehicle(fiddleroad, sinkIt->second->getFiddleLane());
                    while (veh != NULL)
                    {
                        TrafficSimulation::instance()->getVehicleManager()->removeVehicle(veh, fiddleroad);
                        veh = TrafficSimulation::instance()->getVehicleManager()->getLastVehicle(fiddleroad, sinkIt->second->getFiddleLane());
                    }
                    //sinkIt->second->update(timer);
                }
                //Prüfen ob Fiddleyard im zulässigen Bereich ist
                if (distance > TrafficSimulation::min_distance && distance < TrafficSimulation::delete_at)
                {
                    double time = system->getFiddleyard(i)->incrementTimer(dt);
                    //std::cout << "FIDDLEYARD ACTIVE " << std::endl;
                    acitve_fiddleyards.push_back(fiddleyard_pos); //Speichern der Fiddleyard-Position, damit in dieser Kachel keine FFZ über bewegliche Fiddleyards erzeugt werden

                    const std::map<int, VehicleSource *> &sourceMap = system->getFiddleyard(i)->getVehicleSourceMap();
                    //std::cout << "::::::: SourceMap.size: " << sourceMap.size() << std::endl;
                    for (std::map<int, VehicleSource *>::const_iterator sourceIt = sourceMap.begin(); sourceIt != sourceMap.end(); sourceIt++)
                    {
                        if (sourceIt->second->spawnNextVehicle(time))
                        {
                            std::ostringstream timeStringStream;
                            timeStringStream << time;
                            //std::cout << " ----------------- TimeString: " << timeStringStream.str() << " -- dt: " << dt << std::endl;
                            std::string nameString = sourceIt->second->getId() + std::string("_") + timeStringStream.str();
                            double vel = 0.0;

                            vel = sourceIt->second->getStartVelocity() - sourceIt->second->getStartVelocityDeviation()
                                  + 2 * sourceIt->second->getStartVelocityDeviation() * TrafficSimulation::instance()->getZeroOneRandomNumber();

                            Vehicle *veh = NULL;
                            if (sourceIt->second->isRatioVehicleMapEmpty())
                            {
                                veh = VehicleFactory::Instance()->createRoadVehicle(nameString, nameString, fiddleroad, 0.0, sourceIt->second->getFiddleLane(), -1, vel);
                            }
                            else
                            {
                                double targetRatio = TrafficSimulation::instance()->getZeroOneRandomNumber() * sourceIt->second->getOverallRatio();
                                std::string vehString = sourceIt->second->getVehicleByRatio(targetRatio);

                                veh = VehicleFactory::Instance()->cloneRoadVehicle(vehString, nameString, fiddleroad, 0.0, sourceIt->second->getFiddleLane(), vel, -1);
                                //std::cout << "~~~~~~~~ cloneRoadVehicle() Road-ID: " <<(RoadSystem::Instance()->getRoad(2))->getId() << " Veh-Name: " <<  veh->getName() << " Veh-Lane: " <<  veh->getLane() << std::endl;
                                //std::cout << " >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> FFZ ueber Fiddleyard erstellt <<<<<<<<<<<< " << std::endl;
                            }
                            if (veh)
                            {
                                TrafficSimulation::instance()->getVehicleManager()->addVehicle(veh);
                            }
                        }
                        //sourceIt->second->update(timer);
                    }
                    //system->getFiddleyard(i)->update(dt);
                }
            }
        }
        else
        {
            for (int i = 0; i < system->getNumFiddleyards(); ++i)
            {
                FiddleRoad *fiddleroad = system->getFiddleyard(i)->getFiddleroad();

                int direction = system->getFiddleyard(i)->getTarmacConnection()->getConnectingTarmacDirection();
                //Senken sind immer aktiv
                const std::map<int, VehicleSink *> &sinkMap = system->getFiddleyard(i)->getVehicleSinkMap();
                for (std::map<int, VehicleSink *>::const_iterator sinkIt = sinkMap.begin(); sinkIt != sinkMap.end(); ++sinkIt)
                {
                    Vehicle *veh = TrafficSimulation::instance()->getVehicleManager()->getLastVehicle(fiddleroad, sinkIt->second->getFiddleLane());
                    while (veh != NULL)
                    {
                        TrafficSimulation::instance()->getVehicleManager()->removeVehicle(veh, fiddleroad);
                        veh = TrafficSimulation::instance()->getVehicleManager()->getLastVehicle(fiddleroad, sinkIt->second->getFiddleLane());
                    }
                    //sinkIt->second->update(timer);
                }
                double time = system->getFiddleyard(i)->incrementTimer(dt);

                const std::map<int, VehicleSource *> &sourceMap = system->getFiddleyard(i)->getVehicleSourceMap();
                //std::cout << "::::::: SourceMap.size: " << sourceMap.size() << std::endl;
                for (std::map<int, VehicleSource *>::const_iterator sourceIt = sourceMap.begin(); sourceIt != sourceMap.end(); sourceIt++)
                {
                    if (sourceIt->second->spawnNextVehicle(time))
                    {
                        std::ostringstream timeStringStream;
                        timeStringStream << time;
                        std::string nameString = sourceIt->second->getId() + std::string("_") + timeStringStream.str();
                        double vel = 0.0;

                        vel = sourceIt->second->getStartVelocity() - sourceIt->second->getStartVelocityDeviation()
                              + 2 * sourceIt->second->getStartVelocityDeviation() * TrafficSimulation::instance()->getZeroOneRandomNumber();

                        Vehicle *veh = NULL;
                        if (sourceIt->second->isRatioVehicleMapEmpty())
                        {
                            veh = VehicleFactory::Instance()->createRoadVehicle(nameString, nameString, fiddleroad, 0.0, sourceIt->second->getFiddleLane(), -1, vel);
                        }
                        else
                        {
                            double targetRatio = TrafficSimulation::instance()->getZeroOneRandomNumber() * sourceIt->second->getOverallRatio();
                            std::string vehString = sourceIt->second->getVehicleByRatio(targetRatio);

                            veh = VehicleFactory::Instance()->cloneRoadVehicle(vehString, nameString, fiddleroad, 0.0, sourceIt->second->getFiddleLane(), vel, -1);
                            //std::cout << "~~~~~~~~ cloneRoadVehicle() Road-ID: " <<(RoadSystem::Instance()->getRoad(2))->getId() << " Veh-Name: " <<  veh->getName() << " Veh-Lane: " <<  veh->getLane() << std::endl;
                            //std::cout << " >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> FFZ ueber Fiddleyard erstellt - keine Pools <<<<<<<<<<<< " << std::endl;
                        }
                        if (veh)
                        {
                            TrafficSimulation::instance()->getVehicleManager()->addVehicle(veh);
                        }
                    }
                    //sourceIt->second->update(timer);
                }
                //system->getFiddleyard(i)->update(dt);
            }
        }
    }
}

/** Get the vehicle with a specific ID.
 * Constant time. Returns NULL pointer if not found.
 @param id The ID of the vehicle to be returned.
*/
Vehicle *
VehicleManager::getVehicleByID(unsigned int id)
{
    VehicleList::iterator vehIt = vehicleOverallList.begin();
    while (vehIt != vehicleOverallList.end())
    {
        Vehicle *veh = (*vehIt);
        if (veh->getVehicleID() == id)
        {
            return veh;
        }
        ++vehIt; // go to next one in list
    }
    return NULL; // not found
}

/** Get human driver vehicle.
 * Returns the first vehicle in vehicleOverallList
 * that is of type HumanVehicle.
*/
HumanVehicle *
VehicleManager::getHumanVehicle()
{
    if (!humanVehicle)
    {
        // first usage: go looking for human driver
        // usually first in list, so it's fast
        VehicleList::iterator vehIt = vehicleOverallList.begin();
        while (vehIt != vehicleOverallList.end())
        {
            HumanVehicle *driver = dynamic_cast<HumanVehicle *>(*vehIt);
            if (driver)
            {
                humanVehicle = driver;
                return humanVehicle;
            }
            else
            {
                ++vehIt; // go to next one in list
            }
        }
        return NULL; // no driver?
    }
    return humanVehicle;
}

/** Send data to an TabletUI Operator Map.
 @param operatorMap The Operator Map that receives the data.
*/
// void
// 	VehicleManager
// 	::sendDataTo(coTUIMap* operatorMap)
// {
// 	// TODO: definition in config xml, see UDP broadcast
// 	#define SEND_EVERY_X_SEC 0.5
// 	static double nextTime = cover->frameTime();
// 	if(cover->frameTime() >= nextTime) {
// 		int n = vehicleOverallList.size();
// 		if(n>0){
// 			double array[2*n];
// 			int i=0;
// 			for(VehicleList::iterator vehIt = vehicleOverallList.begin(); vehIt != vehicleOverallList.end(); ++vehIt) {
// 				array[i] = (*vehIt)->getVehicleTransform().v().x();
// 				array[i+1] = (*vehIt)->getVehicleTransform().v().y();;
// 				i = i+2;
// 			}
// 			operatorMap->sendXYPosData(n, array);
// 			nextTime = cover->frameTime() + SEND_EVERY_X_SEC;
// 		}
// 	}
// 	#undef SEND_EVERY_X_SEC
// }

/** Send data via UDP.
 @param ffzBroadcaster The object that manages the transfer.
*/
void
VehicleManager::sendDataTo(PorscheFFZ *ffzBroadcaster)
{
    static double nextSend = cover->frameTime();
    if (cover->frameTime() >= nextSend)
    {
        ffzBroadcaster->sendData(vehicleOverallList);
        nextSend = cover->frameTime() + 1.0 / ffzBroadcaster->getSendFrequency();
    }
}

/** Send data via UDP.
 @param ffzBroadcaster The object that manages the transfer.
*/
void
VehicleManager::receiveDataFrom(PorscheFFZ *ffzBroadcaster)
{
    //	static double nextSend = cover->frameTime();
    //	if(cover->frameTime() >= nextSend) {
    ffzBroadcaster->receiveData();
    //		nextSend = cover->frameTime() + 1.0/ffzBroadcaster->getSendFrequency();
    //	}
}

//

VehicleList VehicleManager::getVehicleOverallList()
{
    return vehicleOverallList;
}
