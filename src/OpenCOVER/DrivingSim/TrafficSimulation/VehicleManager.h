/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VehicleManager_h
#define VehicleManager_h

#include "Vehicle.h"
#include <VehicleUtil/RoadSystem/RoadSystem.h>
#include <map>
#include <list>

#include "VehicleUtils.h"

#include <cover/coTabletUI.h>

namespace TrafficSimulation
{
// forward declarations //
//
	class HumanVehicle;
	class PorscheFFZ;

    class TRAFFICSIMULATIONEXPORT VehicleManager
    {
    public:
        static VehicleManager* Instance();
        static void Destroy();

        void addVehicle(Vehicle*);
        void removeVehicle(VehicleList::iterator, vehicleUtil::Road*);
        void removeVehicle(Vehicle*, vehicleUtil::Road*);
        void removeAllAgents(double maxVel = 1.0); // delete all vehicles slower than maxVel
        void changeRoad(VehicleList::iterator, vehicleUtil::Road*, vehicleUtil::Road*, int);
        void changeRoad(Vehicle*, vehicleUtil::Road*, vehicleUtil::Road*, int);
        void moveVehicle(VehicleList::iterator, int);
        void moveVehicle(Vehicle*, int);
        Vehicle* getNextVehicle(VehicleList::iterator, int);
        Vehicle* getNextVehicle(Vehicle*, int);
        Vehicle* getNextVehicle(VehicleList::iterator, int, int);
        Vehicle* getNextVehicle(Vehicle*, int, int);
        Vehicle* getFirstVehicle(vehicleUtil::Road*);
        Vehicle* getLastVehicle(vehicleUtil::Road*);
        Vehicle* getFirstVehicle(vehicleUtil::Road*, int);
        Vehicle* getLastVehicle(vehicleUtil::Road*, int);
        Vehicle* getVehicleByID(unsigned int);
        HumanVehicle* getHumanVehicle();

        void setMaximumNumberOfVehicles(unsigned int maxNum)
        {
            maximumNumberOfVehicles = maxNum;
        }
        unsigned int getMaximumNumberOfVehicles()
        {
            return maximumNumberOfVehicles;
        }

        void sortVehicleList(vehicleUtil::Road*);

        const VehicleList& getVehicleList(vehicleUtil::Road*);

        std::map<double, Vehicle*> getSurroundingVehicles(Vehicle*);
        std::map<double, Vehicle*> getSurroundingVehicles(VehicleList::iterator);

        void setCameraVehicle(int);
        void switchToNextCamera();
        void switchToPreviousCamera();
        void unbindCamera();
        //void brakeCameraVehicle();

        bool isJunctionEmpty(vehicleUtil::Junction*);

        void moveAllVehicles(double);
        void updateFiddleyards(double, osg::Vec2d);

        //void sendDataTo(coTUIMap* operatorMap);
        void sendDataTo(PorscheFFZ* ffzBroadcaster);

        void receiveDataFrom(PorscheFFZ* ffzBroadcaster);
        VehicleList getVehicleOverallList();

        std::vector<vehicleUtil::Vector3D> acitve_fiddleyards;

    protected:
        VehicleManager();
        static VehicleManager* __instance;

        void insertVehicleAtFront(Vehicle*, vehicleUtil::Road*);
        void insertVehicleAtBack(Vehicle*, vehicleUtil::Road*);

        void moveVehicleForward(VehicleList::iterator);
        void moveVehicleBackward(VehicleList::iterator);

        void showVehicleList(vehicleUtil::Road*);

        vehicleUtil::RoadSystem* system;

        std::map<vehicleUtil::Road*, VehicleList> roadVehicleListMap;

        VehicleList vehicleOverallList;
        Vehicle* cameraVehicle;
        VehicleList::iterator cameraVehicleIt;

        VehicleDeque vehicleDecisionDeque;

        unsigned int maximumNumberOfVehicles;

    private:
        HumanVehicle* humanVehicle; // lazy initialization, so use getHumanVehicle()
    };
}
#endif
