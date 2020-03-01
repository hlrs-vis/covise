/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VehicleFactory_h
#define VehicleFactory_h

#include "Vehicle.h"
#include "VehicleGeometry.h"
#include <VehicleUtil/RoadSystem/Road.h>

#include <xercesc/dom/DOM.hpp>
#include <VehicleUtil/RoadSystem/RoadSystem.h>

class TRAFFICSIMULATIONEXPORT VehicleFactory
{
public:
    enum RoadVehicleType
    {
        ROADVEHICLE_CAR,
        ROADVEHICLE_TRUCK,
        ROADVEHICLE_BICYCLE
    };

    enum IntelligenceType
    {
        INTELLIGENCE_AGENT,
        INTELLIGENCE_DONKEY,
        INTELLIGENCE_SCRIPTING,
        INTELLIGENCE_HUMAN,
    };

    static VehicleFactory *Instance();
    static void Destroy();

    void deleteRoadVehicle(Vehicle *);
    Vehicle *cloneRoadVehicle(std::string, std::string, vehicleUtil::Road *, double, int, double, int);

    Vehicle *createRoadVehicle(std::string, std::string, RoadVehicleType = ROADVEHICLE_CAR, IntelligenceType = INTELLIGENCE_AGENT, std::string = "cars/ibiza.3ds");
    Vehicle *createRoadVehicle(std::string, std::string, vehicleUtil::Road *road, double, int, int, double = 100, RoadVehicleType = ROADVEHICLE_CAR, IntelligenceType = INTELLIGENCE_AGENT, std::string = "cars/ibiza.3ds");
    Vehicle *createRoadVehicle(std::string, std::string, std::string, std::string, std::string = "cars/ibiza.3ds");
    Vehicle *createRoadVehicle(std::string, std::string, std::string, double, int, int, double, std::string, std::string, std::string = "cars/ibiza.3ds");

    void parseOpenDrive(xercesc::DOMElement *, const std::string & = ".");
    void createTileVehicle(int, int, std::vector<vehicleUtil::Vector3D>);
    // Neu Andreas 27-11-2012
    double generateStartVelocity(std::list<vehicleUtil::RoadLineSegment *>::iterator it, vehicleUtil::Pool *currentPool, int);

protected:
    VehicleFactory();

    static VehicleFactory *__instance;
    std::map<std::string, Vehicle *> roadVehicleMap;
    std::set<Vehicle *> blueprintVehicleSet;
};

#endif
