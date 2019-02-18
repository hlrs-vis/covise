/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef SUMOTRACI_H
#define SUMOTRACI_H

/****************************************************************************\ 
 **                                                            (C)2017 HLRS  **
 **                                                                          **
 ** Description: SumoTraCI - Traffic Control Interface client                **
 ** for traffic simulations with Sumo software - http://sumo.dlr.de          **
 **                                                                          **
\****************************************************************************/

#include <cover/coVRPlugin.h>

#include <osg/ShapeDrawable>

#include <utils/traci/TraCIAPI.h>

#include <vector>
#include <TrafficSimulation/AgentVehicle.h>
#include <TrafficSimulation/PedestrianFactory.h>

using namespace opencover;

struct simData
{
    osg::Vec3d position;
    double  angle;
    std::string vehicleClass;
    std::string vehicleType;
    std::string vehicleID;
};

class SumoTraCI : public opencover::coVRPlugin
{
public:
    SumoTraCI();
    ~SumoTraCI();

    void preFrame();
	bool init();

private:
	TraCIAPI client;
    libsumo::SubscriptionResults simResults;
    libsumo::SubscriptionResults pedestrianSimResults;
	std::vector<simData> currentResults;
	std::vector<simData> previousResults;
        std::map<std::string, AgentVehicle *> vehicleMap;
        AgentVehicle *getAgentVehicle(const std::string &vehicleID, const std::string &vehicleClass, const std::string &vehicleType);

	osg::Group *vehicleGroup;
	std::string vehicleDirectory;

    osg::ref_ptr<osg::Group> pedestrianGroup;

    PedestrianFactory *pf;
    std::map<std::string, PedestrianGeometry *> pedestrianMap;
    std::map<std::string, PedestrianGeometry *> loadedPedestrians;
    PedestrianGeometry* createPedestrian(const std::string &vehicleClass, const std::string &vehicleType, const std::string &vehicleID);
    PedestrianGeometry* getPedestrian(const std::string &vehicleID, const std::string &vehicleClass, const std::string &vehicleType);
    double interpolateAngles(double lambda, double pastAngle, double futureAngle);

	double simTime;
	double nextSimTime;
	double currentTime;
	std::vector<int> variables;
	std::map<const std::string, AgentVehicle *> loadedVehicles;

	void subscribeToSimulation();
	void updateVehiclePosition();
	AgentVehicle* createVehicle(const std::string &vehicleClass, const std::string &vehicleType, const std::string &vehicleID);
	void interpolateVehiclePosition();
	osg::Vec3d interpolatePositions(double lambda, osg::Vec3d pastPosition, osg::Vec3d futurePosition);
void sendSimResults();
void readSimResults();
};
#endif
