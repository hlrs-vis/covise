/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

 /****************************************************************************\
  **                                                            (C)2017 HLRS  **
  **                                                                          **
  ** Description: SumoTraCI - Traffic Control Interface client                **
  ** for traffic simulations with Sumo software - http://sumo.dlr.de          **
  **                                                                          **
 \****************************************************************************/

#ifdef _MSC_VER
#include "windows_config.h"
#else
#include "config.h"
#endif

#include "SumoTraCI.h"

#include <utils/common/SUMOTime.h>

#include <iostream>
#include <algorithm>
#include <cmath>

#include <cover/coVRPluginSupport.h>
#include <config/CoviseConfig.h>

int gPrecision;

using namespace opencover;

SumoTraCI::SumoTraCI() {
	fprintf(stderr, "SumoTraCI::SumoTraCI\n");
	const char *coviseDir = getenv("COVISEDIR");
	std::string defaultDir = std::string(coviseDir) + "/share/covise/vehicles";
	vehicleDirectory = covise::coCoviseConfig::getEntry("value","COVER.Plugin.SumoTraCI.VehicleDirectory", defaultDir.c_str());
}

SumoTraCI::~SumoTraCI() {
	fprintf(stderr, "SumoTraCI::~SumoTraCI\n");
	client.close();
	cover->getScene()->removeChild(vehicleGroup);
}

bool SumoTraCI::init() {
	fprintf(stderr, "SumoTraCI::init\n");
	client.connect("localhost", 1337);

	// identifiers: 57, 64, 67, 73, 79
	variables = { VAR_POSITION3D, VAR_SPEED, VAR_ANGLE, VAR_VEHICLECLASS, VAR_TYPE };
	subscribeToSimulation();

	client.simulationStep();
	simResults = client.simulation.getSubscriptionResults();
	simTime = cover->frameTime();

	client.simulationStep();
	nextSimResults = client.simulation.getSubscriptionResults();
	nextSimTime = cover->frameTime();

	updateVehiclePosition();


	//AgentVehicle* tmpVehicle = createVehicle("passenger", "audi", "12");
	//tmpVehicle->setTransform(osg::Matrix::translate(5,0,0));

	return true;
}

	void SumoTraCI::preFrame() {
	currentTime = cover->frameTime();
	if ((currentTime - nextSimTime) > 1) {
		subscribeToSimulation();
		simTime = nextSimTime;
		client.simulationStep();
		nextSimTime = cover->frameTime();
		simResults = nextSimResults;
		nextSimResults = client.simulation.getSubscriptionResults();
		updateVehiclePosition();
	}
	else {
		interpolateVehiclePosition();
	}
}

void SumoTraCI::subscribeToSimulation() {
	if (client.simulation.getMinExpectedNumber() > 0) {
		std::vector<std::string> departedIDList = client.simulation.getDepartedIDList();
		for (std::vector<std::string>::iterator it = departedIDList.begin(); it != departedIDList.end(); ++it) {
			client.simulation.subscribe(CMD_SUBSCRIBE_VEHICLE_VARIABLE, *it, 0, TIME2STEPS(1000), variables);
		}
	} 
	else {
		fprintf(stderr, "no expected vehicles in simulation\n");
	}
}

void SumoTraCI::updateVehiclePosition() {
	osg::Matrix rotOffset;
	rotOffset.makeRotate(M_PI_2, 0, 0, 1);
	for (std::map<std::string, TraCIAPI::TraCIValues>::iterator it = simResults.begin(); it != simResults.end(); ++it) {
		osg::Vec3d position(it->second[VAR_POSITION3D].position.x, it->second[VAR_POSITION3D].position.y, it->second[VAR_POSITION3D].position.z);
		osg::Quat orientation(osg::DegreesToRadians(it->second[VAR_ANGLE].scalar), osg::Vec3d(0, 0, -1));

		// new vehicle appeared
		if (loadedVehicles.find(it->first) == loadedVehicles.end()) {
			std::string vehicleClass = it->second[VAR_VEHICLECLASS].string;
			std::string vehicleType = it->second[VAR_TYPE].string;
			std::string vehicleID = it->first;
			loadedVehicles.insert(std::pair<const std::string, AgentVehicle *>((it->first), createVehicle(vehicleClass, vehicleType, vehicleID)));
		}
		else {
			osg::Matrix rmat,tmat;
			rmat.makeRotate(orientation);
			tmat.makeTranslate(position);
			AgentVehicle * av = loadedVehicles.find(it->first)->second;
			av->setTransform(rotOffset*rmat*tmat);
			VehicleState vs;
			av->getCarGeometry()->updateCarParts(1, 0, vs);
		}
	}
}

void SumoTraCI::interpolateVehiclePosition() {

	osg::Matrix rotOffset;
	rotOffset.makeRotate(M_PI_2, 0, 0, 1);
	for (std::map<std::string, TraCIAPI::TraCIValues>::iterator it = simResults.begin(); it != simResults.end(); ++it) {
		std::map<const std::string, AgentVehicle *>::iterator itr = loadedVehicles.find(it->first);
		// delete vehicle that will vanish in next step
		std::map<std::string, TraCIAPI::TraCIValues>::iterator vehicleInNextSim = nextSimResults.find(it->first);
		if (vehicleInNextSim == nextSimResults.end()) {
			if (itr != loadedVehicles.end()) {
				delete itr->second;
				loadedVehicles.erase(itr);
			}
		}
		else {
			double weight = currentTime - nextSimTime;

			osg::Vec3d pastPosition(it->second[VAR_POSITION3D].position.x, it->second[VAR_POSITION3D].position.y, it->second[VAR_POSITION3D].position.z);
			osg::Vec3d futurePosition(vehicleInNextSim->second[VAR_POSITION3D].position.x, vehicleInNextSim->second[VAR_POSITION3D].position.y, vehicleInNextSim->second[VAR_POSITION3D].position.z);
			osg::Vec3d position = interpolatePositions(weight, pastPosition, futurePosition);

			osg::Quat pastOrientation(osg::DegreesToRadians(it->second[VAR_ANGLE].scalar), osg::Vec3d(0, 0, -1));
			osg::Quat futureOrientation(osg::DegreesToRadians(vehicleInNextSim->second[VAR_ANGLE].scalar), osg::Vec3d(0, 0, -1));
			osg::Quat orientation;
			orientation.slerp(weight, pastOrientation, futureOrientation);

			osg::Matrix rmat, tmat;
			rmat.makeRotate(orientation);
			tmat.makeTranslate(position);
			loadedVehicles.find(it->first)->second->setTransform(rotOffset*rmat*tmat);
		}
	}
}

osg::Vec3d SumoTraCI::interpolatePositions(double lambda, osg::Vec3d pastPosition, osg::Vec3d futurePosition) {
	osg::Vec3d interpolatedPosition;
	for (int i = 0; i < 3; ++i) {
		double interpolatedPoint = futurePosition[i] + (1.0 - lambda) * (pastPosition[i] - futurePosition[i]);
		interpolatedPosition[i] = interpolatedPoint;
	}
	return interpolatedPosition;
}

AgentVehicle* SumoTraCI::createVehicle(const std::string &vehicleClass, const std::string &vehicleType, const std::string &vehicleID)
{
	return new AgentVehicle(vehicleID, new CarGeometry(vehicleID, vehicleDirectory+"/"+vehicleClass+"/"+vehicleType+"/"+vehicleType+".wrl", true), 0, NULL, 0, 1, 0.0, 1);;
}

COVERPLUGIN(SumoTraCI)
