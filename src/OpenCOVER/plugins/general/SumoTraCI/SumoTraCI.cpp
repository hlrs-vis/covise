/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

 /****************************************************************************\
  **                                                            (C)2017 HLRS  **
  **                                                                          **
  **                                                                          **
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

int gPrecision;

using namespace opencover;

SumoTraCI::SumoTraCI() {
	fprintf(stderr, "SumoTraCI::SumoTraCI\n");
}

SumoTraCI::~SumoTraCI() {
	fprintf(stderr, "SumoTraCI::~SumoTraCI\n");
	client.close();
	cover->getScene()->removeChild(vehicleGroup);
}

bool SumoTraCI::init() {
	fprintf(stderr, "SumoTraCI::init\n");
	client.connect("localhost", 1337);

	// identifiers: 57, 64, 67, 73
	variables = { VAR_POSITION3D, VAR_SPEED, VAR_ANGLE, VAR_VEHICLECLASS };
	subscribeToSimulation();

	client.simulationStep();
	simResults = client.simulation.getSubscriptionResults();
	startTime0 = cover->frameTime();

	client.simulationStep();
	nextSimResults = client.simulation.getSubscriptionResults();
	startTime1 = cover->frameTime();

	vehicleGroup = new osg::Group();
	vehicleGroup->setName("VehicleGroup");
	cover->getObjectsRoot()->addChild(vehicleGroup);

	updateVehiclePosition();

	return true;
}

	void SumoTraCI::preFrame() {
	currentTime = cover->frameTime() - getTimeSpan();
	if ((currentTime - startTime0) > 1) {
		startTime0 = currentTime;
		subscribeToSimulation();	// subscribe to departed no of vehicles
		client.simulationStep();
		startTime1 = cover->frameTime();
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
	for (std::map<std::string, TraCIAPI::TraCIValues>::iterator it = simResults.begin(); it != simResults.end(); ++it) {
		osg::Vec3d position(it->second[VAR_POSITION3D].position.x, it->second[VAR_POSITION3D].position.y, it->second[VAR_POSITION3D].position.z);
		osg::Quat orientation(osg::DegreesToRadians(it->second[VAR_ANGLE].scalar), osg::Vec3d(0, 0, -1));

		// new vehicle appeared
		if (loadedVehicles.find(it->first) == loadedVehicles.end()) {
			osg::Geode *vehicleGeode = new osg::Geode();
			std::string vehicleType = it->second[VAR_VEHICLECLASS].string;
			vehicleGeode->addDrawable(getVehicle(vehicleType));
			vehicleGeode->setName(vehicleType);

			vehiclePositionAttitudeTransform = new osg::PositionAttitudeTransform();
			vehiclePositionAttitudeTransform->setName(it->first);
			vehiclePositionAttitudeTransform->setPosition(position);
			vehiclePositionAttitudeTransform->setAttitude(orientation);
			loadedVehicles.insert(std::pair<const std::string, osg::PositionAttitudeTransform *>((it->first), vehiclePositionAttitudeTransform));

			vehiclePositionAttitudeTransform->addChild(vehicleGeode);
			vehicleGroup->addChild(vehiclePositionAttitudeTransform);
		}
		else {
			loadedVehicles.find(it->first)->second->setAttitude(orientation);
			loadedVehicles.find(it->first)->second->setPosition(position);
		}
	}
}

void SumoTraCI::interpolateVehiclePosition() {
	for (std::map<std::string, TraCIAPI::TraCIValues>::iterator it = simResults.begin(); it != simResults.end(); ++it) {
		// delete vehicle that will vanish in next step
		std::map<std::string, TraCIAPI::TraCIValues>::iterator vehicleInNextSim = nextSimResults.find(it->first);
		if (vehicleInNextSim == nextSimResults.end()) {
			std::map<const std::string, osg::PositionAttitudeTransform *>::iterator itr = loadedVehicles.find(it->first);
			if (itr != loadedVehicles.end()) {
				itr->second->getParent(0)->removeChild(itr->second);
				loadedVehicles.erase(itr);
			}
		}
		else {
			double lambda = currentTime - startTime0;

			double x0 = it->second[VAR_POSITION3D].position.x;
			double x1 = vehicleInNextSim->second[VAR_POSITION3D].position.x;
			double y0 = it->second[VAR_POSITION3D].position.y;
			double y1 = vehicleInNextSim->second[VAR_POSITION3D].position.y;
			double z0 = it->second[VAR_POSITION3D].position.z;
			double z1 = vehicleInNextSim->second[VAR_POSITION3D].position.z;
			double x = interpolatePositions(x1, x0, lambda);
			double y = interpolatePositions(y1, y0, lambda);
			double z = interpolatePositions(z1, z0, lambda);

			osg::Quat orientation0(osg::DegreesToRadians(it->second[VAR_ANGLE].scalar), osg::Vec3d(0, 0, -1));
			osg::Quat orientation1(osg::DegreesToRadians(vehicleInNextSim->second[VAR_ANGLE].scalar), osg::Vec3d(0, 0, -1));
			osg::Vec3d position(x, y, z);
			osg::Quat orientation;
			orientation.slerp(lambda, orientation0, orientation1);

			loadedVehicles.find(it->first)->second->setAttitude(orientation);
			loadedVehicles.find(it->first)->second->setPosition(position);
		}
	}
}

double SumoTraCI::interpolatePositions(double start, double end, double weight) {
	return start + (1.0 - weight) * (end - start);
}

osg::ShapeDrawable* SumoTraCI::getVehicle(const std::string &vehicleType) {
	osg::ShapeDrawable *vehicleDrawable = new osg::ShapeDrawable;
	if (vehicleType.compare("bicycle") == 0) {
		vehicleDrawable->setShape(vehicleBox = new osg::Box(osg::Vec3(0, 0, 0), 0.65, 1.6, 1.7));
	}
	else if (vehicleType.compare("passenger") == 0) {
		vehicleDrawable->setShape(vehicleBox = new osg::Box(osg::Vec3(0, 0, 0), 1.8, 4.3, 1.5));
	}
	else if (vehicleType.compare("truck") == 0) {
		vehicleDrawable->setShape(vehicleBox = new osg::Box(osg::Vec3(0, 0, 0), 2.4, 7.1, 2.4));
	}
	else if (vehicleType.compare("bus") == 0) {
		vehicleDrawable->setShape(vehicleBox = new osg::Box(osg::Vec3(0, 0, 0), 2.5, 12.0, 3.4));
	}
	else if (vehicleType.compare("motorcycle") == 0) {
		vehicleDrawable->setShape(vehicleBox = new osg::Box(osg::Vec3(0, 0, 0), 0.9, 2.2, 1.5));
	}
	else {
		vehicleDrawable->setShape(vehicleBox = new osg::Box(osg::Vec3(0, 0, 0), 2.5, 2.5, 2.5));
	}
	return vehicleDrawable;
}

double SumoTraCI::getTimeSpan() {
	return startTime1 - startTime0;
}

COVERPLUGIN(SumoTraCI)
