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

#include <osg/ShapeDrawable>

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

	client.simulationStep();

	// test different methods
	int test = client.simulation.getMinExpectedNumber();
	int test2 = client.simulation.getLoadedNumber();
	int test3 = client.vehicle.getIDCount();
	std::vector<std::string> departedIDList = client.simulation.getDepartedIDList();
	std::vector<std::string> vehicleIDList = client.vehicle.getIDList();

	subscribeToSimulation();
	client.simulationStep();
	startTime = cover->frameTime();
	simResults = client.simulation.getSubscriptionResults();

	vehicleGroup = new osg::Group();
	vehicleGroup->setName("VehicleGroup");
	cover->getObjectsRoot()->addChild(vehicleGroup);

	for (std::map<std::string, TraCIAPI::TraCIValues>::iterator it = simResults.begin(); it != simResults.end(); ++it) {
		osg::Matrix vehicleMatrix = osg::Matrix::translate(it->second[VAR_POSITION3D].position.x, it->second[VAR_POSITION3D].position.y, it->second[VAR_POSITION3D].position.z);
		vehicleMatrixTransform = new osg::MatrixTransform;
		vehicleMatrixTransform->setMatrix(vehicleMatrix);
		vehicleMatrixTransform->setName("VehicleID_" + (it->first));

		osg::Geode *vehicleGeode = new osg::Geode();
		osg::ShapeDrawable *vehicleDrawable = new osg::ShapeDrawable;
		vehicleDrawable->setShape(vehicleSphere = new osg::Sphere(osg::Vec3(0, 0, 0), 30.0));
		vehicleGeode->setName("VehicleSphere");
		vehicleGeode->addDrawable(vehicleDrawable);

		vehicleMatrixTransform->addChild(vehicleGeode);
		vehicleGroup->addChild(vehicleMatrixTransform);
	}

	return true;
}

void SumoTraCI::preFrame() {
	currentTime = cover->frameTime();
	if ((currentTime - startTime) > 1) {
		startTime = currentTime;
		// subscribe to departed no of vehicles
		subscribeToSimulation();
		// make simulation step
		client.simulationStep();
		// get subscription results
		simResults = client.simulation.getSubscriptionResults();
		// update positions
		updateVehiclePosition();
	}
	//updateVehiclePosition();
}

void SumoTraCI::subscribeToSimulation() {
	std::vector<int> variables;
	variables.push_back(VAR_POSITION3D);
	if (client.simulation.getMinExpectedNumber() > 0) {
		std::vector<std::string> departedIDList = client.simulation.getDepartedIDList();
		for (std::vector<std::string>::iterator it = departedIDList.begin(); it != departedIDList.end(); ++it) {
			client.simulation.subscribe(CMD_SUBSCRIBE_VEHICLE_VARIABLE, *it, 0, TIME2STEPS(100), variables);
		}
	}
	else {
		fprintf(stderr, "no expected vehicles in simulation\n");
	}
}

TraCIAPI::TraCIValues SumoTraCI::getSimulationResults(const std::string &objID) {
	return std::map<int, TraCIAPI::TraCIValue>();
}

void SumoTraCI::updateVehiclePosition() {
	for (std::map<std::string, TraCIAPI::TraCIValues>::iterator it = simResults.begin(); it != simResults.end(); ++it) {
		osg::Matrix vehicleMatrix; 
		vehicleMatrix.makeTranslate(it->second[VAR_POSITION3D].position.x, it->second[VAR_POSITION3D].position.y, it->second[VAR_POSITION3D].position.z);
		vehicleMatrixTransform->setMatrix(vehicleMatrix);
	}
}

void SumoTraCI::updateVehiclePosition(double time) {
	double delta = currentTime - startTime;
	double x = interpolateLinear(result[VAR_POSITION3D].position.x, nextResult[VAR_POSITION3D].position.x, delta);
	double y = interpolateLinear(result[VAR_POSITION3D].position.y, nextResult[VAR_POSITION3D].position.y, delta);
	double z = interpolateLinear(result[VAR_POSITION3D].position.z, nextResult[VAR_POSITION3D].position.z, delta);
	osg::Matrix vehicleMatrix;
	fprintf(stderr, "%f %f %f\n", x, y, z);
	vehicleMatrix.makeTranslate(x, y, z);
	vehicleMatrixTransform->setMatrix(vehicleMatrix);
}

double SumoTraCI::interpolateLinear(double start, double end, double delta) {
	return start + (1.0 - delta) * (end - start);
}

void SumoTraCI::addVehicle() {
}

void SumoTraCI::removeVehicle() {
}

COVERPLUGIN(SumoTraCI)
