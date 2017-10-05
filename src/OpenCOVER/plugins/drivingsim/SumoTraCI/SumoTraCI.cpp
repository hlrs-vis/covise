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
#include <TrafficSimulation/Vehicle.h>
#include <TrafficSimulation/CarGeometry.h>
#include <net/tokenbuffer.h>
#include <cover/coVRMSController.h>

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
	if(coVRMSController::instance()->isMaster())
	{
	client.close();
	}
	//cover->getScene()->removeChild(vehicleGroup);
}

bool SumoTraCI::init() {
	fprintf(stderr, "SumoTraCI::init\n");
	if(coVRMSController::instance()->isMaster())
	{
	   client.connect("localhost", 1337);
	}

	// identifiers: 57, 64, 67, 73, 79
	variables = { VAR_POSITION3D, VAR_SPEED, VAR_ANGLE, VAR_VEHICLECLASS, VAR_TYPE };
	subscribeToSimulation();

	if(coVRMSController::instance()->isMaster())
	{
	    client.simulationStep();
	    simResults = client.simulation.getSubscriptionResults();
	    sendSimResults();
	}
	else
	{
	    readSimResults();
	}
	previousResults = currentResults;
	simTime = cover->frameTime();

	if(coVRMSController::instance()->isMaster())
	{
	    client.simulationStep();
	    simResults = client.simulation.getSubscriptionResults();
	    sendSimResults();
	}
	else
	{
	    readSimResults();
	}
	nextSimTime = cover->frameTime();

	updateVehiclePosition();


	//AgentVehicle* tmpVehicle = createVehicle("passenger", "audi", "12");
	//tmpVehicle->setTransform(osg::Matrix::translate(5,0,0));

	return true;
}

void SumoTraCI::preFrame() {
	currentTime = cover->frameTime();
	if ((currentTime - nextSimTime) > 1) 
	{
		subscribeToSimulation();
		simTime = nextSimTime;
		nextSimTime = cover->frameTime();
	        previousResults = currentResults;
		
		if(coVRMSController::instance()->isMaster())
		{
		    client.simulationStep();
		    simResults = client.simulation.getSubscriptionResults();
		    sendSimResults();
		}
		else
		{
		    readSimResults();
		}
		
		updateVehiclePosition();
	}
	else {
		interpolateVehiclePosition();
	}
}

void SumoTraCI::sendSimResults()
{
    int i=0;
    if(currentResults.size() != simResults.size())
    {
        currentResults.resize(simResults.size());
    }
    for (std::map<std::string, TraCIAPI::TraCIValues>::iterator it = simResults.begin(); it != simResults.end(); ++it) 
    {
        currentResults[i].position = osg::Vec3d(it->second[VAR_POSITION3D].position.x,it->second[VAR_POSITION3D].position.y,it->second[VAR_POSITION3D].position.z);
	currentResults[i].angle = it->second[VAR_ANGLE].scalar;
	currentResults[i].vehicleClass = it->second[VAR_VEHICLECLASS].string;
	currentResults[i].vehicleType = it->second[VAR_TYPE].string;
	currentResults[i].vehicleID = it->first;		
	i++;
    }
    covise::TokenBuffer stb;
    stb << currentResults.size();
    for(int i=0;i < currentResults.size(); i++)
    {
        double x = currentResults[i].position[0];
        double y = currentResults[i].position[1];
        double z = currentResults[i].position[2];
        stb << x;
	stb << y;
	stb << z;
        stb << currentResults[i].angle;
        stb << currentResults[i].vehicleClass;
        stb << currentResults[i].vehicleType;
        stb << currentResults[i].vehicleID;
    }
    unsigned int sizeInBytes=stb.get_length();
    coVRMSController::instance()->sendSlaves(&sizeInBytes,sizeof(sizeInBytes));
    coVRMSController::instance()->sendSlaves(stb.get_data(),sizeInBytes);
}
void SumoTraCI::readSimResults()
{
    unsigned int sizeInBytes=0;
    coVRMSController::instance()->readMaster(&sizeInBytes,sizeof(sizeInBytes));
    char *buf = new char[sizeInBytes];
    coVRMSController::instance()->readMaster(buf,sizeInBytes);
    covise::TokenBuffer rtb((const char *)buf,sizeInBytes);
    unsigned int currentSize;
    rtb >> currentSize;
    if(currentSize !=currentResults.size() )
    {
        currentResults.resize(currentSize);
    }
    
    for(int i=0;i < currentResults.size(); i++)
    {
        double x;
        double y;
        double z;
        rtb >> x;
	rtb >> y;
	rtb >> z;
	currentResults[i].position[0]=x;
	currentResults[i].position[1]=y;
	currentResults[i].position[2]=z;
        rtb >>  currentResults[i].angle;
        rtb >>  currentResults[i].vehicleClass;
        rtb >>  currentResults[i].vehicleType;
        rtb >>  currentResults[i].vehicleID;
    }
}

void SumoTraCI::subscribeToSimulation() {
	if(coVRMSController::instance()->isMaster())
	{
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
}

void SumoTraCI::updateVehiclePosition() {
	osg::Matrix rotOffset;
	rotOffset.makeRotate(M_PI_2, 0, 0, 1);
        for(int i=0;i < currentResults.size(); i++)
	{
		osg::Quat orientation(osg::DegreesToRadians(currentResults[i].angle), osg::Vec3d(0, 0, -1));

		// new vehicle appeared
		if (loadedVehicles.find(currentResults[i].vehicleID) == loadedVehicles.end()) {
			loadedVehicles.insert(std::pair<const std::string, AgentVehicle *>((currentResults[i].vehicleID), createVehicle(currentResults[i].vehicleClass, currentResults[i].vehicleType, currentResults[i].vehicleID)));
		}
		else {
			/*osg::Matrix rmat,tmat;
			rmat.makeRotate(orientation);
			tmat.makeTranslate(currentResults[i].position);
			loadedVehicles.find(currentResults[i].vehicleID)->second->setTransform(rotOffset*rmat*tmat);*/
		}
	}
}

void SumoTraCI::interpolateVehiclePosition() {

	osg::Matrix rotOffset;
	rotOffset.makeRotate(M_PI_2, 0, 0, 1);
        for(int i=0;i < previousResults.size(); i++)
	{
	
		//osg::Quat orientation(osg::DegreesToRadians(previousResults[i].angle), osg::Vec3d(0, 0, -1));
	
	
		std::map<const std::string, AgentVehicle *>::iterator itr = loadedVehicles.find(previousResults[i].vehicleID);
		int currentIndex =-1;
		// delete vehicle that will vanish in next step
                for(int n=0;n < currentResults.size(); n++)
		{
		    if(previousResults[i].vehicleID == currentResults[n].vehicleID)
		    {
			currentIndex = n;
			break;
		    }
		}
		
		if (itr != loadedVehicles.end()) 
		{
		    if (currentIndex == -1) {
				//delete itr->second;
				loadedVehicles.erase(itr);
	        }
		else {
			double weight = currentTime - nextSimTime;

			osg::Vec3d position = interpolatePositions(weight, previousResults[i].position, currentResults[currentIndex].position);

			osg::Quat pastOrientation(osg::DegreesToRadians(previousResults[i].angle), osg::Vec3d(0, 0, -1));
			osg::Quat futureOrientation(osg::DegreesToRadians(currentResults[currentIndex].angle), osg::Vec3d(0, 0, -1));
			osg::Quat orientation;
			orientation.slerp(weight, pastOrientation, futureOrientation);

			osg::Matrix rmat, tmat;
			rmat.makeRotate(orientation);
			tmat.makeTranslate(position);
            AgentVehicle * av = itr->second;
			av->setTransform(rotOffset*rmat*tmat);
			VehicleState vs;
			av->getCarGeometry()->updateCarParts(1, 0, vs);
		}
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
