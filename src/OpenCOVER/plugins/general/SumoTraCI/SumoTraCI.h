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
 **                                                                          **
 ** Author: Myriam Guedey	                                                 **
 **                                                                          **
 ** History:  								                                 **
 ** Feb-17  v1	    				       		                             **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include <cover/coVRPlugin.h>

#include <osg/MatrixTransform>

#include <utils/traci/TraCIAPI.h>

#include <vector>

using namespace opencover;

class SumoTraCI : public opencover::coVRPlugin
{
public:
    SumoTraCI();
    ~SumoTraCI();

    void preFrame();
	bool init();

private:
	TraCIAPI client;
	TraCIAPI::TraCIValues result;
	TraCIAPI::TraCIValues nextResult;
	TraCIAPI::SubscribedValues simResults;

	osg::Group *vehicleGroup;
	osg::Sphere *vehicleSphere;
	osg::ref_ptr<osg::MatrixTransform> vehicleMatrixTransform;

	double startTime;
	double currentTime;
	double resultTime;
	double nextResultTime;

	void SumoTraCI::addVehicle();
	void SumoTraCI::removeVehicle();
	void SumoTraCI::subscribeToSimulation();
	void SumoTraCI::updateVehiclePosition();
	void SumoTraCI::updateVehiclePosition(double time);
	double SumoTraCI::interpolateLinear(double start, double end, double fraction);
	TraCIAPI::TraCIValues SumoTraCI::getSimulationResults(const std::string& objID);
	TraCIAPI::SubscribedValues SumoTraCI::getSimulationResults();
};
#endif
