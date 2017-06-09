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
#include <osg/PositionAttitudeTransform>

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
	TraCIAPI::SubscribedValues simResults;
	TraCIAPI::SubscribedValues nextSimResults;

	osg::Group *vehicleGroup;
	osg::Box *vehicleBox;
	osg::ref_ptr<osg::PositionAttitudeTransform> vehiclePositionAttitudeTransform;

	double simTime;
	double nextSimTime;
	double currentTime;
	std::vector<int> variables;
	std::map<const std::string, osg::PositionAttitudeTransform *> loadedVehicles;

	void subscribeToSimulation();
	void updateVehiclePosition();
	osg::ShapeDrawable* getVehicle(const std::string &vehicle);
	void interpolateVehiclePosition();
	osg::Vec3d interpolatePositions(double lambda, osg::Vec3d pastPosition, osg::Vec3d futurePosition);
};
#endif
