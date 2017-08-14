/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef OPENSCENARIO_PLUGIN_H
#define OPENSCENARIO_PLUGIN_H
/****************************************************************************\ 
 **                                                            (C)2001 HLRS  **
 **                                                                          **
 ** Description: OpenScenario                           					  **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner		                                                **
 **                                                                          **
 ** History:  								                                **
 ** Nov-01  v1	    				       		                            **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include <cover/coVRPlugin.h>

#include <TrafficSimulation/VehicleManager.h>
#include <TrafficSimulation/VehicleFactory.h>
#include <TrafficSimulation/PedestrianManager.h>
#include <TrafficSimulation/PedestrianFactory.h>
#include <VehicleUtil/RoadSystem/RoadSystem.h>
#include <TrafficSimulation/UDPBroadcast.h>
#include <TrafficSimulation/Vehicle.h>
#include "myFactory.h"


namespace OpenScenario
{
class OpenScenarioBase;
}

class Source;

class OpenScenarioPlugin : public opencover::coVRPlugin
{
public:
    OpenScenarioPlugin();
    ~OpenScenarioPlugin();
	
	static int loadOSC(const char *filename, osg::Group *loadParent, const char *key);
	int loadOSCFile(const char *filename, osg::Group *loadParent, const char *key);

	static OpenScenarioPlugin *plugin;
	static OpenScenarioPlugin *instance() { return plugin; };

	bool init();

    // this will be called in PreFrame
    void preFrame();

	void addSource(Source *s) { sources.push_back(s); };

private:
	OpenScenario::OpenScenarioBase *osdb;

	//benoetigt fuer loadRoadSystem
	bool loadRoadSystem(const char *filename);
	std::string xodrDirectory;
	std::string xoscDirectory;
	xercesc::DOMElement *getOpenDriveRootElement(std::string);
	void parseOpenDrive(xercesc::DOMElement *);
	xercesc::DOMElement *rootElement;
	osg::PositionAttitudeTransform *roadGroup;
	RoadSystem *system;
	std::list<Source *> sources;

	VehicleManager *manager;
	PedestrianManager *pedestrianManager;
    VehicleFactory *factory;
	PedestrianFactory *pedestrianFactory;

	bool tessellateRoads;
	bool tessellatePaths;
    bool tessellateBatters;
    bool tessellateObjects;
};

#endif //OPENSCENARIO_PLUGIN_H
