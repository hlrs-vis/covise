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
#include <net/covise_connect.h>

#include <TrafficSimulation/VehicleManager.h>
#include <TrafficSimulation/VehicleFactory.h>
#include <TrafficSimulation/PedestrianManager.h>
#include <TrafficSimulation/PedestrianFactory.h>
#include <VehicleUtil/RoadSystem/RoadSystem.h>
#include <TrafficSimulation/UDPBroadcast.h>
#include <TrafficSimulation/Vehicle.h>
#include "myFactory.h"
#include "ScenarioManager.h"


namespace OpenScenario
{
class OpenScenarioBase;
}

class Source;
class CameraSensor;

class OpenScenarioPlugin : public opencover::coVRPlugin
{
public:
    OpenScenarioPlugin();
    ~OpenScenarioPlugin();
	
	static int loadOSC(const char *filename, osg::Group *loadParent, const char *key);
	int loadOSCFile(const char *filename, osg::Group *loadParent, const char *key);

	static OpenScenarioPlugin *plugin;
	static OpenScenarioPlugin *instance() { return plugin; };
	std::list<CameraSensor *> cameras;
	CameraSensor *currentCamera=NULL;

	bool init();

    // this will be called in PreFrame
    void preFrame();
	// return if rendering is required
	bool update();

	void handleMessage(const char *buf);

	bool readTCPData(void * buf, unsigned int numBytes);

	void addSource(Source *s) { sources.push_back(s); };

	ScenarioManager *scenarioManager;
	OpenScenario::OpenScenarioBase *osdb;
	RoadSystem *getRoadSystem() { return system; };

private:

	//benoetigt fuer loadRoadSystem
	bool loadRoadSystem(const char *filename);
	std::string xodrDirectory;
	std::string xoscDirectory;
	xercesc::DOMElement *getOpenDriveRootElement(std::string);
	void parseOpenDrive(xercesc::DOMElement *);
	xercesc::DOMElement *rootElement;
	osg::PositionAttitudeTransform *roadGroup;
	osg::Group *trafficSignalGroup;
	RoadSystem *system;
	std::list<Source *> sources;

	VehicleManager *manager;
	PedestrianManager *pedestrianManager;
    VehicleFactory *factory;
	PedestrianFactory *pedestrianFactory;


	covise::ServerConnection *serverConn;
	covise::SimpleServerConnection *toClientConn;
	int port;


	bool tessellateRoads;
	bool tessellatePaths;
    bool tessellateBatters;
    bool tessellateObjects;
};

#endif //OPENSCENARIO_PLUGIN_H
