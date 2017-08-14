/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_TRAFFICSIMULATION_H
#define _CO_TRAFFICSIMULATION_H
/****************************************************************************\
 **                                                            (C)2001 HLRS  **
 **                                                                          **
 ** Description: TrafficSimulation Plugin                                    **
 **                                                                          **
 **                                                                          **
 ** Author: Florian Seybold, U.Woessner		                                **
 **                                                                          **
 ** History:  								                                         **
 ** Nov-01  v1	    				       		                                   **
 **                                                                          **
 **                                                                          **
\****************************************************************************/
#include <cover/coVRPlugin.h>
#include <osg/Texture2D>
#include <osg/Material>
#include <osg/Geode>
#include <osg/ShapeDrawable>
#include <osg/Switch>
#include <osg/Geometry>
#include <osg/MatrixTransform>
#include <osg/StateSet>
#include <osg/Material>
#include <vector>
#include <osg/PositionAttitudeTransform>
#include <util/coExport.h>

#include <random>

#include "RoadSystem/RoadSystem.h"
#include "VehicleManager.h"
#include "VehicleFactory.h"
#include "PedestrianManager.h"
#include "PedestrianFactory.h"
//#include "RoadFootprint.h"

#include "UDPBroadcast.h"

#include "Vehicle.h"
using namespace covise;
using namespace opencover;

// forward declarations //
//
class PorscheFFZ;
class TRAFFICSIMULATIONEXPORT coTrafficSimulation : public coVRPlugin, public coTUIListener
{
public:
    ~coTrafficSimulation();

    void runSimulation();
    void haltSimulation();

    VehicleManager *getVehicleManager();
    PedestrianManager *getPedestrianManager();

    bool init();


    // this will be called in PreFrame
    void preFrame();


    unsigned long getIntegerRandomNumber();
    double getZeroOneRandomNumber();

    static int counter;
    static int createFreq;
    static double min_distance;
    static int min_distance_tui;
    static int max_distance_tui;
    // static int min_distance_50;
    // static int min_distance_100;
    static double delete_at;
    static int delete_delta;
    static int useCarpool;
    static float td_multiplier;
    static float placeholder;
    static int maxVel;
    static int minVel;
    static int maxVehicles;

	static coTrafficSimulation *instance();
	static void useInstance();
	static void freeInstance();

	xercesc::DOMElement *getOpenDriveRootElement(std::string);


	void parseOpenDrive(xercesc::DOMElement *);

	bool loadRoadSystem(const char *filename);
	void deleteRoadSystem();
	RoadSystem *system;
	VehicleFactory *factory;
	PedestrianFactory *pedestrianFactory;
	osg::PositionAttitudeTransform *roadGroup;
	osg::Group *trafficSignalGroup;
	osg::Node *terrain;
	xercesc::DOMElement *rootElement;
	bool tessellateRoads;
	bool tessellatePaths;
	bool tessellateBatters;
	bool tessellateObjects;
	bool runSim;
private:
	coTrafficSimulation();
    //osg::Group* roadGroup;

	static coTrafficSimulation *myInstance;
	static int numInstances;


    // operator map //
    //coTUITab* operatorMapTab;
    //coTUIMap* operatorMap;

    // broadcaster for ffz positions //
    PorscheFFZ *ffzBroadcaster;


    osg::ref_ptr<osg::MatrixTransform> sphereTransform;
    osg::ref_ptr<osg::Sphere> sphere;
    osg::ref_ptr<osg::Geode> sphereGeode;
    osg::ref_ptr<osg::StateSet> sphereGeoState;
    osg::ref_ptr<osg::Material> redmtl;
    osg::ref_ptr<osg::Material> greenmtl;

    std::mt19937 mersenneTwisterEngine;
    std::uniform_real_distribution<double> uniformDist;
    std::string xodrDirectory;


};
#endif
