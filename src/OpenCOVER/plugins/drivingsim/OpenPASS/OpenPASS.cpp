/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <iostream>
#include <algorithm>
#include <cmath>
#include <random>
#include "OpenPASS.h"

#include <cover/coVRPluginSupport.h>
#include <cover/coVRMSController.h>
#include <config/CoviseConfig.h>
#include <TrafficSimulation/Vehicle.h>
#include <TrafficSimulation/CarGeometry.h>
#include <net/tokenbuffer.h>
#include <util/unixcompat.h>

#include <cover/ui/Menu.h>
#include <cover/ui/Button.h>
#include <cover/ui/Slider.h>
#include <osg/Switch>


int gPrecision;

using namespace opencover;

OpenPASS::OpenPASS() : ui::Owner("OpenPASS", cover->ui)
{
    fprintf(stderr, "OpenPASS::OpenPASS\n");
    initUI();
    const char *coviseDir = getenv("COVISEDIR");
    std::string defaultDir = std::string(coviseDir) + "/share/covise/vehicles";
    vehicleDirectory = covise::coCoviseConfig::getEntry("value","COVER.Plugin.OpenPASS.VehicleDirectory", defaultDir.c_str());
    //AgentVehicle *av = getAgentVehicle("veh_passenger","passenger","veh_passenger");
    //av = getAgentVehicle("truck","truck","truck_truck");
    pf = PedestrianFactory::Instance();
    pedestrianGroup =  new osg::Switch;
    pedestrianGroup->setName("pedestrianGroup");
	cover->getObjectsRoot()->addChild(pedestrianGroup.get());
	passengerGroup = new osg::Switch;
	passengerGroup->setName("passengerGroup");
	cover->getObjectsRoot()->addChild(passengerGroup.get());
	bicycleGroup = new osg::Switch;
	bicycleGroup->setName("bicycleGroup");
	cover->getObjectsRoot()->addChild(bicycleGroup.get());
	busGroup = new osg::Switch;
	busGroup->setName("busGroup");
	cover->getObjectsRoot()->addChild(busGroup.get());
    pedestrianGroup->setNodeMask(pedestrianGroup->getNodeMask() & ~Isect::Update); // don't use the update traversal, tey are updated manually when in range
    getPedestriansFromConfig();
    getVehiclesFromConfig();
    loadAllVehicles(); 
	lineUpAllPedestrianModels(); // preload pedestrians;
}
/*
AgentVehicle *OpenPASS::getAgentVehicle(const std::string &vehicleID, const std::string &vehicleClass, const std::string &vehicleType)
{

    auto vehiclesPair = vehicleModelMap.find(vehicleClass);
    auto vehicles = vehiclesPair->second;
    static std::mt19937 gen2(0);
    std::uniform_int_distribution<> dis(0, vehicles->size()-1);
    int vehicleIndex = dis(gen2);

    AgentVehicle *av;
    auto avIt = vehicleMap.find(vehicles->at(vehicleIndex).vehicleName);
    if(avIt != vehicleMap.end())
    {
        av = avIt->second;
    }
    else
    {
		osg::Group* parent = nullptr;
		if (vehicleClass == "bus")
			parent = busGroup;
		if (vehicleClass == "passenger")
			parent = passengerGroup;
		if (vehicleClass == "bicycle")
			parent = bicycleGroup;
        av= new AgentVehicle(vehicleID, new CarGeometry(vehicleID, vehicles->at(vehicleIndex).fileName, false, parent), 0, NULL, 0, 1, 0.0, 1);
        vehicleMap.insert(std::pair<std::string, AgentVehicle *>(vehicles->at(vehicleIndex).vehicleName,av));
    }
    return av;
}
*/


OpenPASS::~OpenPASS()
{
    fprintf(stderr, "OpenPASS::~OpenPASS\n");
    
}

void OpenPASS::lineUpAllPedestrianModels()
{
    for (int i = 0; i < pedestrianModels.size(); i++)
    {
        std::string ID = "pedestrianTest" + std::to_string(i);
        PedestrianAnimations a = PedestrianAnimations();

        int pedestrianIndex = i;

        pedestrianModel p = pedestrianModels[pedestrianIndex];
        PedestrianGeometry* pedgeom = new PedestrianGeometry(ID, p.fileName, p.scale, 40.0, a, pedestrianGroup);
        osg::Vec3d position = osg::Vec3d((double)i, 0.0, 50.0);

        osg::Quat orientation(osg::DegreesToRadians(0.0), osg::Vec3d(0, 0, -1));

        Transform trans = Transform(Vector3D(position.x(), position.y(), position.z()), Quaternion(orientation.w(), orientation.x(), orientation.y(), orientation.z()));
        pedgeom->setTransform(trans, M_PI);
    }
}

bool OpenPASS::initUI()
{
	openpassMenu = new ui::Menu("OpenPASS", this);
	pedestriansVisible = new ui::Button(openpassMenu, "Pedestrians");
	pedestriansVisible->setState(true);
	pedestriansVisible->setCallback([this](bool state) {
		setPedestriansVisible(state);
		});
    
    return true;
}

void OpenPASS::preFrame()
{

}


AgentVehicle* OpenPASS::createVehicle(const std::string &vehicleClass, const std::string &vehicleType, const std::string &vehicleID)
{
    AgentVehicle *av = getAgentVehicle(vehicleID,vehicleClass,vehicleType);

    VehicleParameters vp;
    if (vehicleType.compare("escooter"))
    {
        vp.rangeLOD = 400;
    }
    else
    {
        vp.rangeLOD = 1600;
    }
    return new AgentVehicle(av, vehicleID,vp,NULL,0.0,0);
}

COVERPLUGIN(OpenPASS)
