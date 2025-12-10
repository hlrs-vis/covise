/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef OPENCOVER_PLUGINS_TRAFFIC_TRAFFIC_H
#define OPENCOVER_PLUGINS_TRAFFIC_TRAFFIC_H

#include <cover/coVRPlugin.h>
#include <cover/ui/Button.h>
#include <cover/ui/Menu.h>
#include <cover/ui/Owner.h>
#include <cover/ui/Slider.h>

#include <osg/ShapeDrawable>

#include <random>
#include <vector>

#include "Geometry.h"
#include "sumo/ConnectorSumo.h"

struct VehicleState
{
    std::string id;
    std::string vehicleClass;
    osg::Vec3d position;
    double angle = 0.0;
};

struct SimulationState
{
    std::vector<VehicleState> vehicles;
};

struct VehicleModel
{
    std::string name;
    std::string fileName;
    double scale = 1;
};

enum GeometryType : uint8_t
{
    // A simple static mesh, moving around.
    STATIC,

    // A car geometry may have steerable and turning wheels and conditional
    // lights like break lights and turn indicators.
    CAR,

    // Using cal3d/osgcal for character rendering with
    // skeleton/skin animations.
    CHARACTER,

};

struct VehicleClass
{
    std::string name;
    GeometryType geometryType = GeometryType::CAR;
    std::vector<VehicleModel> models;
};

struct Vehicle
{
    std::unique_ptr<Geometry> geometry;

    osg::Vec3 position;

    // We call this heading, but it is obviously still a mathematical angle in
    // radians, from positive X, counter-clockwise around positive Z
    // (right hand rule).
    double heading;

    osg::Vec3 targetPosition;
    double targetHeading;
    double timeToTarget = 0.0;
};

class Traffic : public opencover::coVRPlugin, public opencover::ui::Owner
{
public:
    Traffic();
    ~Traffic();
    static Traffic *instance() { return thisPlugin; };

    void preFrame() override;

private:
    bool initUI();

    // UI components
    opencover::ui::Menu *trafficMenu;
    // std::map<std::string, opencover::ui::Button*> vehicleClassVisibleButtons;
    opencover::ui::Button *pauseButton;
    opencover::ui::Slider *simulationSpeedSlider;

    // Simulation data, to be partially moved to subclass for connector
    SimulationState currentSimulationState;
    SimulationState previousSimulationState;
    std::map<std::string, Vehicle> vehicles;

    ConnectorSumo connector;

    // References to OSG nodes
    osg::ref_ptr<osg::Switch> trafficGroup;
    std::map<std::string, osg::ref_ptr<osg::Switch>> vehicleClassGroups;

    double interpolateAngles(double lambda, double pastAngle, double futureAngle);

    std::map<std::string, VehicleClass> vehicleClasses;
    void loadVehicleClasses();

    bool update() override;

    void getVehiclesFromConfig();
    bool connected = false;
    double lastResultTime = 0.0;
    double previousTime = 0.0;
    double currentTime = 0.0;
    int timeStep = 0;
    std::vector<int> variables;

    void getSimulationResults(double deltaTime);

    int uniqueIDValue = 0;

    void processNewResults();
    Vehicle &createVehicle(const std::string &id, const VehicleClass &vehicleClass);
    void interpolateVehiclePosition(double deltaTime);
    osg::Vec3d interpolatePositions(double lambda, osg::Vec3d pastPosition, osg::Vec3d futurePosition);
    void sendSimResults();
    void readSimResults();

    static Traffic *thisPlugin;
};

#endif
