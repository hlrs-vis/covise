/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Traffic.h"

#include <cmath>
#include <iostream>
#include <random>

#include <PluginUtil/PluginMessageTypes.h>
#include <boost/algorithm/string.hpp>
#include <config/CoviseConfig.h>
#include <cover/coVRFileManager.h>
#include <cover/coVRMSController.h>
#include <cover/coVRPluginSupport.h>
#include <net/tokenbuffer.h>
#include <util/unixcompat.h>

#include <cover/ui/Button.h>
#include <cover/ui/FileBrowser.h>
#include <cover/ui/Menu.h>
#include <cover/ui/Slider.h>
#include <osg/Switch>

#include "CarGeometry.h"
#include "PedestrianGeometry.h"

Traffic *Traffic::thisPlugin = nullptr;

Traffic::Traffic()
    : coVRPlugin(COVER_PLUGIN_NAME)
    , opencover::ui::Owner("Traffic", opencover::cover->ui)
{
    thisPlugin = this;
    initUI();

    trafficGroup = new osg::Switch;
    trafficGroup->setName("traffic");
    opencover::cover->getObjectsRoot()->addChild(trafficGroup.get());

    // Load models
    loadVehicleClasses();

    for (auto &[_, vehicleClass] : vehicleClasses)
    {
        osg::ref_ptr<osg::Switch> group = new osg::Switch;
        group->setName(vehicleClass.name);
        trafficGroup->addChild(group.get());
        vehicleClassGroups[vehicleClass.name] = group;
    }

    // Don't use the update traversal on pedestrians, they are updated manually
    // when in range.
    auto &pedestrianGroup = vehicleClassGroups["pedestrian"];
    if (pedestrianGroup)
    {
        pedestrianGroup->setNodeMask(pedestrianGroup->getNodeMask() & ~opencover::Isect::Update);
    }
}

// void Traffic::loadConfig(const std::string &configFileName) {
//     if (connected) {
//         for (auto currentEntity = vehicles.begin(); currentEntity != vehicles.end();
//              currentEntity++) // delete all vehicles
//         {
//             delete currentEntity->second;
//         }
//         vehicles.clear();
//         std::vector<std::string> args(1);
//         args[0] = configFileName;
//         client.load(args);
//         subscribeToSimulation();
//     }
// }

Traffic::~Traffic()
{
    // TODO: check cleanup

    if (opencover::coVRMSController::instance()->isMaster())
    {
        vehicles.clear();
    }

    opencover::cover->getScene()->removeChild(trafficGroup);
}

void Traffic::getSimulationResults(double deltaTime)
{
    if (opencover::coVRMSController::instance()->isMaster())
    {
        if (connector.isConnected())
        {
            previousSimulationState = currentSimulationState; // TODO: is this copy?
            connector.getSimulationState(currentSimulationState);
        }

        // sendSimResults();
    }
    else
    {
        // readSimResults();
    }
}

bool Traffic::initUI()
{
    trafficMenu = new opencover::ui::Menu("Traffic", this);

    // for (auto& [_, vehicleClass]: vehicleClasses) {
    //     opencover::ui::Button* button = new opencover::ui::Button(trafficMenu, vehicleClass.name);
    //     button->setState(true);
    //     // button->setCallback([this, vehicleClass](bool state) { setVehicleClassVisible(vehicleClass, state); });
    //     // vehicleClassVisibleButtons.emplace(vehicleClass, button);
    // }

    pauseButton = new opencover::ui::Button(trafficMenu, "Pause");

    simulationSpeedSlider = new opencover::ui::Slider(trafficMenu, "timeScale");
    simulationSpeedSlider->setText("Time scale");
    simulationSpeedSlider->setBounds(0.1, 10.0);
    simulationSpeedSlider->setIntegral(false);
    simulationSpeedSlider->setScale(opencover::ui::Slider::Scale::Logarithmic);
    simulationSpeedSlider->setValue(1.0);

    return true;
}

bool Traffic::update()
{
    return connector.isConnected();
}

void Traffic::preFrame()
{
    previousTime = currentTime;
    currentTime = opencover::cover->frameTime();

    if (previousTime == 0.0)
    {
        return;
    }

    double deltaTime = currentTime - previousTime;

    const double simulationSpeed = pauseButton->state() ? 0.0 : simulationSpeedSlider->value();

    if (connector.update(deltaTime, deltaTime * simulationSpeed))
    {
        getSimulationResults(deltaTime);
        processNewResults();
    }

    interpolateVehiclePosition(deltaTime);
}

void Traffic::sendSimResults()
{
    // if (currentResults.size() != simResults.size() + pedestrianSimResults.size()) {
    //     currentResults.resize(simResults.size() + pedestrianSimResults.size());
    // }
    // size_t i = 0;
    //
    // for (std::map<std::string, libsumo::TraCIResults>::iterator it = simResults.begin(); it != simResults.end(); ++it) {
    //     std::shared_ptr<libsumo::TraCIPosition> position = std::dynamic_pointer_cast<libsumo::TraCIPosition>(it->second[VAR_POSITION3D]);
    //     currentResults[i].position = osg::Vec3d(position->x, position->y, position->z);
    //     std::shared_ptr<libsumo::TraCIDouble> angle = osg::DegreesToRadians(std::dynamic_pointer_cast<libsumo::TraCIDouble>(it->second[VAR_ANGLE]);
    //     currentResults[i].angle = angle->value;
    //     std::shared_ptr<libsumo::TraCIString> vehicleClass = std::dynamic_pointer_cast<libsumo::TraCIString>(it->second[VAR_VEHICLECLASS]);
    //     currentResults[i].vehicleClass = vehicleClass->value;
    //     std::shared_ptr<libsumo::TraCIString> vehicleType = std::dynamic_pointer_cast<libsumo::TraCIString>(it->second[VAR_TYPE]);
    //     currentResults[i].vehicleType = vehicleType->value;
    //     currentResults[i].vehicleID = it->first;
    //     i++;
    // }
    //
    // for (std::map<std::string, libsumo::TraCIResults>::iterator it = pedestrianSimResults.begin();
    //      it != pedestrianSimResults.end(); ++it) {
    //     std::shared_ptr<libsumo::TraCIPosition> position = std::dynamic_pointer_cast<libsumo::TraCIPosition>(it->second[VAR_POSITION3D]);
    //     currentResults[i].position = osg::Vec3d(position->x, position->y, position->z);
    //     std::shared_ptr<libsumo::TraCIDouble> angle = std::dynamic_pointer_cast<libsumo::TraCIDouble>(it->second[VAR_ANGLE]);
    //     currentResults[i].angle = angle->value;
    //     std::shared_ptr<libsumo::TraCIString> vehicleClass = std::dynamic_pointer_cast<libsumo::TraCIString>(it->second[VAR_VEHICLECLASS]);
    //     currentResults[i].vehicleClass = vehicleClass->value;
    //     std::shared_ptr<libsumo::TraCIString> vehicleType = std::dynamic_pointer_cast<libsumo::TraCIString>(it->second[VAR_TYPE]);
    //     currentResults[i].vehicleType = vehicleType->value;
    //     currentResults[i].vehicleID = it->first;
    //     i++;
    // }
    //
    // covise::TokenBuffer stb;
    // unsigned int currentSize = currentResults.size();
    // stb << currentSize;
    // for (size_t i = 0; i < currentResults.size(); i++) {
    //     double x = currentResults[i].position[0];
    //     double y = currentResults[i].position[1];
    //     double z = currentResults[i].position[2];
    //     stb << x;
    //     stb << y;
    //     stb << z;
    //     stb << currentResults[i].angle;
    //     stb << currentResults[i].vehicleClass;
    //     stb << currentResults[i].vehicleType;
    //     stb << currentResults[i].vehicleID;
    // }
    // unsigned int sizeInBytes = stb.getData().length();
    // opencover::coVRMSController::instance()->sendSlaves(&sizeInBytes, sizeof(sizeInBytes));
    // opencover::coVRMSController::instance()->sendSlaves(stb.getData().data(), sizeInBytes);
}

void Traffic::readSimResults()
{
    // unsigned int sizeInBytes = 0;
    // opencover::coVRMSController::instance()->readMaster(&sizeInBytes, sizeof(sizeInBytes));
    // char *buf = new char[sizeInBytes];
    // opencover::coVRMSController::instance()->readMaster(buf, sizeInBytes);
    // covise::TokenBuffer rtb((const char *)buf, sizeInBytes);
    // unsigned int currentSize;
    // rtb >> currentSize;
    // if (currentSize != currentResults.size()) {
    //     currentResults.resize(currentSize);
    // }
    //
    // for (int i = 0; i < currentResults.size(); i++) {
    //     double x;
    //     double y;
    //     double z;
    //     rtb >> x;
    //     rtb >> y;
    //     rtb >> z;
    //     currentResults[i].position[0] = x;
    //     currentResults[i].position[1] = y;
    //     currentResults[i].position[2] = z;
    //     rtb >> currentResults[i].angle;
    //     rtb >> currentResults[i].vehicleClass;
    //     rtb >> currentResults[i].vehicleType;
    //     rtb >> currentResults[i].vehicleID;
    // }
}

void Traffic::processNewResults()
{
    for (auto &vehicleState : currentSimulationState.vehicles)
    {
        auto currentEntity = vehicles.find(vehicleState.id);

        if (currentEntity == vehicles.end())
        { // not found, create new one
            if (vehicleClasses.find(vehicleState.vehicleClass) == vehicleClasses.end())
            {
                std::cerr << "[Traffic] found vehicle of class " << vehicleState.vehicleClass
                          << ", but no such vehicle class is loaded" << std::endl;
                continue;
            }

            const auto &vehicleClass = vehicleClasses.at(vehicleState.vehicleClass);

            if (!vehicleClass.models.size())
            {
                std::cerr << "[Traffic] vehicle class " << vehicleClass.name << " has no models" << std::endl;
                continue;
            }

            Vehicle &vehicle = createVehicle(vehicleState.id, vehicleClass);

            vehicle.position = vehicleState.position;
            vehicle.heading = vehicleState.angle;

            vehicle.targetPosition = vehicle.position;
            vehicle.targetHeading = vehicle.heading;
        }
        else
        {
            Vehicle &vehicle = currentEntity->second;
            vehicle.targetPosition = vehicleState.position;
            vehicle.targetHeading = vehicleState.angle;
        }
    }
}

/* utils for rotating angles the shortest way */
double angle_difference(double a, double b)
{
    double difference = fmod(b - a, M_PI_2);
    return fmod(2.0 * difference, M_PI_2) - difference;
}
double lerp_angle(double a, double b, double f)
{
    return a + angle_difference(a, b) * f;
}

void Traffic::interpolateVehiclePosition(double deltaTime)
{
    // float timeToDest = 1 - (currentTime - lastResultTime);

    // Drive a little slower so that if we have jitter in network
    // communication, we don't see that in the car movement.
    // TODO: Validate this methods works as intended.
    // timeToDest *= 0.9;

    // Loop over all entities and compute their new position
    for (auto &[_, vehicle] : vehicles)
    {
        // TODO: Interpolation instead of this immediate update
        vehicle.position = vehicle.targetPosition;
        vehicle.heading = vehicle.targetHeading;

        // osg::Vec3 velocity();
        // double angular_velocity = 0.0;
        //
        // if (timeToDest > 0.9999) {
        //     velocity = (vehicle.targetPosition - vehicle.position) * timeToDest;
        //     // TODO: check direction
        //     angular_velocity = angle_difference(vehicle.targetAngle, vehicle.angle) * timeToDest;
        // }
        // vehicle.position += velocity * deltaTime;
        // vehicle.angle = angular_velocity * deltaTime;

        // Update position of the geometry
        vehicle.geometry->setTransform(vehicle.position, vehicle.heading);
        vehicle.geometry->update(deltaTime);
    }
}

osg::Vec3d Traffic::interpolatePositions(double lambda, osg::Vec3d pastPosition, osg::Vec3d futurePosition)
{
    osg::Vec3d interpolatedPosition;
    for (int i = 0; i < 3; ++i)
    {
        double interpolatedPoint = futurePosition[i] + (1.0 - lambda) * (pastPosition[i] - futurePosition[i]);
        interpolatedPosition[i] = interpolatedPoint;
    }
    return interpolatedPosition;
}

double Traffic::interpolateAngles(double lambda, double pastAngle, double futureAngle)
{
    double interpolatedPosition = futureAngle + (1.0 - lambda) * (pastAngle - futureAngle);
    return interpolatedPosition;
}

Vehicle &Traffic::createVehicle(const std::string &id, const VehicleClass &vehicleClass)
{
    static std::mt19937 gen2(0);

    std::uniform_int_distribution<> dis(0, vehicleClass.models.size() - 1);
    int modelIndex = dis(gen2);
    auto &model = vehicleClass.models[modelIndex];

    osg::Group *parent = vehicleClassGroups.at(vehicleClass.name);

    std::unique_ptr<Geometry> geometry;

    switch (vehicleClass.geometryType)
    {
    case GeometryType::CHARACTER:
        geometry = std::make_unique<PedestrianGeometry>(id, model.fileName, model.scale, parent);
    // case GeometryType::STATIC:
    case GeometryType::CAR:
    default:
        geometry = std::make_unique<CarGeometry>(id, model.fileName, parent);
    }

    return vehicles.emplace(id, Vehicle { std::move(geometry) }).first->second;
}

void Traffic::loadVehicleClasses()
{
    vehicleClasses.clear();

    const auto &entries = covise::coCoviseConfig::getScopeEntries("COVER.Plugin.Traffic.VehicleClasses");

    for (auto &it : entries)
    {
        const std::string &vehicleClassName = it.first;

        auto &vehicleClass = vehicleClasses[vehicleClassName];
        vehicleClass.name = vehicleClassName;

        std::string geometryType = covise::coCoviseConfig::getEntry(
            "value", "COVER.Plugin.Traffic.VehicleClasses." + vehicleClassName + ".GeometryType", "car");
        boost::algorithm::to_lower(geometryType);

        if (geometryType == "static")
        {
            vehicleClass.geometryType = GeometryType::STATIC;
        }
        else if (geometryType == "car")
        {
            vehicleClass.geometryType = GeometryType::CAR;
        }
        else if (geometryType == "character")
        {
            vehicleClass.geometryType = GeometryType::CHARACTER;
        }
        else
        {
            std::cerr << "[Traffic] VehicleClass " << vehicleClassName << " has invalid GeometryType: " << geometryType << std::endl;
        }

        const auto &entries = covise::coCoviseConfig::getScopeEntries("COVER.Plugin.Traffic.VehicleClasses." + vehicleClassName + ".Models");

        for (auto &entry : entries)
        {
            // TODO: Character models are not scaled properly in the imported
            // file. They use the key as name and fileName, and the value for a
            // scaling factor. If we can, we should fix the scale in the model
            // itself and remove the extra scaling here.
            if (vehicleClass.geometryType == GeometryType::CHARACTER)
            {
                vehicleClass.models.push_back(VehicleModel { entry.first, entry.first, atof(entry.second.c_str()) });
            }
            else
            {
                vehicleClass.models.push_back(VehicleModel { entry.first, entry.second });
            }
        }

        // construct temporary geometry objects to preload models
        for (auto &model : vehicleClass.models)
        {
            switch (vehicleClass.geometryType)
            {
            case GeometryType::CHARACTER:
            {
                PedestrianGeometry geo("dummy", model.fileName, model.scale, nullptr);
            }

                // case GeometryType::STATIC:
            case GeometryType::CAR:
            default:
            {
                CarGeometry geo("dummy", model.fileName, nullptr);
            }
            }
        }
    }
}

COVERPLUGIN(Traffic)
