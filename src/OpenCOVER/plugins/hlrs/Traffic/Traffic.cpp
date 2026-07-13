/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Traffic.h"

#include <cmath>
#include <cover/VRSceneGraph.h>
#include <iostream>
#include <random>

#include <PluginUtil/PluginMessageTypes.h>
#include <boost/algorithm/string.hpp>
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
    double deltaTime = opencover::cover->frameDuration();

    if (deltaTime <= 0.0)
    {
        return;
    }

    const double simulationSpeed = pauseButton->state() ? 0.0 : simulationSpeedSlider->value();

    if (connector.update(deltaTime, deltaTime * simulationSpeed))
    {
        getSimulationResults(deltaTime);
        processNewResults();
    }

    interpolateVehiclePosition(deltaTime * simulationSpeed);
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

osg::Vec2 toVec2(osg::Vec3 v) { return osg::Vec2(v.x(), v.y()); }
float distanceRatio(osg::Vec2 x, osg::Vec2 y)
{
    return (x * y) / y.length2();
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

            vehicle.targetPosition = vehicleState.position;
            vehicle.targetHeading = vehicleState.angle;
            vehicle.targetSpeed = vehicleState.speed;

            vehicle.sourcePosition = vehicleState.position;
            vehicle.sourceHeading = vehicleState.angle;
            vehicle.sourceSpeed = vehicleState.speed;

            vehicle.timeFromSourceToTarget = 1.0;
            vehicle.timeSinceSource = 0.0;
        }
        else
        {
            Vehicle &vehicle = currentEntity->second;

            vehicle.sourcePosition = vehicle.targetPosition;
            vehicle.sourceHeading = vehicle.targetHeading;
            vehicle.sourceSpeed = vehicle.targetSpeed;

            vehicle.targetPosition = vehicleState.position;
            vehicle.targetHeading = vehicleState.angle;
            vehicle.targetSpeed = vehicleState.speed;

            vehicle.timeSinceSource = 0.0;
            vehicle.timeFromSourceToTarget = 0.2; // TODO: use simulation dt here

            // Compute bezier points
            osg::Vec3 forward0(cos(vehicle.sourceHeading), sin(vehicle.sourceHeading), 0);
            vehicle.p0 = vehicle.sourcePosition;
            vehicle.p1 = vehicle.p0 + forward0 * vehicle.sourceSpeed * 0.333 * vehicle.timeFromSourceToTarget;

            osg::Vec3 forward3(cos(vehicle.targetHeading), sin(vehicle.targetHeading), 0);
            vehicle.p3 = vehicle.targetPosition;
            vehicle.p2 = vehicle.p3 - forward3 * vehicle.targetSpeed * 0.333 * vehicle.timeFromSourceToTarget;

            // Fix interpolation points z height
            vehicle.p1.z() = lerp(vehicle.p0.z(), vehicle.p3.z(),
                distanceRatio(toVec2(vehicle.p1 - vehicle.p0), toVec2(vehicle.p3 - vehicle.p0)));

            vehicle.p2.z() = lerp(vehicle.p0.z(), vehicle.p3.z(),
                distanceRatio(toVec2(vehicle.p2 - vehicle.p0), toVec2(vehicle.p3 - vehicle.p0)));

#ifdef TRAFFIC_DEBUG_DRAW
            vehicle.points[0]->setMatrix(osg::Matrix::translate(vehicle.p0));
            vehicle.points[1]->setMatrix(osg::Matrix::translate(vehicle.p1));
            vehicle.points[2]->setMatrix(osg::Matrix::translate(vehicle.p2));
            vehicle.points[3]->setMatrix(osg::Matrix::translate(vehicle.p3));
#endif
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

template <typename T>
T cubic_bezier(T p0, T p1, T p2, T p3, float t)
{
    float t1 = 1.f - t;
    return p0 * (t1 * t1 * t1) + p1 * (3 * t1 * t1 * t) + p2 * (3 * t1 * t * t) + p3 * (t * t * t);
}

template <typename T>
T cubic_bezier_tangent(T p0, T p1, T p2, T p3, float t)
{
    float t1 = 1.f - t;
    return p0 * (-3 * t1 * t1) + p1 * (3 * t1 * t1) - p1 * (6 * t * t1) - p2 * (3 * t * t) + p2 * (6 * t * t1) + p3 * (3 * t * t);
}

void Traffic::interpolateVehiclePosition(double deltaTime)
{
    // Loop over all entities and compute their new position
    for (auto &[_, vehicle] : vehicles)
    {
        vehicle.timeSinceSource += deltaTime;

        float t = std::clamp(vehicle.timeSinceSource / vehicle.timeFromSourceToTarget, 0.0, 1.05);

        vehicle.position = cubic_bezier(vehicle.p0, vehicle.p1, vehicle.p2, vehicle.p3, t);
        auto tan = cubic_bezier_tangent(vehicle.p0, vehicle.p1, vehicle.p2, vehicle.p3, t);
        vehicle.heading = lerp_angle(vehicle.sourceHeading, vehicle.targetHeading, t);
        // vehicle.heading = atan2(tan.y(), tan.x());
        vehicle.pitch = -asin(tan.z() / tan.length());
        vehicle.speed = tan.length() / vehicle.timeFromSourceToTarget;

        // Update position of the geometry
        vehicle.geometry->setTransform(vehicle.position, vehicle.heading, vehicle.pitch);
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

osg::ref_ptr<osg::Geode> makeSphere(osg::Vec4 color)
{
    osg::Sphere *mySphere = new osg::Sphere(osg::Vec3(0, 0, 0), 0.3);
    osg::TessellationHints *hint = new osg::TessellationHints();
    hint->setDetailRatio(0.5);
    osg::ShapeDrawable *mySphereDrawable = new osg::ShapeDrawable(mySphere, hint);
    mySphereDrawable->setColor(color);
    auto geode = new osg::Geode();
    geode->addDrawable(mySphereDrawable);
    geode->setStateSet(opencover::VRSceneGraph::instance()->loadDefaultGeostate(osg::Material::AMBIENT_AND_DIFFUSE));
    return geode;
}

osg::Vec4 colors[] = {
    { 1, 0, 0, 1 },
    { 1, 1, 0, 1 },
    { 0, 1, 0, 1 },
    { 0, 1, 1, 1 },
};

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
        geometry = std::make_unique<PedestrianGeometry>(id, model, parent);
        break;

    // case GeometryType::STATIC:
    case GeometryType::VEHICLE:
    default:
        geometry = std::make_unique<CarGeometry>(id, model, parent);
        break;
    }

    auto &v = vehicles.emplace(id, Vehicle { std::move(geometry), &model }).first->second;

#ifdef TRAFFIC_DEBUG_DRAW
    for (int i = 0; i < 4; i++)
    {
        osg::ref_ptr<osg::MatrixTransform> t = new osg::MatrixTransform;
        t->addChild(makeSphere(colors[i]));
        opencover::cover->getObjectsRoot()->addChild(t);
        v.points[i] = t;
    }
#endif

    return v;
}

void Traffic::loadVehicleClasses()
{
    vehicleClasses.clear();

    auto configFile = config();

    auto vehicleClassesSection = configFile->value<opencover::config::Section>("", "vehicleClasses")->value();
    auto vehicleClassNames = vehicleClassesSection.sections();

    for (auto sectionName : vehicleClassNames)
    {
        std::string vehicleClassName = sectionName;
        vehicleClassName.replace(0, 15, "");

        opencover::config::Section entry = vehicleClassesSection.value<opencover::config::Section>("", vehicleClassName)->value();

        auto &vehicleClass = vehicleClasses[vehicleClassName];
        vehicleClass.name = vehicleClassName;

        std::string geometryType = entry.value<std::string>("", "type", "vehicle")->value();
        boost::algorithm::to_lower(geometryType);

        if (geometryType == "static")
        {
            vehicleClass.geometryType = GeometryType::STATIC;
        }
        else if (geometryType == "vehicle")
        {
            vehicleClass.geometryType = GeometryType::VEHICLE;
        }
        else if (geometryType == "character")
        {
            vehicleClass.geometryType = GeometryType::CHARACTER;
        }
        else
        {
            std::cerr << "[Traffic] VehicleClass " << vehicleClassName << " has invalid GeometryType: " << geometryType << std::endl;
        }

        auto modelSections = entry.array<opencover::config::Section>("", "models")->value();

        for (auto modelSection : modelSections)
        {
            vehicleClass.models.push_back(VehicleModel {
                modelSection.value<std::string>("", "name")->value(),
                modelSection.value<std::string>("", "path")->value(),
                modelSection.value<double>("", "scale", 1.0)->value(),
                modelSection.value<double>("", "front_axis", 0.0)->value(),
                modelSection.value<double>("", "back_axis", 0.0)->value(),
                modelSection.value<double>("", "length", 0.0)->value(),
            });
        }

        // construct temporary geometry objects to preload models
        for (auto &model : vehicleClass.models)
        {
            switch (vehicleClass.geometryType)
            {
            case GeometryType::CHARACTER:
            {
                PedestrianGeometry geo("dummy", model, nullptr);
                break;
            }

                // case GeometryType::STATIC:
            case GeometryType::VEHICLE:
            default:
            {
                CarGeometry geo("dummy", model, nullptr);
                break;
            }
            }
        }
    }
}

COVERPLUGIN(Traffic)
