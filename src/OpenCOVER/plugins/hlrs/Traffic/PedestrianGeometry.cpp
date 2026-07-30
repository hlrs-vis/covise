/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "PedestrianGeometry.h"

#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <cover/VRSceneGraph.h>
#include <osg/Vec3>
#include <osgDB/FileNameUtils>

#include "traffic_utils.h"
#include "Traffic.h"

using namespace opencover;

constexpr double LOD_RANGE = 500;
constexpr double LOD_RANGE_ANIMATED = 150;

osg::ref_ptr<osgCal::CoreModel> PedestrianGeometry::loadFile(const std::string &file)
{
    // TODO: resolve and normalize path for key lookup?

    static std::map<std::string, osg::ref_ptr<osgCal::CoreModel>> cache;
    static osg::ref_ptr<osg::Group> dummyParent = new osg::Group();

    if (cache.find(file) == cache.end())
    {
        // Load core model
        osg::ref_ptr<osgCal::CoreModel> coreModel = new osgCal::CoreModel();
        osg::ref_ptr<osgCal::MeshParameters> meshParams = new osgCal::MeshParameters;
        meshParams->useDepthFirstMesh = false;
        meshParams->software = false;

        try
        {
            coreModel->load(file, meshParams.get());
        }
        catch (std::exception &e)
        {
            std::cerr << "PedestrianGeometry::loadFile(" << file << "): exception during load:" << std::endl
                      << e.what() << std::endl;
            return nullptr;
        }

        cache[file] = coreModel;
    }

    return cache[file];
}

osg::ref_ptr<osg::Node> addScaleNode(osg::ref_ptr<osg::Node> node, double scale)
{
    auto scaleNode = new osg::MatrixTransform();
    scaleNode->setName("scalePedestrian");
    scaleNode->setMatrix(osg::Matrix::scale(scale, scale, scale));
    scaleNode->addChild(node);
    return scaleNode;
}

/**
 * Construct a new pedestrian geometry object
 */
PedestrianGeometry::PedestrianGeometry(Vehicle &vehicle, osg::Group *parentNode)
    : vehicle(vehicle)
{
    // Create a new transform, add it to the group
    transformNode = new osg::MatrixTransform();
    transformNode->setName(vehicle.id);
    if (parentNode)
    {
        parentNode->addChild(transformNode);
    }

    // Create a new LOD, set range, and add it to the transform
    lodNode = new osg::LOD();
    transformNode->addChild(lodNode);

    auto coreModel = loadFile(vehicle.model->path);
    if (coreModel)
    {
        auto meshAdder = new osgCal::DefaultMeshAdder();
        model = new osgCal::Model();
        model->load(coreModel, meshAdder);
        model->setNodeMask(model->getNodeMask() & ~Isect::Update); // we update ourselves

        meshAdder = new osgCal::DefaultMeshAdder();
        staticModel = new osgCal::Model();
        staticModel->load(coreModel, meshAdder);
        staticModel->blendCycle(ANIMATION_INDEX_IDLE, 1.0, 0.0, 0.0);

        // TODO: maybe randomize lod range a little to "fade" crowds in?
        lodNode->addChild(addScaleNode(model, vehicle.model->scale), 0.0, LOD_RANGE_ANIMATED);
        lodNode->addChild(addScaleNode(staticModel, vehicle.model->scale), LOD_RANGE_ANIMATED, LOD_RANGE);
    }

    transformNode->setNodeMask(transformNode->getNodeMask() & ~(Isect::Intersection | Isect::Collision | Isect::Walk));
}

PedestrianGeometry::~PedestrianGeometry()
{
    removeFromSceneGraph();
}

void PedestrianGeometry::removeFromSceneGraph()
{
    while (transformNode->getNumParents() > 0)
    {
        transformNode->getParent(0)->removeChild(transformNode);
    }
}

void PedestrianGeometry::update(double deltaTime, double simulationDeltaTime)
{
    vehicle.timeSinceSource += simulationDeltaTime;
    float t = std::clamp(vehicle.timeSinceSource / vehicle.timeFromSourceToTarget, 0.0, 1.05);

    auto position = lerp(vehicle.sourcePosition, vehicle.targetPosition, t);

    auto difference = previousPosition - position;
    double distance = difference.length();
    auto heading = distance > 0 ? atan2(difference.y(), difference.x()) : vehicle.heading;
    vehicle.heading = lerp_angle(vehicle.heading, heading, deltaTime * 3.f);
    previousPosition = position;

    auto matrix = osg::Matrix::rotate(vehicle.heading - M_PI_2, osg::Vec3d(0, 0, 1)) * osg::Matrix::translate(position);
    transformNode->setMatrix(matrix);

    if (simulationDeltaTime > 0)
    {
        double speed = distance / simulationDeltaTime;
        smoothedWalkSpeed = std::lerp(smoothedWalkSpeed, speed, deltaTime * 2.0);
    }
    setWalkingSpeed(smoothedWalkSpeed);

    model->update(simulationDeltaTime);
}

#define UC(a, b) std::clamp(unlerp(a, b, speed), 0.0, 1.0)

/**
 * Adjust the geometry's animation settings to match the given speed, according to the geometry's animation mapping
 */
void PedestrianGeometry::setWalkingSpeed(double speed)
{
    // Only allow positive speeds
    speed = std::abs(speed);

    // Compute which movement animation we're in to which amount
    double idleAmount = UC(SPEED_SLOW, SPEED_IDLE);
    double slowAmount = std::min(UC(SPEED_IDLE, SPEED_SLOW), UC(SPEED_WALK, SPEED_SLOW));
    double walkAmount = std::min(UC(SPEED_WALK, SPEED_SLOW), UC(SPEED_WALK, SPEED_RUN));
    double runAmount = UC(SPEED_RUN, SPEED_WALK);

    model->blendCycle(ANIMATION_INDEX_IDLE, idleAmount, ANIMATION_BLEND_TIME);
    model->blendCycle(ANIMATION_INDEX_SLOW, slowAmount, ANIMATION_BLEND_TIME, speed / SPEED_SLOW);
    model->blendCycle(ANIMATION_INDEX_WALK, walkAmount, ANIMATION_BLEND_TIME, speed / SPEED_WALK);
    model->blendCycle(ANIMATION_INDEX_RUN, runAmount, ANIMATION_BLEND_TIME, speed / SPEED_RUN);

    // TODO: can we let bicycles roll out without pedaling when they are slowing down?
}
