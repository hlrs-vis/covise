/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "PedestrianGeometry.h"

#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <osg/Vec3>

using namespace opencover;

inline double lerp(double a, double b, double f)
{
    return a + (b - a) * f;
}
inline double unlerp(double a, double b, double f)
{
    return (f - a) / (b - a);
}

osg::ref_ptr<osgCal::CoreModel> PedestrianGeometry::loadFile(const std::string &file)
{
    static std::map<std::string, osg::ref_ptr<osgCal::CoreModel>> cache;
    static osg::ref_ptr<osg::Group> dummyParent = new osg::Group();

    if (cache.find(file) == cache.end())
    {
        std::string modelFile = (boost::format { "/data/cal3d/%s/%s.cfg" } % file % file).str();

        // Load core model
        osg::ref_ptr<osgCal::CoreModel> model = new osgCal::CoreModel();
        osg::ref_ptr<osgCal::MeshParameters> meshParams = new osgCal::MeshParameters;
        meshParams->useDepthFirstMesh = false;
        meshParams->software = false;
        try
        {
            model->load(modelFile, meshParams.get());
        }
        catch (std::exception &e)
        {
            std::cerr << "CarGeometry::loadFile(" << modelFile << "): exception during load:" << std::endl
                      << e.what() << std::endl;
            return nullptr;
        }

        cache[file] = model;
    }

    return cache[file];
}

/**
 * Construct a new pedestrian geometry object
 */
PedestrianGeometry::PedestrianGeometry(const std::string &name, const std::string &fileName, double scale, osg::Group *parentNode)
{
    // Create a new transform, add it to the group
    transformNode = new osg::MatrixTransform();
    transformNode->setName(name);
    if (parentNode)
    {
        parentNode->addChild(transformNode);
    }

    // Create a new LOD, set range, and add it to the transform
    lodNode = new osg::LOD();
    transformNode->addChild(lodNode);
    lodNode->setRange(0, 0.0, 500.0); // TODO: maybe randomize lod range a little to "fade" crowds in?

    // Create a new transform only for scaling the rendered model
    scaleNode = new osg::MatrixTransform();
    scaleNode->setName("scalePedestrian");
    osg::Matrix scaleMatrix;
    scaleMatrix.makeScale(scale, scale, scale);
    scaleNode->setMatrix(scaleMatrix);
    lodNode->addChild(scaleNode);

    // Create a new instance of the core model, add it to the LOD
    model = new osgCal::Model();
    meshAdder = new osgCal::DefaultMeshAdder;
    osg::ref_ptr<osgCal::CoreModel> coreModel = loadFile(fileName);
    if (coreModel)
    {
        model->load(coreModel, meshAdder);
        scaleNode->addChild(model);
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

void PedestrianGeometry::update(double deltaTime)
{
    osg::Vec3 position = transformNode->getMatrix().getTrans();
    double distance = (previousPosition - position).length();
    previousPosition = position;

    if (deltaTime > 0)
    {
        double speed = distance / deltaTime;
        smoothedWalkSpeed = lerp(smoothedWalkSpeed, speed, deltaTime * 5.0);
        setWalkingSpeed(smoothedWalkSpeed);
    }

    model->update(deltaTime);
}

/**
 * Adjust the geometry's animation settings to match the given speed, according to the geometry's animation mapping
 */
void PedestrianGeometry::setWalkingSpeed(double speed)
{
    // Only allow positive speeds
    speed = std::abs(speed);

    // Compute which movement animation we're in to which amount
    double idleAmount = std::clamp(unlerp(SPEED_SLOW, SPEED_IDLE, speed), 0.0, 1.0);
    double slowAmount = std::clamp(unlerp(SPEED_IDLE, SPEED_SLOW, speed), 0.0, 1.0) * std::clamp(unlerp(SPEED_SLOW, SPEED_WALK, speed), 0.0, 1.0);
    double walkAmount = std::clamp(unlerp(SPEED_SLOW, SPEED_WALK, speed), 0.0, 1.0) * std::clamp(unlerp(SPEED_WALK, SPEED_RUN, speed), 0.0, 1.0);
    double runAmount = std::clamp(unlerp(SPEED_WALK, SPEED_WALK, speed), 0.0, 1.0);

    model->blendCycle(ANIMATION_INDEX_IDLE, idleAmount, ANIMATION_BLEND_TIME);
    model->blendCycle(ANIMATION_INDEX_SLOW, slowAmount, ANIMATION_BLEND_TIME);
    model->blendCycle(ANIMATION_INDEX_WALK, walkAmount, ANIMATION_BLEND_TIME);
    model->blendCycle(ANIMATION_INDEX_RUN, runAmount, ANIMATION_BLEND_TIME);

    // Mix the time scales for each movement animation by the weights -- this
    // might be stupid, let's check :)
    double slowTimeScale = speed / SPEED_SLOW;
    double walkTimeScale = speed / SPEED_WALK;
    double runTimeScale = speed / SPEED_RUN;
    double timeScale = 1.0 * idleAmount + slowTimeScale * slowAmount + walkTimeScale * walkAmount + runTimeScale * runAmount;
    model->setTimeFactor(timeScale);

    // TODO: What special handling do bicycles need?
    // if (pedGroup->getName().find("bicycle") != std::string::npos) {
    //     if (speed <= anim.slowVel) {
    //         model->clearCycle(0, ANIMATION_BLEND_TIME );
    //     } else {
    //         model->blendCycle(0, 2*speed, ANIMATION_BLEND_TIME );
    //         if (!PedestrianUtils::floatEq(timeFactorScale, 1.0)) {
    //             timeFactorScale = 1.0;
    //             model->setTimeFactor(timeFactorScale);
    //         }
    //     }
    // }
    //
}

void PedestrianGeometry::setTransform(osg::Matrix transform)
{
    transformNode->setMatrix(transform);
}

/**
 * Perform the one-time action to "look both ways before crossing the street"
 * (i.e., turn head left and right)
 */
void PedestrianGeometry::executeLook(double factor)
{
    model->executeAction(ANIMATION_INDEX_LOOK, factor);
}

/**
 * Perform the one-time action to "wave" (e.g., when approaching a fellow pedestrian)
 */
void PedestrianGeometry::executeWave(double factor)
{
    model->executeAction(ANIMATION_INDEX_WAVE, factor);
}

/**
 * Perform a one-time action, where idx is its position in the Cal3d .cfg
 * file's animation list (first item has idx 0)
 */
void PedestrianGeometry::executeAction(int idx, double factor)
{
    model->executeAction(idx, 0.0, 0.0, 1.0, false, factor);
}
