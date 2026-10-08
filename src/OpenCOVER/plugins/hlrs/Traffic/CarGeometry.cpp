/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "CarGeometry.h"

#include <cover/coVRFileManager.h>
#include <cover/coVRMSController.h>
#include <cover/coVRPluginSupport.h>

#include <boost/algorithm/string/replace.hpp>
#include <linux/stat.h>
#include <osg/CopyOp>
#include <osg/Material>
#include <sys/stat.h>
#include <random>

#include <osg/Group>
#include <osg/LOD>
#include <osg/MatrixTransform>
#include <osg/Node>
#include <osg/Switch>
#include <osg/Transform>
#include <osgDB/ReadFile>

#include "Traffic.h"
#include "traffic_utils.h"

// #include <math.h>
#define PI 3.141592653589793238

inline bool fileExists(const std::string &name)
{
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}

osg::Node *CarGeometry::loadFile(const std::string &file)
{
    if (!fileExists(file))
    {
        std::cerr << "CarGeometry::getCarNode(): File not YET found: " << file << "..." << std::endl;
        return nullptr;
    }

    static std::map<std::string, osg::Node *> cache;

    // Create a dummy group so the loaded content isn't added to the real root node.
    static osg::ref_ptr<osg::Group> dummyParent = new osg::Group();

    if (cache.find(file) == cache.end())
    {
        osg::Node *node = opencover::coVRFileManager::instance()->loadFile(file.c_str(), nullptr, dummyParent, nullptr, true);
        if (!node)
        {
            return nullptr;
        }

        // Create a safe(r) node name from the filename
        std::string name(file);
        boost::algorithm::replace_all(name, "/", "_");
        boost::algorithm::replace_all(name, ".", "_");
        boost::algorithm::replace_all(name, "\\", "_");
        node->setName(name);

        cache[file] = node;
    }

    return cache.at(file);
}

osg::Node *applyBodyColor(osg::Node *node, osg::Vec4 color)
{
    osg::Node *n = node;
    while (n->asGroup() && !n->asGeode())
    {
        if (n->asGroup()->getNumChildren() == 0)
        {
            return node;
        }
        n = n->asGroup()->getChild(0);
    }

    if (!n->asGeode())
    {
        return node;
    }

    if (n->asGeode()->getNumDrawables() == 0)
    {
        return node;
    }

    node = node->clone(osg::CopyOp::DEEP_COPY_ALL)->asNode();
    n = node;

    while (n->asGroup() && !n->asGeode())
        n = n->asGroup()->getChild(0);

    auto geode = n->asGeode();

    osg::ref_ptr<osg::Material> mat = new osg::Material;
    mat->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
    mat->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(0.2f, 0.2f, 0.2f, 1.0)); // where do I get ambient from?
    mat->setDiffuse(osg::Material::FRONT_AND_BACK, color);
    mat->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(0.9f, 0.9f, 0.9f, 1.0));
    mat->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4(0.0f, 0.0f, 0.0f, 1.0));
    mat->setShininess(osg::Material::FRONT_AND_BACK, 16.0f);
    // for (int i = 0; i < geode->getNumDrawables(); i++)
    // {
    // }
    geode->getDrawable(0)->getOrCreateStateSet()->setAttributeAndModes(mat, osg::StateAttribute::ON);

    return node;
}

CarGeometry::CarGeometry(Vehicle &vehicle, osg::Group *parentNode)
    : vehicle(vehicle)
{
    transformNode = new osg::MatrixTransform();
    transformNode->setName(vehicle.id);
    transformNode->setNodeMask(transformNode->getNodeMask() & ~(opencover::Isect::Intersection | opencover::Isect::Collision | opencover::Isect::Walk));

    if (parentNode)
    {
        parentNode->addChild(transformNode);
    }

    lodNode = new osg::LOD();
    transformNode->addChild(lodNode);

    static std::mt19937 gen2(0);
    // https://www.kba.de/DE/Statistik/Fahrzeuge/Neuzulassungen/Farbe/2023/2023_n_farbe_zeitreihe.html?nn=837402&fromStatistic=837402&yearFilter=2023&fromStatistic=837402&yearFilter=2023
    static std::discrete_distribution<> dis({ 589, 146, 200, 73, 73, 500, 500, 754 });
    static osg::Vec4 colors[] = {
        osg::Vec4(1, 1, 1, 1), // white
        osg::Vec4(0.8, 0, 0, 1), // red
        osg::Vec4(0, 0.1, 0.7, 1), // blue
        osg::Vec4(0, 0.7, 0.9, 1), // cyan
        osg::Vec4(0, 0.7, 0, 1), // green
        osg::Vec4(0.4, 0.4, 0.4, 1), // gray
        osg::Vec4(0.8, 0.8, 0.84, 1), // silver
        osg::Vec4(0, 0, 0, 1), // black
    };

    auto color = colors[dis(gen2)];
    osg::Node *modelNode = applyBodyColor(loadFile(vehicle.model->path), color);
    lodNode->addChild(modelNode, 0, 250.0);

    osg::Node *lodCar = applyBodyColor(loadFile("/data/traffic/cars/lod_car.glb"), color);
    lodNode->addChild(lodCar, 250.0, 1000.0);
}

void CarGeometry::updateTrajectory()
{
    auto diff = vehicle.targetPosition - vehicle.sourcePosition;
    auto diffSpeed = diff.length() / vehicle.timeFromSourceToTarget;

    bool notMoving = diffSpeed < 0.01;
    if (notMoving)
    {
        stationaryTimer += vehicle.timeFromSourceToTarget;
    }

    if (diffSpeed > vehicle.sourceSpeed + vehicle.targetSpeed || (stationary && notMoving) || (notMoving && stationaryTimer > 20.0))
    {
        stationary = true;
    }
    else
    {
        stationary = false;

        // Compute bezier points
        osg::Vec3 forward0(cos(vehicle.sourceHeading), sin(vehicle.sourceHeading), 0);
        p0 = vehicle.sourcePosition;
        p1 = p0 + forward0 * vehicle.sourceSpeed * 0.333 * vehicle.timeFromSourceToTarget;

        osg::Vec3 forward3(cos(vehicle.targetHeading), sin(vehicle.targetHeading), 0);
        p3 = vehicle.targetPosition;
        p2 = p3 - forward3 * vehicle.targetSpeed * 0.333 * vehicle.timeFromSourceToTarget;

        // Fix interpolation points z height
        p1.z() = std::lerp(p0.z(), p3.z(), distanceRatio(toVec2(p1 - p0), toVec2(p3 - p0)));
        p2.z() = std::lerp(p0.z(), p3.z(), distanceRatio(toVec2(p2 - p0), toVec2(p3 - p0)));
    }
}

void CarGeometry::update(double deltaTime, double simulationDeltaTime)
{
    vehicle.timeSinceSource += deltaTime;

    if (stationary)
    {
        vehicle.position = vehicle.targetPosition;
        vehicle.heading = vehicle.targetHeading;
        vehicle.speed = vehicle.targetSpeed;
        vehicle.pitch = 0.0;
    }
    else
    {
        float t = std::clamp(vehicle.timeSinceSource / vehicle.timeFromSourceToTarget, 0.0, 1.05);

        vehicle.position = vehicle.targetPosition;
        vehicle.position = cubic_bezier(p0, p1, p2, p3, t);
        auto tan = cubic_bezier_tangent(p0, p1, p2, p3, t);
        // vehicle.heading = lerp_angle(vehicle.sourceHeading, vehicle.targetHeading, t);
        // vehicle.heading = atan2(tan.y(), tan.x());
        // vehicle.pitch = -asin(tan.z() / tan.length());
        // vehicle.speed = tan.length() / vehicle.timeFromSourceToTarget;

        // Let the back axle follow the position
        double backAxleLength = abs(vehicle.model->backAxle - vehicle.model->frontAxle);
        osg::Vec3 backAxleDir = backAxle - vehicle.position;
        backAxleDir.normalize();
        backAxle = vehicle.position + backAxleDir * backAxleLength;

        vehicle.heading = atan2(-backAxleDir.y(), -backAxleDir.x());
        vehicle.pitch = asin(backAxleDir.z() / backAxleDir.length());
    }

    // TODO: actually rotate from axle placement
    auto matrix = (osg::Matrix::translate(osg::Vec3d(-vehicle.model->frontAxle, 0, 0))
        * osg::Matrix::rotate(vehicle.pitch, osg::Vec3d(0, 1, 0))
        * osg::Matrix::rotate(vehicle.heading, osg::Vec3d(0, 0, 1))
        * osg::Matrix::translate(vehicle.position));
    transformNode->setMatrix(matrix);
}

CarGeometry::~CarGeometry()
{
    removeFromSceneGraph();
}

void CarGeometry::removeFromSceneGraph()
{
    while (transformNode->getNumParents() > 0)
    {
        transformNode->getParent(0)->removeChild(transformNode);
    }
}
