/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "CarGeometry.h"

#include <cover/coVRFileManager.h>
#include <cover/coVRMSController.h>
#include <cover/coVRPluginSupport.h>

#include <boost/algorithm/string/replace.hpp>
#include <sys/stat.h>

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
        osg::Node *node = opencover::coVRFileManager::instance()->loadFile(file.c_str(), nullptr, dummyParent);
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

    osg::Node *modelNode = loadFile(vehicle.model->path);
    lodNode->addChild(modelNode, 0, 1000.0);
}

void CarGeometry::updateTrajectory()
{
    // Compute bezier points
    osg::Vec3 forward0(cos(vehicle.sourceHeading), sin(vehicle.sourceHeading), 0);
    p0 = vehicle.sourcePosition;
    p1 = p0 + forward0 * vehicle.sourceSpeed * 0.333 * vehicle.timeFromSourceToTarget;

    osg::Vec3 forward3(cos(vehicle.targetHeading), sin(vehicle.targetHeading), 0);
    p3 = vehicle.targetPosition;
    p2 = p3 - forward3 * vehicle.targetSpeed * 0.333 * vehicle.timeFromSourceToTarget;

    // Fix interpolation points z height
    p1.z() = lerp(p0.z(), p3.z(), distanceRatio(toVec2(p1 - p0), toVec2(p3 - p0)));
    p2.z() = lerp(p0.z(), p3.z(), distanceRatio(toVec2(p2 - p0), toVec2(p3 - p0)));
}

void CarGeometry::update(double deltaTime)
{
    vehicle.timeSinceSource += deltaTime;

    float t = std::clamp(vehicle.timeSinceSource / vehicle.timeFromSourceToTarget, 0.0, 1.05);

    vehicle.position = cubic_bezier(p0, p1, p2, p3, t);
    auto tan = cubic_bezier_tangent(p0, p1, p2, p3, t);
    // vehicle.heading = lerp_angle(vehicle.sourceHeading, vehicle.targetHeading, t);
    // vehicle.heading = atan2(tan.y(), tan.x());
    // vehicle.pitch = -asin(tan.z() / tan.length());
    // vehicle.speed = tan.length() / vehicle.timeFromSourceToTarget;

    // Let the back axle follow the position
    double backAxleLength = abs(vehicle.model->back_axis - vehicle.model->front_axis);
    osg::Vec3 backAxleDir = backAxle - vehicle.position;
    backAxleDir.normalize();
    backAxle = vehicle.position + backAxleDir * backAxleLength;

    vehicle.heading = atan2(-backAxleDir.y(), -backAxleDir.x());
    vehicle.pitch = asin(backAxleDir.z() / backAxleDir.length());

    // TODO: actually rotate from axle placement
    auto matrix = (osg::Matrix::translate(osg::Vec3d(-vehicle.model->front_axis, 0, 0))
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
