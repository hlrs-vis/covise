/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef OPENCOVER_PLUGINS_TRAFFIC_PEDESTRIANGEOMETRY_H
#define OPENCOVER_PLUGINS_TRAFFIC_PEDESTRIANGEOMETRY_H

#include <cover/coVRPluginSupport.h>
#include <osg/Group>
#include <osg/LOD>
#include <osg/MatrixTransform>
#include <osg/Transform>
#include <osgCal/CoreModel>
#include <osgCal/Model>

#include "Geometry.h"

constexpr double SPEED_IDLE = 0.0f;
constexpr double SPEED_SLOW = 0.6f;
constexpr double SPEED_WALK = 1.5f;
constexpr double SPEED_RUN = 3.0f;

// Generally avoid abrupt motion changes
constexpr double ANIMATION_BLEND_TIME = 0.1;

class VehicleModel;
class PedestrianGeometry : public Geometry
{
public:
    PedestrianGeometry(Vehicle &vehicle, osg::Group *parentNode);
    ~PedestrianGeometry();

    void update(double deltaTime, double simulationDeltaTime) override;
    void identifyAnimations(std::string_view filename);

protected:
    Vehicle &vehicle;

    osg::Vec3 previousPosition;
    double smoothedWalkSpeed = 0.0;

    static osg::ref_ptr<osgCal::CoreModel> loadFile(const std::string &file);

    void removeFromSceneGraph();
    void setWalkingSpeed(double speed);

    bool activeState = true;

    std::string geometryName;

    osg::ref_ptr<osgCal::CoreModel> coreModel;
    osg::ref_ptr<osg::Group> groupNode;
    osg::ref_ptr<osg::MatrixTransform> transformNode;
    osg::ref_ptr<osg::LOD> lodNode;

    osg::ref_ptr<osgCal::Model> model;
    osg::ref_ptr<osgCal::Model> staticModel;

    int animationIndexIdle = -1;
    int animationIndexSlow = -1;
    int animationIndexWalk = -1;
    int animationIndexRun = -1;
    int animationIndexLook = -1;
    int animationIndexWave = -1;
};

#endif
