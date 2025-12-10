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

constexpr int ANIMATION_INDEX_IDLE = 0;
constexpr int ANIMATION_INDEX_SLOW = 1;
constexpr int ANIMATION_INDEX_WALK = 2;
constexpr int ANIMATION_INDEX_RUN = 3;
constexpr int ANIMATION_INDEX_LOOK = 4;
constexpr int ANIMATION_INDEX_WAVE = 5;

constexpr double SPEED_IDLE = 0.0f;
constexpr double SPEED_SLOW = 0.6f;
constexpr double SPEED_WALK = 1.5f;
constexpr double SPEED_RUN = 3.0f;

// Generally avoid abrupt motion changes
constexpr double ANIMATION_BLEND_TIME = 0.1;

class PedestrianGeometry : public Geometry
{
public:
    PedestrianGeometry(const std::string &name, const std::string &fileName, double scale, osg::Group *parentNode);
    ~PedestrianGeometry();

    void setTransform(osg::Matrix transform);

    void update(double deltaTime);

protected:
    osg::Vec3 previousPosition;
    double smoothedWalkSpeed = 0.0;

    static osg::ref_ptr<osgCal::CoreModel> loadFile(const std::string &file);

    void removeFromSceneGraph();
    void setWalkingSpeed(double speed);
    void executeLook(double factor = 1.0);
    void executeWave(double factor = 1.0);
    void executeAction(int idx, double factor = 1.0);

    bool activeState = true;

    std::string geometryName;

    osg::ref_ptr<osg::Group> groupNode;
    osg::ref_ptr<osg::MatrixTransform> transformNode;
    osg::ref_ptr<osg::LOD> lodNode;
    osg::ref_ptr<osg::MatrixTransform> scaleNode;

    osg::ref_ptr<osgCal::Model> model;
    osg::ref_ptr<osgCal::BasicMeshAdder> meshAdder;
};

#endif
