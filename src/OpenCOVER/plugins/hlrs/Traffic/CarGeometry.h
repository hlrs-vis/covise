/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef OPENCOVER_PLUGINS_TRAFFIC_CARGEOMETRY_H
#define OPENCOVER_PLUGINS_TRAFFIC_CARGEOMETRY_H

#include <osg/BoundingBox>
#include <osg/MatrixTransform>
#include <osg/Node>
#include <vector>

#include "Geometry.h"

class VehicleModel;
class Vehicle;
class CarGeometry : public Geometry
{
public:
    CarGeometry(Vehicle &vehicle, osg::Group *parentNode);
    ~CarGeometry();

    void updateTrajectory();
    void update(double deltaTime);

    void setTransform(osg::Matrix transform);

protected:
    Vehicle &vehicle;
    static osg::Node *loadFile(const std::string &file);

    void removeFromSceneGraph();

    osg::ref_ptr<osg::MatrixTransform> transformNode;
    osg::ref_ptr<osg::LOD> lodNode;

    osg::Vec3 p0, p1, p2, p3;
};

#endif
