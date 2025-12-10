/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef OPENCOVER_PLUGINS_TRAFFIC_GEOMETRY_H
#define OPENCOVER_PLUGINS_TRAFFIC_GEOMETRY_H

#include <osg/Matrix>
#include <osg/Vec3>

class Geometry
{
public:
    virtual void setTransform(osg::Matrix transform) = 0;
    void setTransform(osg::Vec3 position, double heading);

    // Called when the model was updated, can be used to track state,
    // derive current and past behaviour, and update graphis.
    virtual void update(double deltaTime) { }
};

#endif
