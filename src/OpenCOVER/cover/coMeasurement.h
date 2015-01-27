/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_MEASUREMENT_H
#define CO_MEASUREMENT_H

#include <osg/Vec3>
#include <osg/Vec4>
#include <osg/Group>
#include <osg/Geometry>

namespace opencover
{

class coVRLabel;

class coMeasurement
{
public:
    coMeasurement();
    ~coMeasurement();

    void start();
    void update();

    void preFrame();

private:
    osg::Vec3 measureStartHitWorld_;
    osg::ref_ptr<osg::Group> measureGroup_;
    osg::ref_ptr<osg::Geometry> measureGeometry_;
    osg::ref_ptr<osg::Vec3Array> measureVertices_;
    coVRLabel *measureLabel_;
    coVRLabel *measureOrthoLabel_;
    float measureScale_;
    std::string measureUnit_;
};
}
#endif
