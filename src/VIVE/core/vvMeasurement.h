/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_MEASUREMENT_H
#define CO_MEASUREMENT_H

#include <vsg/maths/vec3.h>
#include <vsg/nodes/Group.h>

namespace vive
{

class vvLabel;

class vvMeasurement
{
public:
    vvMeasurement();
    ~vvMeasurement();

    void start();
    void update();

    void preFrame();

private:
    vsg::vec3 measureStartHitWorld_;
    vsg::ref_ptr<vsg::Group> measureGroup_;
    vsg::ref_ptr<vsg::Node> measureGeometry_;
    vsg::ref_ptr<vsg::vec3Array> measureVertices_;
    vvLabel *measureLabel_;
    vvLabel *measureOrthoLabel_;
    float measureScale_;
    std::string measureUnit_;
};
}
#endif
