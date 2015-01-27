/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_PLANE
#define CO_PLANE

#include <osg/Vec3>
#include <util/coExport.h>
#include <osg/BoundingBox>

namespace opencover
{
class PLUGIN_UTILEXPORT coPlane
{

protected:
    osg::Vec3 _normal;
    osg::Vec3 _point;
    float _d;

public:
    coPlane(osg::Vec3 normal, osg::Vec3 point);
    virtual ~coPlane();

    // update plane normal an position
    virtual void update(osg::Vec3 normal, osg::Vec3 point);

    // return current position
    osg::Vec3 &getPosition()
    {
        return _point;
    };

    // return current normal
    osg::Vec3 &getNormal()
    {
        return _normal;
    };

    // return distance point to plane
    float getPointDistance(osg::Vec3 &point);

    osg::Vec3 getProjectedPoint(osg::Vec3 &point);

    // get the intersection point between an infinite line (Gerade)
    // defined by lp1 and lp2 and the plane
    // returns true, if an intersection point was found
    // false if the line and plane are parallel
    bool getLineIntersectionPoint(osg::Vec3 &lp1, osg::Vec3 &lp2, osg::Vec3 &isectPoint);

    // get the intersection point between a finite line (Linie)
    // and the plane
    // returns true if an intersection point is between lp1 and lp2
    // returns false if the plane and line are parallel or the
    // intersection point is not between lp1 and lp2
    bool getLineSegmentIntersectionPoint(osg::Vec3 &lp1, osg::Vec3 &lp2, osg::Vec3 &isectPoint);

    // get the intersection between the lines of a bounding box and the plane
    int getBoxIntersectionPoints(osg::BoundingBox &box, osg::Vec3 *isectPoints);
};
}
#endif
