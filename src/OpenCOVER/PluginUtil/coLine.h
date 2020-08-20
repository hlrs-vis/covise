/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_LINE
#define CO_LINE

#include <osg/Vec3>
#include <util/coExport.h>

namespace opencover
{
class PLUGIN_UTILEXPORT coLine
{

protected:
    osg::Vec3 _dirVec;
    osg::Vec3 _point;

public:
    coLine(osg::Vec3 point1, osg::Vec3 point2);
    virtual ~coLine();

    // update line
    virtual void update(osg::Vec3 point1, osg::Vec3 point2);

    // return current position
    osg::Vec3 getPosition() const 
    {
        return _point;
    };

    // return current direction vector
    osg::Vec3 getDirectionVector() const
    {
        return _dirVec;
    };

    // get the shortest distance between two infinite lines (Gerade)
    // returns true, if a shortest distance can be calculated
    // false if the two lines are parallel
    bool getShortestLineDistance(const osg::Vec3 &lp1, const osg::Vec3 &lp2, double &shortestDistance) const; 

    // find the two points X1 on Line1 and X2 on Line2 such that the distance between X1 and X2 is minimal
    // returns true is lines are skew and a point of shortest distance can be calculated
    // returns false if the two lines are parallel
    // lp1 and lp2 are the two points which create the Line2
    // Line 1 ist the this coLine object
    bool getPointsOfShortestDistance(const osg::Vec3 &lp1, const osg::Vec3 &lp2, osg::Vec3& pointLine1, osg::Vec3& pointLine2) const; 

};
}
#endif
