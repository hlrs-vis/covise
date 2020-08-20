/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coLine.h"
#include <osg/Vec3>
#include <stdio.h>

using namespace opencover;

coLine::coLine(osg::Vec3 point1, osg::Vec3 point2)
{
    update(point1, point2);
}

void
coLine::update(osg::Vec3 point1, osg::Vec3 point2)
{
    //fprintf(stderr,"coLine::update\n");
    _dirVec = point2 - point1;
    _dirVec.normalize();
    _point = point1;
}

coLine::~coLine()
{
}

bool coLine::getShortestLineDistance(const osg::Vec3& point1, const osg::Vec3 &point2, double &shortestDistance) const
{
    osg::Vec3 u1 = _dirVec;
    osg::Vec3 u2 = point2 - point1;
    osg::Vec3 cross = u1^u2;
    osg::Vec3 s = _point - point1;
    if(cross.length() == 0) // parallel
        return false;
    
    shortestDistance  = std::fabs(((s*cross)/cross.length()));

    return true;
}

// math can be found here: https://en.wikipedia.org/wiki/Skew_lines#Nearest_Points
bool coLine::getPointsOfShortestDistance(const osg::Vec3 &lp1, const osg::Vec3 &lp2, osg::Vec3& pointLine1, osg::Vec3& pointLine2) const
{
    osg::Vec3 u1 = _dirVec;
    osg::Vec3 u2 = lp2 - lp1;
    osg::Vec3 cross = u1^u2;
    if(cross.length() == 0) // parallel
        return false;

    pointLine1 = _point +  u1.operator*(((lp1-_point)*(u2^cross))/(u1*(u2^cross)));
    pointLine2 = lp1 +  u1.operator*(((_point-lp1)*(u1^cross))/(u2*(u1^cross)));

    return true;
}
