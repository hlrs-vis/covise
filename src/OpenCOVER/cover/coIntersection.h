/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_INTERSECTION_H
#define CO_INTERSECTION_H

/*! \file
 \brief  OpenVRUI interface to (object-pointing device) intersections

 \author Andreas Kopecki <kopecki@hlrs.de>
 \author (C) 2004
         High Performance Computing Center Stuttgart,
         Allmandring 30,
         D-70550 Stuttgart,
         Germany

 \date  2004
 */

#include <OpenVRUI/sginterface/vruiIntersection.h>
#include <osg/Matrix>
#include <osgUtil/IntersectVisitor>
namespace opencover
{
class COVEREXPORT coIntersection : public virtual vrui::vruiIntersection
{
public:
    coIntersection();
    virtual ~coIntersection();

    virtual const char *getClassName() const;
    virtual const char *getActionName() const;

    static coIntersection *instance();
    osgUtil::Hit hitInformation;

    void isectAllNodes(bool isectAll);
    void forceIsectAllNodes(bool isectAll);
    int numIsectAllNodes;

    static bool isVerboseIntersection();

protected:
    virtual void intersect(); // do the intersection
    // do the intersection
    void intersect(const osg::Matrix &mat, bool mouseHit);

    /// value of the PointerAppearance.Intersection config var
    float intersectionDist;

    static int myFrameIndex;
    static int myFrame;

    static coIntersection *intersector;

private:
    std::vector<std::vector<float> > elapsedTimes;
};
}
#endif
