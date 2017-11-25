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
#include <osgUtil/LineSegmentIntersector>

namespace osgUtil {
class IntersectionVisitor;
}

namespace opencover
{

class coIntersector;

class COVEREXPORT IntersectionHandler: public osg::Referenced
{
public:
    virtual ~IntersectionHandler() {}

    virtual bool canHandleDrawable(osg::Drawable *drawable) const = 0;
    virtual void intersect(osgUtil::IntersectionVisitor &iv, coIntersector &is, osg::Drawable *drawable) = 0;
};

class COVEREXPORT coIntersector: public osgUtil::LineSegmentIntersector
{
public:
    coIntersector(const osg::Vec3& start, const osg::Vec3& end);

    virtual Intersector* clone(osgUtil::IntersectionVisitor& iv) override;
    virtual void intersect(osgUtil::IntersectionVisitor& iv, osg::Drawable* drawable) override;

    void addHandler(osg::ref_ptr<IntersectionHandler> handler);

    bool intersectAndClip(osg::Vec3d &s, osg::Vec3d &e, const osg::BoundingBox &bb);

protected:
    std::vector<osg::ref_ptr<IntersectionHandler>> _handlers;
};

class COVEREXPORT coIntersection : public virtual vrui::vruiIntersection
{
    static coIntersection *intersector;
    coIntersection();
public:
    static coIntersection *instance();
    virtual ~coIntersection();

    void addHandler(osg::ref_ptr<IntersectionHandler> handler);

    virtual const char *getClassName() const;
    virtual const char *getActionName() const;

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

private:
    std::vector<std::vector<float> > elapsedTimes;
    std::vector<osg::ref_ptr<IntersectionHandler>> handlers;
};
}
#endif
