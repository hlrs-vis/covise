/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once

#include <OpenVRUI/sginterface/vruiIntersection.h>
#include <vsg/maths/mat4.h>
#include <vsg/utils/LineSegmentIntersector.h>

namespace osgUtil {
class IntersectionVisitor;
}

namespace vive
{

class vvIntersector;

class VVCORE_EXPORT IntersectionHandler : public vsg::Inherit<vsg::Object, IntersectionHandler>
{
public:
    virtual ~IntersectionHandler() {}

    virtual bool canHandleDrawable(vsg::Node *drawable) const = 0;
    virtual void intersect(vsg::Visitor &iv, vvIntersector &is, vsg::Node *drawable) = 0;
};

class VVCORE_EXPORT vvIntersector: public vsg::Inherit<vsg::LineSegmentIntersector, vvIntersector>
{
public:
    vvIntersector(const vsg::dvec3& start, const vsg::dvec3& end);

    /// check for intersection with sphere
    bool intersects(const vsg::dsphere& bs) override;

    void addHandler(vsg::ref_ptr<IntersectionHandler> handler);

protected:
    std::vector<vsg::ref_ptr<IntersectionHandler>> _handlers;
};

class VVCORE_EXPORT vvIntersection : public virtual vrui::vruiIntersection
{
    static vvIntersection *intersector;
    vvIntersection();
public:
    static vvIntersection *instance();
    virtual ~vvIntersection();

    vvIntersector *newIntersector(const vsg::dvec3 &start, const vsg::dvec3 &end);

    void addHandler(vsg::ref_ptr<IntersectionHandler> handler);

    virtual const char *getClassName() const;
    virtual const char *getActionName() const;

    void isectAllNodes(bool isectAll);
    void forceIsectAllNodes(bool isectAll);
    int numIsectAllNodes;

    static bool isVerboseIntersection();

protected:
    virtual void intersect(); // do the intersection
    // do the intersection
    void intersect(const vsg::dmat4 &mat, bool mouseHit);

    /// value of the PointerAppearance.Intersection config var
    float intersectionDist;

    static int myFrameIndex;
    static int myFrame;

private:
    std::vector<std::vector<float> > elapsedTimes;
    std::vector<vsg::ref_ptr<IntersectionHandler>> handlers;
};
}
EVSG_type_name(vive::vvIntersector);
EVSG_type_name(vive::IntersectionHandler);

