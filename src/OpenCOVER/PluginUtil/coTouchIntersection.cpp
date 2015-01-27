/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cover/coTouchIntersection.h>

#include <cover/coVRPluginSupport.h>
#include <cover/VRSceneGraph.h>

#include <osg/LineSegment>
#include <osg/Matrix>
#include <osg/Vec3>
#include <osgUtil/IntersectVisitor>

#include <OpenVRUI/osg/OSGVruiHit.h>
#include <OpenVRUI/osg/OSGVruiNode.h>

#include <config/CoviseConfig.h>

using namespace osg;
using namespace osgUtil;
using namespace covise;
using namespace opencover;

coTouchIntersection *coTouchIntersection::touchIntersector = 0;

int coTouchIntersection::myFrameIndex = -1;
int coTouchIntersection::myFrame = 0;

coTouchIntersection::coTouchIntersection()
{

    if (myFrameIndex < 0)
    {
        myFrameIndex = frames.size();
        frames.push_back(&myFrame);
    }

    handSize = coCoviseConfig::getFloat("COVER.HandSize", 100.0f);
    touchIntersector = this;
}

coTouchIntersection::~coTouchIntersection()
{
}

coTouchIntersection *coTouchIntersection::the()
{
    if (touchIntersector == 0)
        touchIntersector = new coTouchIntersection();
    return touchIntersector;
}

void coTouchIntersection::intersect()
{
    cover->intersectedNode = 0;

    if (!cover->mouseTracking())
        intersect(cover->getPointerMat(), false);

    if (!cover->intersectedNode && cover->mouseNav())
        intersect(cover->getMouseMat(), true);
}

void coTouchIntersection::intersect(const osg::Matrix &handMat, bool mouseHit)
{

    cerr << "coTouchIntersection::intersect info: called" << endl;

    Vec3 q0, q1, q2, q3, q4, q5;

    // 3 line segments for interscetion test hand-object
    // of course this test can fail
    q0.set(0.0f, -0.5f * handSize, 0.0f);
    q1.set(0.0f, 0.5f * handSize, 0.0f);
    q2.set(-0.5f * handSize, 0.0f, 0.0f);
    q3.set(0.5f * handSize, 0.0f, 0.0f);
    q4.set(0.0f, 0.0f, -0.5f * handSize);
    q5.set(0.0f, 0.0f, 0.5f * handSize);

    // xform the intersection line segment
    q0 = handMat.preMult(q0);
    q1 = handMat.preMult(q1);
    q2 = handMat.preMult(q2);
    q3 = handMat.preMult(q3);
    q4 = handMat.preMult(q4);
    q5 = handMat.preMult(q5);

    ref_ptr<LineSegment> ray1 = new LineSegment();
    ref_ptr<LineSegment> ray2 = new LineSegment();
    ref_ptr<LineSegment> ray3 = new LineSegment();
    ray1->set(q0, q1);
    ray2->set(q2, q3);
    ray3->set(q4, q5);

    IntersectVisitor visitor;
    visitor.addLineSegment(ray1.get());
    visitor.addLineSegment(ray2.get());
    visitor.addLineSegment(ray3.get());

    cover->getScene()->traverse(visitor);

    ref_ptr<LineSegment> hitRay = 0;

    if (visitor.getNumHits(ray1.get()))
        hitRay = ray1;
    else if (visitor.getNumHits(ray2.get()))
        hitRay = ray2;
    else if (visitor.getNumHits(ray3.get()))
        hitRay = ray3;

    if (visitor.getNumHits(hitRay.get()))
    {
        Hit hitInformation = visitor.getHitList(hitRay.get()).front();
        cover->intersectionHitPointWorld = hitInformation.getWorldIntersectPoint();
        cover->intersectionHitPointLocal = hitInformation.getLocalIntersectPoint();
        cover->intersectionMatrix = hitInformation._matrix;
        cover->intersectedNode = hitInformation._geode;
        // walk up to the root and call all coActions
        OSGVruiHit hit(hitInformation, mouseHit);
        OSGVruiNode node(cover->intersectedNode.get());
        callActions(&node, &hit);
    }
}

const char *coTouchIntersection::getClassName() const
{
    return "coTouchIntersection";
}

const char *coTouchIntersection::getActionName() const
{
    return "coTouchAction";
}
