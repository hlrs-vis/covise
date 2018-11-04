/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coIntersection.h"

#include "coVRPluginSupport.h"
#include "coVRConfig.h"
#include "VRSceneGraph.h"
#include "VRViewer.h"

#include <config/CoviseConfig.h>

#include <osg/LineSegment>
#include <osg/Matrix>
#include <osg/Vec3>
#include <osg/io_utils>
#include <osgUtil/IntersectionVisitor>
#include <osgUtil/LineSegmentIntersector>

#include <OpenVRUI/osg/OSGVruiHit.h>
#include <OpenVRUI/osg/OSGVruiNode.h>

#include <OpenVRUI/util/vruiLog.h>

#include <input/input.h>

#include "coVRNavigationManager.h"
#include "coVRIntersectionInteractorManager.h"
#include "coVRMSController.h"

#include <util/coWristWatch.h>
#include <numeric>

using namespace osg;
using namespace osgUtil;
using namespace vrui;
using covise::coCoviseConfig;

namespace  opencover {

coIntersection *coIntersection::intersector = 0;

int coIntersection::myFrameIndex = -1;
int coIntersection::myFrame = 0;


coIntersector::coIntersector(const Vec3 &start, const Vec3 &end)
: osgUtil::LineSegmentIntersector(start, end)
{}

void coIntersector::addHandler(osg::ref_ptr<IntersectionHandler> handler)
{
    _handlers.push_back(handler);
}

bool coIntersector::intersectAndClip(Vec3d &s, Vec3d &e, const BoundingBox &bb)
{
    return LineSegmentIntersector::intersectAndClip(s, e, bb);
}

osgUtil::Intersector *coIntersector::clone(osgUtil::IntersectionVisitor &iv)
{
    if ( _coordinateFrame==MODEL && iv.getModelMatrix()==0 )
    {
        osg::ref_ptr<coIntersector> cloned = new coIntersector(_start, _end);
        cloned->_parent = this;
        cloned->_handlers = _handlers;
        return cloned.release();
    }

    osg::Matrix matrix;
    switch (_coordinateFrame)
    {
    case WINDOW:
        if (iv.getWindowMatrix()) matrix.preMult(*iv.getWindowMatrix());
        if (iv.getProjectionMatrix()) matrix.preMult(*iv.getProjectionMatrix());
        if (iv.getViewMatrix()) matrix.preMult(*iv.getViewMatrix());
        if (iv.getModelMatrix()) matrix.preMult(*iv.getModelMatrix());
        break;
    case PROJECTION:
        if (iv.getProjectionMatrix()) matrix.preMult(*iv.getProjectionMatrix() );
        if (iv.getViewMatrix()) matrix.preMult(*iv.getViewMatrix());
        if (iv.getModelMatrix()) matrix.preMult(*iv.getModelMatrix());
        break;
    case VIEW:
        if (iv.getViewMatrix()) matrix.preMult(*iv.getViewMatrix());
        if (iv.getModelMatrix()) matrix.preMult(*iv.getModelMatrix());
        break;
    case MODEL:
        if (iv.getModelMatrix()) matrix = *iv.getModelMatrix();
        break;
    }

    osg::Matrix inverse = osg::Matrix::inverse(matrix);
    osg::ref_ptr<coIntersector> cloned = new coIntersector(_start*inverse, _end*inverse);
    cloned->_parent = this;
    cloned->_handlers = _handlers;
    return cloned.release();
}

void coIntersector::intersect(osgUtil::IntersectionVisitor &iv, osg::Drawable *drawable)
{
    for (auto &h: _handlers)
    {
        if (h->canHandleDrawable(drawable))
        {
            h->intersect(iv, *this, drawable);
            return;
        }
    }

    osgUtil::LineSegmentIntersector::intersect(iv, drawable);
}





coIntersection::coIntersection()
    : elapsedTimes(1)
{
    assert(!intersector);

    //VRUILOG("coIntersection::<init> info: creating");

    if (myFrameIndex < 0)
    {
        myFrameIndex = frames().size();
        frameIndex = myFrameIndex;
        frames().push_back(&myFrame);
        intersectors().push_back(this);
    }

    intersectionDist = coCoviseConfig::getFloat("COVER.PointerAppearance.Intersection", 1000000.0f);

    intersector = this;
    numIsectAllNodes = 0;
}

bool coIntersection::isVerboseIntersection()
{
    static bool verboseIntersection = coCoviseConfig::isOn(std::string("intersect"), std::string("COVER.DebugLevel"), false);
    return verboseIntersection;
}

coIntersection::~coIntersection()
{
    //VRUILOG("coIntersection::<dest> info: destroying");
    intersector = NULL;
}

coIntersector *coIntersection::newIntersector(const Vec3 &start, const Vec3 &end)
{
    auto intersector = new coIntersector(start, end);
    for (auto h: handlers)
        intersector->addHandler(h);
    return intersector;
}

void coIntersection::addHandler(osg::ref_ptr<IntersectionHandler> handler)
{
    handlers.push_back(handler);
}

coIntersection *coIntersection::instance()
{
    if (intersector == 0)
    {
        //VRUILOG("coIntersection::the info: making new coIntersection");
        intersector = new coIntersection();
    }
    return intersector;
}

void coIntersection::intersect()
{
    double beginTime = VRViewer::instance()->elapsedTime();

    cover->intersectedNode = 0;

    // for debug only
    //coVRMSController::instance()->syncInt(2000);
    if (Input::instance()->isTrackingOn())
    {
    // for debug only
    //coVRMSController::instance()->syncInt(2001);
        if (true /*&& !coVRConfig::instance()->useWiiMote() */)
        {
            //fprintf(stderr, "coIntersection::intersect() NOT wiiMode\n");
            if (coVRNavigationManager::instance()->getMode() == coVRNavigationManager::TraverseInteractors && coVRIntersectionInteractorManager::the()->getCurrentIntersectionInteractor())
            {
                osg::Matrix handMat = cover->getPointerMat();
                osg::Matrix o_to_w = cover->getBaseMat();
                osg::Vec3 currentInterPos_w = coVRIntersectionInteractorManager::the()->getCurrentIntersectionInteractor()->getMatrix().getTrans() * o_to_w;
                //fprintf(stderr, "--- coIntersection::intersect currentInterPos_o(%f %f %f) \n", currentInterPos_w.x(),currentInterPos_w.y(),currentInterPos_w.z());
                handMat.setTrans(currentInterPos_w);

                intersect(handMat, false);
            }
            else
                intersect(cover->getPointerMat(), false);
        }
        /*
      else
      {
         //fprintf(stderr, "coIntersection::intersect() wiiMode\n");
         Matrix m = cover->getPointerMat();
         Vec3 trans = m.getTrans();
         //fprintf(stderr, "trans %f %f %f\n", trans[0], trans[1], trans[2]);
         //trans[1] = -500;
         m.setTrans(trans);
         //       VRTracker::instance()->setHandMat(m);
         intersect(m, false);
      }
      */
    }

    if (!cover->intersectedNode && coVRConfig::instance()->mouseNav())
    {
        intersect(cover->getMouseMat(), true);
	
    // for debug only
    //coVRMSController::instance()->syncInt(2002);
    }

    if (VRViewer::instance()->getViewerStats() && VRViewer::instance()->getViewerStats()->collectStats("isect"))
    {
        int fn = VRViewer::instance()->getFrameStamp()->getFrameNumber();
        double endTime = VRViewer::instance()->elapsedTime();
        VRViewer::instance()->getViewerStats()->setAttribute(fn, "Isect begin time", beginTime);
        VRViewer::instance()->getViewerStats()->setAttribute(fn, "Isect end time", endTime);
        VRViewer::instance()->getViewerStats()->setAttribute(fn, "Isect time taken", endTime - beginTime);
    }
}

void coIntersection::intersect(const osg::Matrix &handMat, bool mouseHit)
{
    //VRUILOG("coIntersection::intersect info: called");
    Vec3 q0, q1;

    q0.set(0.0f, 0.0f, 0.0f);
    q1.set(0.0f, intersectionDist, 0.0f);

    // xform the intersection line segment
    q0 = handMat.preMult(q0);
    q1 = handMat.preMult(q1);

    if ((q1-q0).length2() < std::numeric_limits<float>::epsilon())
    {
        std::cerr << "coIntersection: intersectionDist=" << intersectionDist << " too short" << std::endl;
        return;
    }

#if 0
    fprintf(stderr, "--- coIntersection::intersect q0(%f %f %f) q1(%f %f %f) \n", q0.x(),q0.y(),q0.z(),q1.x(),q1.y(),q1.z());
    std::cerr << "coIntersection::intersect info: hand " << handMat << std::endl;
    std::cerr << "coIntersection::intersect info: ray from " << q0 << " to " << q1 << std::endl;
#endif

    ref_ptr<LineSegment> ray = new LineSegment();
    ray->set(q0, q1);

    if (q0 != q1)
    {
        IntersectionVisitor visitor;
        if (numIsectAllNodes > 0)
        {
            visitor.setTraversalMask(Isect::Pick);
        }
        else
        {
            visitor.setTraversalMask(Isect::Intersection);
        }
        osg::ref_ptr<coIntersector> intersector = new coIntersector(ray->start(), ray->end());
        for (auto h: handlers)
            intersector->addHandler(h);
        visitor.setIntersector(intersector.get());
        //visitor.addLineSegment(ray.get());

        {

            covise::coWristWatch watch;
            cover->getScene()->accept(visitor);

            if (isVerboseIntersection())
            {
                elapsedTimes[0].push_back(watch.elapsed());
                std::cerr << " avg. intersection times";
                for (size_t ctr = 0; ctr < elapsedTimes.size(); ++ctr)
                {
                    std::cerr << " | " << ctr + 1 << ":"
                              << std::accumulate(elapsedTimes[ctr].begin(), elapsedTimes[ctr].end(), 0.0f) / elapsedTimes[ctr].size()
                              << "s";
                }
                std::cerr << " | " << std::endl;
            }
        }

        auto isects = intersector->getIntersections();
        //VRUILOG("coIntersection::intersect info: hit");
        //fprintf(stderr, " --- HIT \n");

        // check which node in the hit list is also visible
        bool hasVisibleHit = false;
        for (const auto &isect: isects)
        {
            osg::Node *node = nullptr;
            if (isect.drawable && isect.drawable->getNumParents()>0)
                node = isect.drawable->getParent(0);

            if (node && (node->getNodeMask() & (Isect::Visible)))
            {
                //hitInformation = hitList[i];
                hasVisibleHit = true;
                // check also parents of this visible node,
                osg::Node *parent = NULL;
                // there could be an invisible dcs above
                if (node->getNumParents())
                    parent = node->getParent(0);
                while (parent && (parent != cover->getObjectsRoot()))
                {

                    if (parent->getNodeMask() & (Isect::Visible))
                    {

                        //parent was also visible, get his parent
                        if (parent->getNumParents())
                        {

                            parent = parent->getParent(0);
                        }
                        else
                            parent = NULL;
                    }
                    else // parent not visible
                    {

                        //stop this while loop for going ip in sg
                        hasVisibleHit = false;
                        break;
                    }
                }
                if (hasVisibleHit) // all parents are also visible
                {
                    cover->intersectionHitPointWorld = isect.getWorldIntersectPoint();
                    cover->intersectionHitPointWorldNormal = isect.getWorldIntersectNormal();
                    cover->intersectionHitPointLocal = isect.getLocalIntersectPoint();
                    cover->intersectionHitPointLocalNormal = isect.getLocalIntersectNormal();
                    cover->intersectionMatrix = isect.matrix;
                    cover->intersectedDrawable = isect.drawable;
                    cover->intersectedNode = nullptr;
                    if (isect.drawable->getNumParents() > 0)
                        cover->intersectedNode = dynamic_cast<osg::Geode *>(isect.drawable->getParent(0));

                    //if( !cover->intersectedNode.get()->getName().empty())
                    //    fprintf(stderr,"coIntersection::intersect hit node %s\n", cover->intersectedNode.get()->getName().c_str());
                    //else
                    //    fprintf(stderr,"coIntersection::intersect hit node without name\n");

                    cover->intersectedNodePath = isect.nodePath;
                    // walk up to the root and call all coActions
                    OSGVruiHit hit(isect, mouseHit);
                    OSGVruiNode node(cover->intersectedNode.get());
                    callActions(&node, &hit);

                    break; // stop this for loop for going through other hits farer away from nearest
                }
            }

            //else
            //{
            //   if (! (hitList[i]._geode->getName().empty()) )
            //   {
            //      fprintf(stderr,"intersceting a unvisible node with name %s\n", hitList[i]._geode->getName().c_str());
            //   }
            //}
        }
    }
   /* // for debug only
    coVRMSController::instance()->syncInt((int)(cover->intersectedNode!=NULL));
    if(cover->intersectedNode!=NULL)
    {
    coVRMSController::instance()->syncStringStop(cover->intersectedNode.get()->getName());
    }*/
    
}

const char *coIntersection::getClassName() const
{
    return "coIntersection";
}

const char *coIntersection::getActionName() const
{
    return "coAction";
}

void coIntersection::isectAllNodes(bool isectAll)
{
    //fprintf(stderr,"coIntersection::isectAllNodes %d...", isectAll);

    if (isectAll)
        numIsectAllNodes++;
    else
    {
        if (numIsectAllNodes > 0)
            numIsectAllNodes--;
    }

    //fprintf(stderr,"numIsectAllNodes=%d\n", numIsectAllNodes);
}

void coIntersection::forceIsectAllNodes(bool isectAll)
{
    //fprintf(stderr,"coIntersection::isectAllNodes %d...", isectAll);

    if (isectAll)
        numIsectAllNodes++;
    else
    {
        numIsectAllNodes = 0;
    }

    //fprintf(stderr,"numIsectAllNodes=%d\n", numIsectAllNodes);
}

}
