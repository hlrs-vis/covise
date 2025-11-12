/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "vvIntersection.h"

#include "vvPluginSupport.h"
#include "vvConfig.h"
#include "vvSceneGraph.h"
#include "vvViewer.h"

#include <config/CoviseConfig.h>

#include <vsg/maths/mat4.h>
#include <vsg/maths/vec3.h>
#include <vsg/utils/LineSegmentIntersector.h>

#include <OpenVRUI/vsg/VSGVruiHit.h>
#include <OpenVRUI/vsg/VSGVruiNode.h>

#include <OpenVRUI/util/vruiLog.h>

#include <input/input.h>

#include "vvNavigationManager.h"
#include "vvIntersectionInteractorManager.h"
#include "vvIntersectionInteractor.h"
#include "vvMSController.h"

#include <util/coWristWatch.h>
#include <numeric>
#include <limits>
#include <algorithm>

using namespace vsg;
using namespace vrui;
using covise::coCoviseConfig;

namespace  vive {

vvIntersection *vvIntersection::intersector = 0;

int vvIntersection::myFrameIndex = -1;
int vvIntersection::myFrame = 0;


void vive::vvIntersector::addHandler(vsg::ref_ptr<IntersectionHandler> handler)
{
    _handlers.push_back(handler);
}

vvIntersector::vvIntersector(const vsg::dvec3&start, const vsg::dvec3&end): Inherit<LineSegmentIntersector,vvIntersector>(start, end)
{}
/*
bool vvIntersector::intersectAndClip(vsg::vec3d &s, vsg::vec3d &e, const BoundingBox &bb)
{
    return LineSegmentIntersector::intersectAndClip(s, e, bb);
}

osgUtil::Intersector *vvIntersector::clone(osgUtil::IntersectionVisitor &iv)
{
    if ( _coordinateFrame==MODEL && iv.getModelMatrix()==0 )
    {
        vsg::ref_ptr<vvIntersector> cloned = new vvIntersector(_start, _end);
        cloned->_parent = this;
        cloned->_handlers = _handlers;
        return cloned.release();
    }

    vsg::dmat4 matrix;
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

    vsg::dmat4 inverse = vsg::inverse(matrix);
    vsg::ref_ptr<vvIntersector> cloned = new vvIntersector(_start*inverse, _end*inverse);
    cloned->_parent = this;
    cloned->_handlers = _handlers;
    return cloned.release();
}

void vvIntersector::intersect(osgUtil::IntersectionVisitor &iv, vsg::Node *drawable)
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


*/

bool vvIntersector::intersects(const vsg::dsphere& bs)
{
    return false;
}

vvIntersection::vvIntersection()
    : elapsedTimes(1)
{
    assert(!intersector);

    //VRUILOG("vvIntersection::<init> info: creating");

    if (myFrameIndex < 0)
    {
        myFrameIndex = (int)frames().size();
        frameIndex = myFrameIndex;
        frames().push_back(&myFrame);
        intersectors().push_back(this);
    }

    intersectionDist = coCoviseConfig::getFloat("VIVE.PointerAppearance.Intersection", 1000000.0f);

    intersector = this;
    numIsectAllNodes = 0;
}

bool vvIntersection::isVerboseIntersection()
{
    static bool verboseIntersection = coCoviseConfig::isOn(std::string("intersect"), std::string("VIVE.DebugLevel"), false);
    return verboseIntersection;
}

vvIntersection::~vvIntersection()
{
    //VRUILOG("vvIntersection::<dest> info: destroying");
    intersector = NULL;
}

vvIntersector *vvIntersection::newIntersector(const vsg::dvec3 &start, const vsg::dvec3 &end)
{
    auto intersector = new vvIntersector(start, end);
    for (auto h: handlers)
        intersector->addHandler(h);
    return intersector;
}

void vvIntersection::addHandler(vsg::ref_ptr<IntersectionHandler> handler)
{
    handlers.push_back(handler);
}

vvIntersection *vvIntersection::instance()
{
    if (intersector == 0)
    {
        //VRUILOG("vvIntersection::the info: making new vvIntersection");
        intersector = new vvIntersection();
    }
    return intersector;
}

void vvIntersection::intersect()
{
   // double beginTime = vvViewer::instance()->elapsedTime();

    vv->intersectedNode = 0;

    // for debug only
    //vvMSController::instance()->agreeInt(2000);
    if (Input::instance()->isTrackingOn())
    {
    // for debug only
    //vvMSController::instance()->agreeInt(2001);
        if (true /*&& !vvConfig::instance()->useWiiMote() */)
        {
            //fprintf(stderr, "vvIntersection::intersect() NOT wiiMode\n");
            if (vvNavigationManager::instance()->getMode() == vvNavigationManager::TraverseInteractors && vvIntersectionInteractorManager::the()->getCurrentIntersectionInteractor())
            {
                vsg::dmat4 handMat = vv->getPointerMat();
                vsg::dmat4 o_to_w = vv->getBaseMat();
                vsg::dvec3 currentInterPos_w = getTrans(vvIntersectionInteractorManager::the()->getCurrentIntersectionInteractor()->getMatrix()) * o_to_w;
                //fprintf(stderr, "--- vvIntersection::intersect currentInterPos_o(%f %f %f) \n", currentInterPos_w.x(),currentInterPos_w.y(),currentInterPos_w.z());
                setTrans(handMat,currentInterPos_w);

                intersect(handMat, false);
            }
            else
                intersect(vv->getPointerMat(), false);
        }
        /*
      else
      {
         //fprintf(stderr, "vvIntersection::intersect() wiiMode\n");
         Matrix m = vv->getPointerMat();
         vsg::vec3 trans = m.getTrans();
         //fprintf(stderr, "trans %f %f %f\n", trans[0], trans[1], trans[2]);
         //trans[1] = -500;
         m.setTrans(trans);
         //       VRTracker::instance()->setHandMat(m);
         intersect(m, false);
      }
      */
    }

    if (!vv->intersectedNode && vvConfig::instance()->mouseNav())
    {
        intersect(vv->getMouseMat(), true);
	
    // for debug only
    //vvMSController::instance()->agreeInt(2002);
    }
    /*
    if (vvViewer::instance()->getViewerStats() && vvViewer::instance()->getViewerStats()->collectStats("isect"))
    {
        int fn = vvViewer::instance()->getFrameStamp()->getFrameNumber();
        double endTime = vvViewer::instance()->elapsedTime();
        vvViewer::instance()->getViewerStats()->setAttribute(fn, "Isect begin time", beginTime);
        vvViewer::instance()->getViewerStats()->setAttribute(fn, "Isect end time", endTime);
        vvViewer::instance()->getViewerStats()->setAttribute(fn, "Isect time taken", endTime - beginTime);
    }*/
}

void vvIntersection::intersect(const vsg::dmat4 &handMat, bool mouseHit)
{
    //VRUILOG("vvIntersection::intersect info: called");
    vsg::dvec3 q0, q1;

    q0.set(0.0, 0.0, 0.0);
    q1.set(0.0f, intersectionDist, 0.0f);

    // xform the intersection line segment
    q0 = handMat * q0;
    q1 = handMat * q1;

    if (length2(q1-q0) < std::numeric_limits<float>::epsilon())
    {
        std::cerr << "vvIntersection: intersectionDist=" << intersectionDist << " too short" << std::endl;
        return;
    }

#if 0
    fprintf(stderr, "--- vvIntersection::intersect q0(%f %f %f) q1(%f %f %f) \n", q0.x(),q0.y(),q0.z(),q1.x(),q1.y(),q1.z());
    std::cerr << "vvIntersection::intersect info: hand " << handMat << std::endl;
    std::cerr << "vvIntersection::intersect info: ray from " << q0 << " to " << q1 << std::endl;
#endif

    LineSegment ray;
    ray.start = q0;
    ray.end = q1;

    if (q0 != q1)
    {
        ref_ptr<vvIntersector> intersector = vvIntersector::create(ray.start, ray.end);
        
        if (numIsectAllNodes > 0) 
        {
            intersector->traversalMask = Isect::Pick;
        }
        else 
        {
            intersector->traversalMask = Isect::Intersection;
        }
        
        vv->getMenuGroup()->accept(*intersector);
        //vv->getScene()->accept(*intersector);

        auto isects = intersector->intersections;

        // sort intersections front to back 
        std::sort(isects.begin(), isects.end(), [](auto& lhs, auto& rhs) { return lhs->ratio < rhs->ratio; });

        // this loop should break after processing nearest element
        for (const auto& isect : isects) 
        {
            // logic to process nearest element and call actions accordingly
            
            vv->intersectionHitPointWorld = isect->worldIntersection;
            vv->intersectionHitPointLocal = isect->localIntersection;
            vv->intersectedNodePath = isect->nodePath;
            vv->intersectedNode = const_cast<Node*>(isect->nodePath.back());

            /*
            cerr << "The intersected node computed transform is: " << computeTransform(vv->intersectedNodePath) << endl;
            cerr << "The localToWorld  matrix of the intersected node is: " << isect->localToWorld << endl;
            cerr << "The world intersection point: " << isect->worldIntersection << endl;
            */

            // callActions goes up through the parents of the node
            // and calls all coActions
            VSGVruiHit hit(*isect, mouseHit);
            if (vv->intersectedNode) 
            {
                VSGVruiNode node(vv->intersectedNode);
                node.setNodePath(isect->nodePath);
                callActions(&node, &hit);
            }
            break; // stop this for loop for going through other hits farer away from nearest
        }

        /*IntersectionVisitor visitor;
        if (numIsectAllNodes > 0)
        {
            visitor.setTraversalMask(Isect::Pick);
        }
        else
        {
            visitor.setTraversalMask(Isect::Intersection);
        }
        vsg::ref_ptr<vvIntersector> intersector = new vvIntersector(ray->start(), ray->end());
        for (auto h: handlers)
            intersector->addHandler(h);
        visitor.setIntersector(intersector.get());
        //visitor.addLineSegment(ray.get());

        {

            covise::coWristWatch watch;
            vv->getScene()->accept(visitor);

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
        //VRUILOG("vvIntersection::intersect info: hit");
        //fprintf(stderr, " --- HIT \n");

        // check which node in the hit list is also visible
        bool hasVisibleHit = false;
        for (const auto &isect: isects)
        {
            vsg::Node *node = nullptr;
            if (isect.drawable && isect.drawable->getNumParents()>0)
                node = isect.drawable->getParent(0);

            if (node && (node->getNodeMask() & (Isect::Visible)))
            {
                //hitInformation = hitList[i];
                hasVisibleHit = true;
                // check also parents of this visible node,
                vsg::Node *parent = NULL;
                // there could be an invisible dcs above
                if (node->getNumParents())
                    parent = node->getParent(0);
                while (parent && (parent != vv->getObjectsRoot()))
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
                    vv->intersectionHitPointWorld = isect.getWorldIntersectPoint();
                    vv->intersectionHitPointWorldNormal = isect.getWorldIntersectNormal();
                    vv->intersectionHitPointLocal = isect.getLocalIntersectPoint();
                    vv->intersectionHitPointLocalNormal = isect.getLocalIntersectNormal();
                    vv->intersectionMatrix = isect.matrix;
                    vv->intersectedDrawable = isect.drawable;
                    vv->intersectedNode = nullptr;
                    if (isect.drawable->getNumParents() > 0)
                        vv->intersectedNode = dynamic_cast<vsg::Node *>(isect.drawable->getParent(0));

                    //if( !vv->intersectedNode.get()->getName().empty())
                    //    fprintf(stderr,"vvIntersection::intersect hit node %s\n", vv->intersectedNode.get()->getName().c_str());
                    //else
                    //    fprintf(stderr,"vvIntersection::intersect hit node without name\n");

                    vv->intersectedNodePath = isect.nodePath;
                    // walk up to the root and call all coActions
                    VSGVruiHit hit(isect, mouseHit);
                    if (vv->intersectedNode.get())
                    {
                        VSGVruiNode node(vv->intersectedNode.get());
                        callActions(&node, &hit);
                    }
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
        }*/
    }
   /* // for debug only
    vvMSController::instance()->agreeInt((int)(vv->intersectedNode!=NULL));
    if(vv->intersectedNode!=NULL)
    {
    vvMSController::instance()->agreeString(vv->intersectedNode.get()->getName());
    }*/
    
}

const char *vvIntersection::getClassName() const
{
    return "vvIntersection";
}

const char *vvIntersection::getActionName() const
{
    return "coAction";
}

void vvIntersection::isectAllNodes(bool isectAll)
{
    //fprintf(stderr,"vvIntersection::isectAllNodes %d...", isectAll);

    if (isectAll)
        numIsectAllNodes++;
    else
    {
        if (numIsectAllNodes > 0)
            numIsectAllNodes--;
    }

    //fprintf(stderr,"numIsectAllNodes=%d\n", numIsectAllNodes);
}

void vvIntersection::forceIsectAllNodes(bool isectAll)
{
    //fprintf(stderr,"vvIntersection::isectAllNodes %d...", isectAll);

    if (isectAll)
        numIsectAllNodes++;
    else
    {
        numIsectAllNodes = 0;
    }

    //fprintf(stderr,"numIsectAllNodes=%d\n", numIsectAllNodes);
}

}
