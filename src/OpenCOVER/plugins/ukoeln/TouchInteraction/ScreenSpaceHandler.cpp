/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ScreenSpaceHandler.cpp

#include "ScreenSpaceHandler.h"

#include <cassert>

#include "levmar/levmar.h"

#include <cover/coVRConfig.h>
#include <cover/coVRPluginSupport.h>

#include <cover/VRSceneGraph.h>
#include <cover/VRViewer.h>

#include <osgUtil/LineSegmentIntersector>

#define MAX_ITERATIONS 100

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
template <typename T>
inline T square(T const &x)
{
    return x * x;
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
void ScreenSpaceHandler::init()
{
    this->camera = opencover::coVRConfig::instance()->screens[0].camera;

    this->currentTransformMatrix = opencover::cover->getXformMat();

    this->setState(Possible);
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
void ScreenSpaceHandler::finish()
{
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
void ScreenSpaceHandler::onTouchPressed(Touches const & /*touches*/, Touch const &reason)
{
    osg::ref_ptr<osg::Viewport> viewport = this->camera->getViewport();

    float mouse_x = (reason.x) * viewport->width();
    float mouse_y = (1.0f - reason.y) * viewport->height();

    Cursor cur;

    cur.id = reason.id;
    cur.targetPos = osg::Vec2d(mouse_x, mouse_y);

    // Compute intersection -- if any -- and add the touch point to the list of valid touch points
    if (this->intersect(cur.objectPos, cur.targetPos))
    {
        this->currentTransformMatrix = opencover::cover->getXformMat();

        cur.objectPos = cur.objectPos * osg::Matrix::inverse(this->currentTransformMatrix);

        this->cursors[cur.id] = cur;

        // Set the handler's state to Began, if this was the first valid touch point
        if (this->getState() == Possible)
        {
            this->setState(Began);
        }
    }
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
void ScreenSpaceHandler::onTouchesMoved(Touches const &touches)
{
    osg::ref_ptr<osg::Viewport> viewport = this->camera->getViewport();

    // For each currently active touch for point which is in the list of valid
    // touch points maintained by the handler...
    for (Touches::const_iterator it = touches.begin(); it != touches.end(); ++it)
    {
        Touch const &touch = it->second;

        // Check if the point is in the list
        if (this->cursors.count(touch.id) > 0)
        {
            float mouse_x = (touch.x) * viewport->width();
            float mouse_y = (1.0f - touch.y) * viewport->height();

            // Update the cursor position
            // This position will be used to minimize the error in the errorFunction() below
            this->cursors[touch.id].targetPos = osg::Vec2d(mouse_x, mouse_y);

            // Signal the plugin that the ScreenSpaceHandler needs to update
            // OpenCOVER's transform matrix, which will be done in onUpdate()
            this->setState(Changed);
        }
    }
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
void ScreenSpaceHandler::onTouchReleased(Touches const &touches, Touch const &reason)
{
    // Check if the touch point released is in the list of valid touch points
    // Ignore if not
    if (this->cursors.count(reason.id) > 0)
    {
        // Remove the touch point from the list of valid touch points
        this->cursors.erase(reason.id);

        // Update the current transform matrix
        this->currentTransformMatrix = opencover::cover->getXformMat();

        // And update all remaining touch points
        // NOTE: This might remove some further touch points from the list

        osg::Matrixd invT = osg::Matrix::inverse(this->currentTransformMatrix);

        for (CursorMap::iterator it = this->cursors.begin(); it != this->cursors.end();)
        {
            Cursor &cur = it->second;

            if (intersect(cur.objectPos, cur.targetPos))
            {
                // Touch point still on object
                // Compute new object-space position
                cur.objectPos = cur.objectPos * invT;
                // Increment iterator!
                ++it;
            }
            else
            {
                // This point is no longer valid...
                // Remove and increment iterator!
                this->cursors.erase(it++);
            }
        }
    }

    // No more touch points?
    if (touches.empty())
    {
        this->setState(Possible);
    }
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
void ScreenSpaceHandler::onUpdate()
{
    if (this->cursors.empty())
        return;

    osg::Vec3 translation;
    osg::Quat rotation;

    this->computeTransform(translation, rotation);

    osg::Matrixd T = osg::Matrixd::translate(translation);
    osg::Matrixd R = osg::Matrixd::rotate(rotation);

    this->currentTransformMatrix *= this->camera->getViewMatrix();
    this->currentTransformMatrix *= R;
    this->currentTransformMatrix *= T;
    this->currentTransformMatrix *= this->camera->getInverseViewMatrix();

    opencover::cover->setXformMat(this->currentTransformMatrix);

    //
    // Changed state was handled...
    // Set state to <Began>. The next time something interesting happens in onTouchesMoved() or
    // onTouchReleased() the state will be set to <Changed> again.
    //
    this->setState(Began);
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
bool ScreenSpaceHandler::intersect(osg::Vec3d &worldIntersectionPoint, osg::Vec2d const &screenPos)
{
    osgUtil::LineSegmentIntersector::Intersections intersections;

    opencover::VRViewer::instance()->computeIntersections(
        float(screenPos.x()),
        float(screenPos.y()),
        intersections);

    if (intersections.size() > 0)
    {
        // Intersections are sort by depth
        // First object in the set is the nearest
        osgUtil::LineSegmentIntersector::Intersection isect = *intersections.begin();
        worldIntersectionPoint = isect.getWorldIntersectPoint();

        return true;
    }

    return false;
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
void ScreenSpaceHandler::getTransformsFromState(double const *q, osg::Vec3 &T, osg::Quat &R)
{
    double alpha = 1.0e+3;

    T = osg::Vec3(q[0] * alpha, q[1] * alpha, q[2] * alpha);

    double x = q[3];
    double y = q[4];
    double z = q[5];
    double w = 1.0 - x * x - y * y - z * z;

    // Q and -Q both represent the same rotation
    // Choose the one with non-negative scalar part
    if (w < 0.0)
    {
        x = -x;
        y = -y;
        z = -z;
        w = -w;
    }

    w = sqrt(w);

    double len = sqrt(x * x + y * y + z * z + w * w);

    R = osg::Quat(x / len, y / len, z / len, w / len);
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
void ScreenSpaceHandler::errorFunction(double *q, double *x, int /*m*/, int /*n*/, void *context)
{
    ScreenSpaceHandler *self = (ScreenSpaceHandler *)context;

    //
    // Compute transformation matrices from state variables
    //
    osg::Vec3 translation;
    osg::Quat rotation;

    getTransformsFromState(q, translation, rotation);

    osg::Matrixd T = osg::Matrixd::translate(translation);
    osg::Matrixd R = osg::Matrixd::rotate(rotation);

    //
    // Compute new model-view matrix:
    // Rotate, then translate in view-space
    //
    osg::Matrix transform;

    transform = self->currentTransformMatrix;
    transform *= self->camera->getViewMatrix();
    transform *= R;
    transform *= T;
    transform *= self->camera->getProjectionMatrix();
    transform *= self->camera->getViewport()->computeWindowMatrix();

    int numTouchPoints = self->cursors.size();

    int height = self->camera->getViewport()->height();

    CursorMap::iterator it = self->cursors.begin();

    //
    // Transform object-space coordinates into screen-space coordinates
    // and compute the difference to the new target positions
    //

    for (int i = 0; i < numTouchPoints; ++i, ++it)
    {
        Cursor const &cur = it->second;

        osg::Vec3 p = cur.objectPos * transform;

        p.y() = height - p.y();

        x[2 * i + 0] = p.x() - cur.targetPos.x();
        x[2 * i + 1] = p.y() - cur.targetPos.y();
    }

    double alpha = 1.0e+3;

    if (numTouchPoints == 1)
    {
        x[2] = square(q[2]) * alpha * 1.0e+3;
        x[3] = square(q[3]) * alpha;
        x[4] = square(q[4]) * alpha;
        x[5] = square(q[5]) * alpha;
    }
    else if (numTouchPoints == 2)
    {
        x[4] = square(q[3]) * alpha;
        x[5] = square(q[4]) * alpha;
    }
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
void ScreenSpaceHandler::computeTransform(osg::Vec3 &T, osg::Quat &R)
{
    double q[6] = {
        0.0, // Tx
        0.0, // Ty
        0.0, // Tz
        0.0, // Rx
        0.0, // Ry
        0.0, // Rz
    };

    int numTouchPoints = this->cursors.size();

    if (numTouchPoints >= 3)
    {
        double clo[] = { -DBL_MAX, -DBL_MAX, -DBL_MAX, -1.0, -1.0, -1.0 };
        double chi[] = { DBL_MAX, DBL_MAX, DBL_MAX, 1.0, 1.0, 1.0 };

        dlevmar_bc_dif(
            &this->errorFunction,
            q,
            NULL,
            6,
            2 * numTouchPoints,
            clo,
            chi,
            MAX_ITERATIONS,
            NULL,
            NULL,
            NULL,
            NULL,
            this);
    }
    else if (numTouchPoints == 2)
    {
        double clo[] = { -DBL_MAX, -DBL_MAX, -DBL_MAX, -1.0, -1.0, -1.0 };
        double chi[] = { DBL_MAX, DBL_MAX, DBL_MAX, 1.0, 1.0, 1.0 };

        dlevmar_bc_dif(
            &this->errorFunction,
            q,
            NULL,
            6,
            6,
            clo,
            chi,
            MAX_ITERATIONS,
            NULL,
            NULL,
            NULL,
            NULL,
            this);
    }
    else // numTouchPoints == 1
    {
        double clo[] = { -DBL_MAX, -DBL_MAX, -DBL_MAX, -1.0, -1.0, -1.0 };
        double chi[] = { DBL_MAX, DBL_MAX, DBL_MAX, 1.0, 1.0, 1.0 };

        dlevmar_bc_dif(
            &this->errorFunction,
            q,
            NULL,
            6,
            6,
            clo,
            chi,
            MAX_ITERATIONS,
            NULL,
            NULL,
            NULL,
            NULL,
            this);
    }

    this->getTransformsFromState(q, T, R);
}
