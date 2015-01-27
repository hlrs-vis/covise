/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// Utouch3DHandler.cpp

#include "Utouch3DHandler.h"

#include <cover/coVRConfig.h>
#include <cover/coVRPluginSupport.h>

#include <cover/VRSceneGraph.h>

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
Utouch3DHandler::Utouch3DHandler()
{
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
Utouch3DHandler::~Utouch3DHandler()
{
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
void Utouch3DHandler::init()
{
    this->setState(Possible);
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
void Utouch3DHandler::finish()
{
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
void Utouch3DHandler::onTouchPressed(Touches const & /*touches*/, Touch const &reason)
{
    this->cursors[reason.id] = Cursor(reason.id, reason.x, reason.y);
    this->updateInternalState();

    if (this->getState() == Possible)
    {
        this->setState(Began);
    }
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
void Utouch3DHandler::onTouchesMoved(Touches const &touches)
{
    //
    // TODO:
    // Filter new position if required
    //

    for (Touches::const_iterator it = touches.begin(); it != touches.end(); ++it)
    {
        Touch const &touch = it->second;

        Cursor &cur = this->cursors[touch.id];

        cur.curr = osg::Vec2(touch.x, touch.y);
    }

    // Needs an update
    this->setState(Changed);
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
void Utouch3DHandler::onTouchReleased(Touches const & /*touches*/, Touch const &reason)
{
    this->cursors.erase(reason.id);
    this->updateInternalState();

    // No more touch points?
    if (this->cursors.empty())
    {
        this->setState(Possible);
    }
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
void Utouch3DHandler::onUpdate()
{
    switch (this->cursors.size())
    {
    case 1:
        this->handleOneTouchPoint();
        break;
    case 2:
        this->handleTwoTouchPoints();
        break;
    case 3:
        this->handleThreeTouchPoints();
        break;
    }
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
void Utouch3DHandler::updateInternalState()
{
    if (!this->cursors.empty())
    {
        // Reset cursor positions
        for (CursorMap::iterator it = this->cursors.begin(); it != this->cursors.end(); ++it)
        {
            Cursor &cur = it->second;
            cur.down = cur.curr;
        }

        // Reset root transform
        this->rootTransform = opencover::cover->getObjectsXform()->getMatrix();

        // Get camera frustum parameters
        osg::Matrix matProj = opencover::coVRConfig::instance()->screens[0].camera->getProjectionMatrix();

        matProj.getFrustum(
            this->frustum.left,
            this->frustum.right,
            this->frustum.bottom,
            this->frustum.top,
            this->frustum.znear,
            this->frustum.zfar);
    }
}

//--------------------------------------------------------------------------------------------------
// From Utouch3D
//--------------------------------------------------------------------------------------------------
void Utouch3DHandler::handleOneTouchPoint()
{
    //
    // There's only one touch point in the list
    //
    Cursor const &cur = this->cursors.begin()->second;

    double dist = opencover::cover->getViewerScreenDistance();

    //
    // The center is (0.5, 0.5) only if the viewer position is exactly in the middle
    // of the screen
    // On the table, we are way more nearer to the bottom, thus, the center is calculated
    // dynamically. This way, head-tracking is possible
    //
    double horzFrustum = this->frustum.right + std::abs(this->frustum.left);
    double vertFrustum = this->frustum.top + std::abs(this->frustum.bottom);

    //
    // For stereo, we are interacting on the screen surface and not on the near plane,
    // thus calculate the motion on the surface
    //
    double surfaceRight = (horzFrustum / 2.0f) * dist / this->frustum.znear;
    double surfaceTop = (vertFrustum / 2.0f) * dist / this->frustum.znear;

    double screenCenterX = this->frustum.right / horzFrustum;
    double screenCenterY = this->frustum.top / vertFrustum;

    //
    // Compute x0 and y0 in world coords using the theorem on intersecting lines
    //
    double x0 = (cur.down.x() - screenCenterX) * 2.0 * surfaceRight;
    double y0 = -(cur.down.y() - screenCenterY) * 2.0 * surfaceTop;

    //
    // Compute xc and yc in world coords using the theorem on intersecting lines
    //
    double xc = (cur.curr.x() - screenCenterX) * 2.0 * surfaceRight;
    double yc = -(cur.curr.y() - screenCenterY) * 2.0 * surfaceTop;

    //
    // Compute rotation angles
    //
    double rotX = atan2(x0, dist) - atan2(xc, dist);
    double rotZ = atan2(y0, dist) - atan2(yc, dist);

    //
    // Get the viewer position
    //
    osg::Vec3 viewerPosition = opencover::cover->getViewerMat().getTrans();

    //
    // Compute new transform matrix
    //

    osg::Matrix transform = this->rootTransform
                            * osg::Matrix::translate(-viewerPosition)
                            * osg::Matrix::rotate(rotZ, osg::Vec3(1, 0, 0), 0.0, osg::Vec3(0, 1, 0), rotX, osg::Vec3(0, 0, 1))
                            * osg::Matrix::translate(viewerPosition);

    //
    // Apply the new transformation matrix -- if it's valid
    //
    if (transform.valid())
    {
        opencover::VRSceneGraph::instance()->getTransform()->setMatrix(transform);
    }
}

//--------------------------------------------------------------------------------------------------
// From Utouch3D
//--------------------------------------------------------------------------------------------------
void Utouch3DHandler::handleTwoTouchPoints()
{
    CursorMap::iterator it = this->cursors.begin();

    //
    // Get the only two touch points
    //
    Cursor const &cur1 = (it++)->second; // begin
    Cursor const &cur2 = (it++)->second; // end

    //
    // The center is (0.5, 0.5) only if the viewer position is exactly in the middle
    // of the screen
    // On the table, we are way more nearer to the bottom, thus, the center is calculated
    // dynamically. This way, head-tracking is possible
    //
    double horzFrustum = this->frustum.right + std::abs(this->frustum.left);
    double vertFrustum = this->frustum.top + std::abs(this->frustum.bottom);

    //
    // For stereo, we are interacting on the screen surface and not on the near plane,
    // thus calculate the motion on the surface
    //
    double dist = opencover::cover->getViewerScreenDistance();

    double surfaceScaleX = horzFrustum * dist / this->frustum.znear;
    double surfaceScaleZ = vertFrustum * dist / this->frustum.znear;

    //
    // Compute direction vectors
    //

    osg::Vec2 delta_down = cur2.down - cur1.down;
    osg::Vec2 delta_curr = cur2.curr - cur1.curr;

    //
    // Compute scale factor
    //

    double dist_down = delta_down.length();
    double dist_curr = delta_curr.length();

    // Don't let the scale factor get too large
    // TODO: Select the min value based on the current screen resolution
    dist_down = std::max(0.01, dist_down);

    double scaleFactor = dist_curr / dist_down;

    //
    // Compute difference of current center to reference center
    //

    osg::Vec2 center_down = (cur1.down + cur2.down) * 0.5;
    osg::Vec2 center_curr = (cur1.curr + cur2.curr) * 0.5;

    osg::Vec2 offset = center_curr - center_down;

    //
    // Compute rotation angle
    //

    double alpha = atan2(
        delta_curr.x() * delta_down.y() - delta_curr.y() * delta_down.x(),
        delta_curr.x() * delta_down.x() + delta_curr.y() * delta_down.y());

    //
    // Compute translation on surface in world coords
    //

    double transX = offset.x() * surfaceScaleX; //2.0 * surfaceRight;
    double transZ = offset.y() * surfaceScaleZ; //2.0 * surfaceTop;

    osg::Vec3 trans(transX, 0.0, transZ);

    //
    // Compute rotation/scaling center
    //

    double deltaX = (center_down.x() - 0.5) * surfaceScaleX; //2.0 * surfaceRight;
    double deltaZ = (center_down.y() - 0.5) * surfaceScaleZ; //2.0 * surfaceTop;

    osg::Vec3 delta(deltaX, 0.0, deltaZ);

    //
    // Accumulate transformations
    //

    osg::Matrix transform = this->rootTransform
                            * osg::Matrix::translate(-delta)
                            * osg::Matrix::rotate(alpha, osg::Vec3(0.0, 1.0, 0.0))
                            * osg::Matrix::scale(scaleFactor, scaleFactor, scaleFactor)
                            * osg::Matrix::translate(delta)
                            * osg::Matrix::translate(trans);

    //
    // Apply the new transformation matrix -- if it's valid
    //

    if (transform.valid())
    {
        opencover::VRSceneGraph::instance()->getTransform()->setMatrix(transform);
    }
}

//--------------------------------------------------------------------------------------------------
// From Utouch3D
//--------------------------------------------------------------------------------------------------
void Utouch3DHandler::handleThreeTouchPoints()
{
#if 0

    CursorMap::iterator it = this->cursors.begin();

    //
    // Get the three touch points
    //
    Cursor const& cur1 = (it++)->second;
    Cursor const& cur2 = (it++)->second;
    Cursor const& cur3 = (it++)->second;

#endif
}
