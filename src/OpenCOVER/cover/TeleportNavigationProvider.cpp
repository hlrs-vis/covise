/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "TeleportNavigationProvider.h"
#include <OpenVRUI/coInteractionManager.h>
#include <cmath>
#include <cover/VRSceneGraph.h>
#include <cover/coIntersection.h>
#include <cover/coVRFileManager.h>
#include <cover/coVRNavigationManager.h>
#include <cover/coVRPluginSupport.h>
#include <osg/MatrixTransform>
#include <osg/PolygonOffset>
#include <osg/Switch>
#include <osg/Vec3>
#include <osg/Matrix>

using namespace opencover;

static osg::Vec3 X = osg::Vec3(1, 0, 0);
static osg::Vec3 UP = osg::Vec3(0, 0, 1);

TeleportNavigationProvider::TeleportNavigationProvider()
    : coVRNavigationProvider("Teleport", nullptr)
    , interactionPoint(vrui::coInteraction::ButtonA, "ProbeMode", vrui::coInteraction::Navigation)
    , interactionTurn(vrui::coInteraction::ButtonB, "ProbeMode", vrui::coInteraction::Navigation)
    , triggerMouse(vrui::coInteraction::ButtonA, "MouseTeleport")
    , triggerWheel(vrui::coInteraction::Wheel, "MouseScroll")
{
    triggerMouse.setGroup(vrui::coInteraction::GroupNavigation);

    switch_ = new osg::Switch;
    cover->getObjectsRoot()->addChild(switch_);

    transform = new osg::MatrixTransform;
    switch_->addChild(transform);

    icon = coVRFileManager::instance()->loadFile("share/covise/icons/teleport_target.glb", nullptr, transform);

    setVisible(false);
}

TeleportNavigationProvider::~TeleportNavigationProvider()
{
}

void TeleportNavigationProvider::setEnabled(bool enabled)
{
    if (isEnabled() == enabled)
        return;

    coVRNavigationProvider::setEnabled(enabled);
    coIntersection::instance()->isectAllNodes(enabled);

    if (enabled)
    {
        vrui::coInteractionManager::the()->registerInteraction(&interactionPoint);
        vrui::coInteractionManager::the()->registerInteraction(&interactionTurn);
        vrui::coInteractionManager::the()->registerInteraction(&triggerMouse);
        vrui::coInteractionManager::the()->registerInteraction(&triggerWheel);
    }
    else
    {
        vrui::coInteractionManager::the()->unregisterInteraction(&interactionPoint);
        vrui::coInteractionManager::the()->unregisterInteraction(&interactionTurn);
        vrui::coInteractionManager::the()->unregisterInteraction(&triggerMouse);
        vrui::coInteractionManager::the()->unregisterInteraction(&triggerWheel);
    }
}

/**
 * Computes where to place the objects, based on where we want to teleport to.
 */
osg::Matrix computeNewObjectsTransform(const osg::Matrix &targetTransform, bool includeViewer = false)
{
    // We'll need this later.
    auto scaleTransform = cover->getObjectsScale()->getMatrix();

    // The `referenceTransform` is location of the viewer's feet in the stage.
    // We use about 1.5m eye height for now.
    auto referenceTransform = osg::Matrix::scale(1000, 1000, 1000) * osg::Matrix::translate(0, 0, -1500);

    // If `includeViewer` is true, the viewer matrix is included in the
    // reference transform, which is useful when navigating with the mouse on a
    // 2D screen, for it places the camera, not the stage origin, into the
    // target location. In VR, we want to align the stage origin with the
    // teleport target (such that head-tracked orientation and offset are
    // ignored), so we do not include the viewer matrix.
    if (includeViewer)
    {
        referenceTransform = referenceTransform * cover->getViewerMat();
    }

    // This equation aligns the "reference transform" (viewer's feet) with the
    // global targetTransform (normal left-to-right matrix order).
    //
    //     ReferenceTransform = newObjectsTransform * scaleTransform * targetTransform
    //
    // We rearrange it to compute only the new objectsTransform:
    //
    //     ~newObjectsTransform * ReferenceTransform = scaleTransform * targetTransform
    //     ~newObjectsTransform = scaleTransform * targetTransform * ~ReferenceTransform
    //     newObjectsTransform = ~(scaleTransform * targetTransform * ~ReferenceTransform)
    //
    // This is implemented below (notice the inverted order of operations
    // because of OSG's matrix structure):

    return osg::Matrix::inverse(osg::Matrix::inverse(referenceTransform) * targetTransform * scaleTransform);
}

bool TeleportNavigationProvider::update()
{
    bool enabledAndValid = isEnabledAndValid();
    setVisible(enabledAndValid);

    if (!enabledAndValid)
    {
        return true;
    }

    // Adjust turn angle based on the mouse wheel
    turn_angle += triggerWheel.getWheelCount() * M_PI / 8;

    // Adjust turn angle based using pointer secondary action and swipe left/right
    if (interactionTurn.wasStarted())
    {
        oldHandMatrix = cover->getPointerMat();
    }
    else if (interactionTurn.isRunning())
    {
        // See how the hand was moved and compute the angle it turned around the Z axis.
        auto handMatrix = cover->getPointerMat();
        auto handMovmentSinceLastFrame = handMatrix * osg::Matrix::inverse(oldHandMatrix);
        auto c = handMovmentSinceLastFrame.getRotate() * X;
        auto angle_change = (c.y() || c.x()) ? atan2(c.y(), c.x()) : 0.f;

        // Turn the target location much more than the sweeping motion from the
        // pointer/hand.
        turn_angle += angle_change * 10.f;

        oldHandMatrix = handMatrix;
    }

    osg::Matrix currentObjectsTransform = cover->getObjectsXform()->getMatrix();

    // Compute the current object's rotation around the Z axis, we want the
    // target angle to be the same as the current angle for `turn_angle = 0`.
    osg::Quat q = currentObjectsTransform.getRotate();
    osg::Vec3 f = q * X;
    float current_angle = f.x() || f.y() ? -atan2(f.y(), f.x()) : 0.0;

    // The target angle around Z is the sum of the current angle and the
    // (relative) turn.
    float target_angle = current_angle + turn_angle;

    osg::Vec3 position = cover->getIntersectionHitPointWorld();

    // The intersection position is relative to the world (stage), so we need
    // to turn it into object coordinates, as we attached the indicator to the object root.
    osg::Matrix objectsRootTransform = osg::computeWorldToLocal(cover->getObjectsRoot()->getParentalNodePaths().at(0));

    auto targetTransform = osg::Matrix::rotate(target_angle, UP) * osg::Matrix::translate(position * objectsRootTransform + UP * 0.001f);

    // Set the indicator transform, scaling and rotating it (because the loaded
    // model is a unit circle and points in the wrong direction).
    float ringSize = 0.4;
    auto ringScale = osg::Matrix::scale(ringSize, ringSize, ringSize);
    transform->setMatrix(ringScale * osg::Matrix::rotate(M_PI / 2, UP) * targetTransform);

    bool mouseTriggered = triggerMouse.wasStopped();
    bool pointerTriggered = interactionPoint.wasStopped();

    if (mouseTriggered || pointerTriggered)
    {
        // Apply the movement transform
        cover->getObjectsXform()->setMatrix(computeNewObjectsTransform(targetTransform, mouseTriggered));

        // Reset turn_angle for the next jump
        turn_angle = 0.f;
    }

    return true;
}

bool TeleportNavigationProvider::isEnabledAndValid()
{
    if (!isEnabled())
        return false;

    if (!cover->getIntersectedNode())
        return false;

    osg::Vec3 normal = cover->getIntersectionHitPointWorldNormal();
    normal.normalize();

    // Get and decompose the object's space transform
    osg::Matrix objectsRootTransform = osg::computeWorldToLocal(cover->getObjectsRoot()->getParentalNodePaths().at(0));
    osg::Vec3f translation, scale;
    osg::Quat rotation, so;
    objectsRootTransform.decompose(translation, rotation, scale, so);

    // Rotate the normal into object space
    normal = rotation * normal;

    // If the normal in object space doesn't roughly point upwards, we don't
    // want to teleport there.
    if (normal * UP < 0.9f)
        return false;

    return true;
}

void TeleportNavigationProvider::setVisible(bool visible)
{
    if (visible)
        switch_->setAllChildrenOn();
    else
        switch_->setAllChildrenOff();
}
