/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "AimMotionNavigationProvider.h"
#include <input/input.h>
#include <input/inputdevice.h>
#include <input/valuator.h>
#include <OpenVRUI/coInteractionManager.h>
#include <algorithm>
#include <cmath>
#include <cover/VRSceneGraph.h>
#include <cover/coIntersection.h>
#include <cover/coVRFileManager.h>
#include <cover/coVRNavigationManager.h>
#include <cover/coVRPluginSupport.h>
#include <osg/LineSegment>
#include <osg/MatrixTransform>
#include <osg/PolygonOffset>
#include <osg/Switch>
#include <osg/Vec3>
#include <osg/Matrix>

using namespace opencover;

static osg::Vec3 X = osg::Vec3(1, 0, 0);
static osg::Vec3 UP = osg::Vec3(0, 0, 1);

ValuatorTrigger::ValuatorTrigger(Valuator *valuator)
    : m_valuator(valuator)
{
}

int ValuatorTrigger::update(double deltaTime)
{
    if (!m_valuator)
        return 0;

    double value = m_valuator->getValue();
    if (m_isTriggered != 0 && abs(value) < m_lowerThreshold)
    {
        m_isTriggered = 0;
        return 0;
    }
    else if (m_isTriggered != 1 && value > m_upperThreshold)
    {
        m_isTriggered = 1;
        m_triggerDuration = m_repeatRate - m_repeatDelay;
        return 1;
    }
    else if (m_isTriggered != -1 && value < -m_upperThreshold)
    {
        m_isTriggered = -1;
        m_triggerDuration = m_repeatRate - m_repeatDelay;
        return -1;
    }

    if (m_isTriggered != 0)
    {
        // key-repeat
        m_triggerDuration += deltaTime;
        if (m_triggerDuration > m_repeatRate)
        {
            m_triggerDuration = 0.0;
            return m_isTriggered;
        }
    }

    return 0;
}

AimMotionNavigationProvider::AimMotionNavigationProvider()
    : coVRNavigationProvider("AimMotion", nullptr)
    , interactionPoint(vrui::coInteraction::ButtonA, "ProbeMode", vrui::coInteraction::Navigation)
    , interactionTurn(vrui::coInteraction::ButtonB, "ProbeMode", vrui::coInteraction::Navigation)
    , interactionReverse(vrui::coInteraction::ButtonC, "ProbeMode", vrui::coInteraction::Navigation)
    , triggerMouse(vrui::coInteraction::ButtonA, "MouseAimMotion")
    , rotateTrigger(Input::instance()->getValuator("RightJoyX"))
{
    triggerMouse.setGroup(vrui::coInteraction::GroupNavigation);
}

AimMotionNavigationProvider::~AimMotionNavigationProvider()
{
}

void AimMotionNavigationProvider::setEnabled(bool enabled)
{
    if (isEnabled() == enabled)
        return;

    coVRNavigationProvider::setEnabled(enabled);
    coIntersection::instance()->isectAllNodes(enabled);

    if (enabled)
    {
        vrui::coInteractionManager::the()->registerInteraction(&interactionPoint);
        vrui::coInteractionManager::the()->registerInteraction(&interactionReverse);
        vrui::coInteractionManager::the()->registerInteraction(&interactionTurn);
        vrui::coInteractionManager::the()->registerInteraction(&triggerMouse);
    }
    else
    {
        vrui::coInteractionManager::the()->unregisterInteraction(&interactionPoint);
        vrui::coInteractionManager::the()->unregisterInteraction(&interactionReverse);
        vrui::coInteractionManager::the()->unregisterInteraction(&interactionTurn);
        vrui::coInteractionManager::the()->unregisterInteraction(&triggerMouse);
    }
}

template <typename T>
inline T lerp(T a, T b, float f) { return a + (b - a) * f; }

double AimMotionNavigationProvider::measureHeightAboveFloor() const
{
    float floorHeight = VRSceneGraph::instance()->floorHeight();

    osg::Matrix viewer = cover->getViewerMat();
    osg::Vec3 pos = viewer.getTrans();

    float castHeight = 1e8;
    osg::Vec3 p0(pos[0], pos[1], pos[2]);
    osg::Vec3 q0(pos[0], pos[1], pos[2] - castHeight);

    osg::ref_ptr<osgUtil::IntersectorGroup> igroup = new osgUtil::IntersectorGroup;
    osg::ref_ptr<osgUtil::LineSegmentIntersector> intersector = coIntersection::instance()->newIntersector(p0, q0);
    igroup->addIntersector(intersector);

    osgUtil::IntersectionVisitor visitor(igroup);
    visitor.setTraversalMask(Isect::Walk);
    VRSceneGraph::instance()->getTransform()->accept(visitor);

    if (!intersector->containsIntersections())
        return 0.0;
    return -(intersector->getFirstIntersection().getWorldIntersectPoint()[2] - floorHeight);
}

bool AimMotionNavigationProvider::update()
{
    if (!isEnabled())
        return true;

    // Adjust turn angle using the joystick
    // auto valuator = Input::instance()->getValuator("CaveJoyX");
    // if (valuator)
    // {
    //     turn_angle -= valuator->getValue() * cover->frameDuration() * 10.f;
    // }

    auto pointer = cover->getPointerMat();

    auto deltaTime = cover->frameDuration();
    osg::Matrix currentObjectsTransform = cover->getObjectsXform()->getMatrix();

    auto dir = pointer.getRotate() * osg::Vec3(0, 1, 0);
    float angleChangeTarget = 0;

    float driveSpeed = coVRNavigationManager::instance()->getDriveSpeed() * 1000.f;
    float height = measureHeightAboveFloor();
    if (height == 0.0)
        height = std::abs(currentObjectsTransform.getTrans().z());
    float maxSpeed = std::max(height * 3.f, driveSpeed);

    if (interactionTurn.isRunning())
    {
        runningDuration += deltaTime;

        // Adjust turn angle based using pointer secondary action and swipe left/right
        auto pointerAngle = atan2(dir.x(), dir.y());
        float angleStrength = 1.0 - abs(dir * UP);
        angleChangeTarget = pointerAngle * angleStrength;
    }
    else if (interactionPoint.isRunning())
    {
        runningDuration += deltaTime;

        auto targetVelocity = dir;
        velocity = lerp(velocity, targetVelocity, deltaTime * 5.0);
    }
    else if (interactionReverse.isRunning())
    {
        runningDuration += deltaTime;

        auto targetVelocity = -dir;
        velocity = lerp(velocity, targetVelocity, deltaTime * 5.0);
    }
    else
    {
        runningDuration = 0.f;
    }

    // slow down
    velocity *= (1.0 - deltaTime * 2.0);
    if (abs(angleChangeTarget) > abs(angleChange))
        angleChange = lerp(angleChange, angleChangeTarget, deltaTime * 2.0);
    else
        angleChange = lerp(angleChange, angleChangeTarget, deltaTime * 8.0);

    // stop properly
    if (!interactionPoint.isRunning() && velocity.length2() < 0.01)
        velocity.set(0, 0, 0);

    // Apply the movement transform
    cover->getObjectsXform()->setMatrix(currentObjectsTransform * osg::Matrix::rotate(angleChange * deltaTime + rotateTrigger.update(deltaTime) * M_PI / 4.0, UP) * osg::Matrix::translate(-velocity * deltaTime * maxSpeed));

    return true;
}
