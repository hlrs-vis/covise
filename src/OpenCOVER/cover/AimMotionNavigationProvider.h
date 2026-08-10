/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _AIMMOTION_NAVIGATION_PROVIDER_H
#define _AIMMOTION_NAVIGATION_PROVIDER_H

#include <OpenVRUI/coMouseButtonInteraction.h>
#include <OpenVRUI/coNavInteraction.h>
#include <osg/MatrixTransform>
#include <osg/ref_ptr>
#include <osg/Switch>

#include "coVRNavigationProvider.h"

namespace opencover
{

class Valuator;

class ValuatorTrigger
{
public:
    ValuatorTrigger(Valuator *valuator);
    int update(double deltaTime);

private:
    Valuator *m_valuator;

    double m_lowerThreshold = 0.3;
    double m_upperThreshold = 0.5;

    double m_repeatDelay = 0.8;
    double m_repeatRate = 0.25;

    int m_isTriggered;
    double m_triggerDuration;
};

class AimMotionNavigationProvider : public opencover::coVRNavigationProvider
{
public:
    AimMotionNavigationProvider();
    virtual ~AimMotionNavigationProvider();

    virtual void setEnabled(bool enabled);

    bool update();

private:
    double measureHeightAboveFloor() const;
    float runningDuration = 0.0;

    vrui::coNavInteraction interactionPoint;
    vrui::coNavInteraction interactionTurn;
    vrui::coMouseButtonInteraction triggerMouse;
    ValuatorTrigger rotateTrigger;

    osg::Matrix oldHandMatrix;

    osg::Vec3 velocity;
    float angleChange;
};

}

#endif
