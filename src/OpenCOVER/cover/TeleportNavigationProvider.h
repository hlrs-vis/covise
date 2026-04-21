/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _TELEPORT_NAVIGATION_PROVIDER_H
#define _TELEPORT_NAVIGATION_PROVIDER_H

#include <OpenVRUI/coMouseButtonInteraction.h>
#include <OpenVRUI/coNavInteraction.h>
#include <osg/MatrixTransform>
#include <osg/ref_ptr>

#include "coVRNavigationProvider.h"

namespace opencover {

class TeleportNavigationProvider : public opencover::coVRNavigationProvider
{
public:
    TeleportNavigationProvider();
    virtual ~TeleportNavigationProvider();

    virtual void setEnabled(bool enabled);

    bool update();

private:
    bool isEnabledAndValid();
    void setVisible(bool visible);

    osg::ref_ptr<osg::Switch> switch_;
    osg::ref_ptr<osg::MatrixTransform> transform;
    osg::ref_ptr<osg::Node> icon;
    float turn_angle = 0.0;
    bool was_visible = false;

    vrui::coNavInteraction interactionPoint;
    vrui::coNavInteraction interactionTurn;
    vrui::coMouseButtonInteraction triggerMouse; ///< trigger interaction
    vrui::coMouseButtonInteraction triggerWheel; ///< adjust turn angle

    osg::Matrix oldHandMatrix;
};

}

#endif
