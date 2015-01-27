/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cover/coVRPluginSupport.h>
#include <cover/coVRMSController.h>
#include "VolumeOfInterestInteractor.h"

#include <math.h>
#include <osg/MatrixTransform>
#include <osg/PositionAttitudeTransform>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Material>
#include <osg/ShapeDrawable>
#include <osg/LineWidth>

#include <util/coRestraint.h>
#include <config/CoviseConfig.h>

#include <cover/VRSceneGraph.h>
#include <cover/coVRAnimationManager.h>

using namespace osg;

VolumeOfInterestInteractor::VolumeOfInterestInteractor(vrui::coInteraction::InteractionType type,
                                                       const char *name, vrui::coInteraction::InteractionPriority priority = Medium)
    : coTrackerButtonInteraction(type, name, priority)
    , m_interactionFinished(NULL)
{
}

VolumeOfInterestInteractor::~VolumeOfInterestInteractor()
{
}

void VolumeOfInterestInteractor::stopInteraction()
{
    if (m_interactionFinished)
        m_interactionFinished();
}

void VolumeOfInterestInteractor::registerInteractionFinishedCallback(void (*interactionFinished)())
{
    m_interactionFinished = interactionFinished;
}

void VolumeOfInterestInteractor::unregisterInteractionFinishedCallback()
{
    m_interactionFinished = NULL;
}
