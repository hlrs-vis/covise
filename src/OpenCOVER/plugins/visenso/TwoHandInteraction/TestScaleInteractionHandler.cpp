/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * TestInteractionHandler.cpp
 *
 *  Created on: Mar 13, 2012
 *      Author: jw_te
 */

#include "TestScaleInteractionHandler.h"

#include <math.h>

namespace TwoHandInteraction
{

TwoHandInteractionPlugin::InteractionResult
TestScaleInteractionHandler::CalculateInteraction(double frameTime, const TwoHandInteractionPlugin::InteractionStart &interactionStart, bool buttonPressed, const osg::Matrix &handMatrix,
                                                  const osg::Matrix &secondHandMatrix)
{
    TwoHandInteractionPlugin::InteractionResult result(interactionStart);

    //result.ScalingMatrix.makeScale(osg::Vec3(5000,5000,5000);

    buttonWasPressed = buttonPressed;
    return result;
}
}
