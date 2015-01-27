/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * TestInteractionHandler.h
 *
 *  Created on: Mar 13, 2012
 *      Author: jw_te
 */

#ifndef TESTSCALEINTERACTIONHANDLER_H_
#define TESTSCALEINTERACTIONHANDLER_H_

#include "InteractionHandler.h"

namespace TwoHandInteraction
{

class TestScaleInteractionHandler : public InteractionHandler
{
public:
    TestScaleInteractionHandler()
    {
    }
    ~TestScaleInteractionHandler()
    {
    }

    // from InteractionHandler
    TwoHandInteractionPlugin::InteractionResult CalculateInteraction(double frameTime,
                                                                     const TwoHandInteractionPlugin::InteractionStart &interactionStart, bool buttonPressed, const osg::Matrix &handMatrix, const osg::Matrix &secondHandMatrix);

private:
    bool buttonWasPressed;
};
}

#endif /* TESTINTERACTIONHANDLER_H_ */
