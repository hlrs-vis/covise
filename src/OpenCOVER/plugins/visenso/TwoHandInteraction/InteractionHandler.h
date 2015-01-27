/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * InteractionHandler.h
 *
 *  Created on: Mar 13, 2012
 *      Author: jw_te
 */

#ifndef INTERACTIONHANDLER_H_
#define INTERACTIONHANDLER_H_

#include "TwoHandInteractionPlugin.h"

namespace TwoHandInteraction
{

class InteractionHandler
{
public:
    InteractionHandler()
    {
    }
    virtual ~InteractionHandler()
    {
    }

    virtual TwoHandInteractionPlugin::InteractionResult
    CalculateInteraction(double frameTime, const TwoHandInteractionPlugin::InteractionStart &interactionStart, bool buttonPressed, const osg::Matrix &handMatrix, const osg::Matrix &secondHandMatrix) = 0;
};
}

#endif /* INTERACTIONHANDLER_H_ */
