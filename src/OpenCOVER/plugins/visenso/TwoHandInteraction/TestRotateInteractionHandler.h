/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * TestRotateInteractionHandler.h
 *
 *  Created on: Mar 13, 2012
 *      Author: jw_te
 */

#ifndef TESTROTATEINTERACTIONHANDLER_H_
#define TESTROTATEINTERACTIONHANDLER_H_

#include "InteractionHandler.h"

namespace TwoHandInteraction
{

class TestRotateInteractionHandler : public InteractionHandler
{
public:
    TestRotateInteractionHandler()
    {
    }
    ~TestRotateInteractionHandler()
    {
    }

    // from InteractionHandler
    TwoHandInteractionPlugin::InteractionResult CalculateInteraction(double frameTime, const TwoHandInteractionPlugin::InteractionStart &interactionStart, bool buttonPressed, const osg::Matrix &handMatrix, const osg::Matrix &secondHandMatrix);

private:
    bool buttonWasPressed;
    //bool buttonReleased;

    osg::Vec3 handRight;
    osg::Vec3 handLeft;

    osg::Vec3 old_delta;
    osg::Vec3 delta;

    osg::Matrix handmat;
    osg::Matrix old_handmat;
    osg::Matrix old_handmatinvert;

    osg::Matrix orig_transrotm, old_transrotm;
    osg::Matrix orig_scalem, old_scalem;

    //osg::ref_ptr<osg::MatrixTransform> m_MiddleHandIndicator;
};
}

#endif /* TESTROTATEINTERACTIONHANDLER_H_ */
