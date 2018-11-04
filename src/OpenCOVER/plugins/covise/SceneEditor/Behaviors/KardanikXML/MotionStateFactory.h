/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * MotionStateFactory.h
 *
 *  Created on: 27 Jan 2012
 *      Author: jw_te
 */

#ifndef MOTIONSTATEFACTORY_H_
#define MOTIONSTATEFACTORY_H_

#include <memory>
class btMotionState;
class btTransform;

namespace KardanikXML
{
class Body;
class MotionState;

class MotionStateFactory
{
public:
    MotionStateFactory()
    {
    }
    virtual ~MotionStateFactory()
    {
    }
    virtual MotionState *CreateMotionStateForBody(std::shared_ptr<KardanikXML::Body> body, const btTransform &centerOfMass)
    {
        return NULL;
    };
};
}

#endif /* MOTIONSTATEFACTORY_H_ */
