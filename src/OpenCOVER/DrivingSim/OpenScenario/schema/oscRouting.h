/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCROUTING_H
#define OSCROUTING_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscFollowRoute.h"
#include "oscFollowTrajectory.h"
#include "oscAcquirePosition.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscRouting : public oscObjectBase
{
public:
oscRouting()
{
        OSC_OBJECT_ADD_MEMBER(FollowRoute, "oscFollowRoute", 1);
        OSC_OBJECT_ADD_MEMBER(FollowTrajectory, "oscFollowTrajectory", 1);
        OSC_OBJECT_ADD_MEMBER(AcquirePosition, "oscAcquirePosition", 1);
    };
        const char *getScope(){return "/OSCPrivateAction";};
    oscFollowRouteMember FollowRoute;
    oscFollowTrajectoryMember FollowTrajectory;
    oscAcquirePositionMember AcquirePosition;

};

typedef oscObjectVariable<oscRouting *> oscRoutingMember;
typedef oscObjectVariableArray<oscRouting *> oscRoutingArrayMember;


}

#endif //OSCROUTING_H
