/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCFOLLOWTRAJECTORY_H
#define OSCFOLLOWTRAJECTORY_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscTrajectory.h"
#include "oscCatalogReference.h"
#include "oscLongitudinalParams.h"
#include "oscLateralParams.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscFollowTrajectory : public oscObjectBase
{
public:
oscFollowTrajectory()
{
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(Trajectory, "oscTrajectory", 0);
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(CatalogReference, "oscCatalogReference", 0);
        OSC_OBJECT_ADD_MEMBER(Longitudinal, "oscLongitudinalParams", 0);
        OSC_OBJECT_ADD_MEMBER(Lateral, "oscLateralParams", 0);
    };
        const char *getScope(){return "/OSCPrivateAction/Routing";};
    oscTrajectoryMember Trajectory;
    oscCatalogReferenceMember CatalogReference;
    oscLongitudinalParamsMember Longitudinal;
    oscLateralParamsMember Lateral;

};

typedef oscObjectVariable<oscFollowTrajectory *> oscFollowTrajectoryMember;
typedef oscObjectVariableArray<oscFollowTrajectory *> oscFollowTrajectoryArrayMember;


}

#endif //OSCFOLLOWTRAJECTORY_H
