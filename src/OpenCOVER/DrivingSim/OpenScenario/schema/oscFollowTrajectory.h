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
#include "oscLongitudinal.h"
#include "oscLateral.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscFollowTrajectory : public oscObjectBase
{
public:
oscFollowTrajectory()
{
        OSC_ADD_MEMBER(name, 0);
        OSC_OBJECT_ADD_MEMBER(Longitudinal, "oscLongitudinal", 0);
        OSC_OBJECT_ADD_MEMBER(Lateral, "oscLateral", 0);
    };
    oscString name;
    oscLongitudinalMember Longitudinal;
    oscLateralMember Lateral;

};

typedef oscObjectVariable<oscFollowTrajectory *> oscFollowTrajectoryMember;
typedef oscObjectVariableArray<oscFollowTrajectory *> oscFollowTrajectoryArrayMember;


}

#endif //OSCFOLLOWTRAJECTORY_H
