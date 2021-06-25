/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCSPEED_H
#define OSCSPEED_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscSpeedDynamics.h"
#include "oscTarget.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscSpeed : public oscObjectBase
{
public:
oscSpeed()
{
        OSC_OBJECT_ADD_MEMBER(Dynamics, "oscSpeedDynamics", 0);
        OSC_OBJECT_ADD_MEMBER(Target, "oscTarget", 0);
    };
        const char *getScope(){return "/OSCPrivateAction/Longitudinal";};
    oscSpeedDynamicsMember Dynamics;
    oscTargetMember Target;

};

typedef oscObjectVariable<oscSpeed *> oscSpeedMember;
typedef oscObjectVariableArray<oscSpeed *> oscSpeedArrayMember;


}

#endif //OSCSPEED_H
