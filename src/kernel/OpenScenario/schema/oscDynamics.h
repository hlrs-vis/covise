/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCDYNAMICS_H
#define OSCDYNAMICS_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscNone.h"
#include "oscLimited.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscDynamics : public oscObjectBase
{
public:
oscDynamics()
{
        OSC_OBJECT_ADD_MEMBER(None, "oscNone", 1);
        OSC_OBJECT_ADD_MEMBER(Limited, "oscLimited", 1);
    };
        const char *getScope(){return "/OSCPrivateAction/Longitudinal/DistanceAction";};
    oscNoneMember None;
    oscLimitedMember Limited;

};

typedef oscObjectVariable<oscDynamics *> oscDynamicsMember;
typedef oscObjectVariableArray<oscDynamics *> oscDynamicsArrayMember;


}

#endif //OSCDYNAMICS_H
