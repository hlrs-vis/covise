/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCLONGITUDINALPARAMS_H
#define OSCLONGITUDINALPARAMS_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscNone.h"
#include "oscTiming.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscLongitudinalParams : public oscObjectBase
{
public:
oscLongitudinalParams()
{
        OSC_OBJECT_ADD_MEMBER(None, "oscNone", 1);
        OSC_OBJECT_ADD_MEMBER(Timing, "oscTiming", 1);
    };
        const char *getScope(){return "/OSCPrivateAction/Routing/FollowTrajectory";};
    oscNoneMember None;
    oscTimingMember Timing;

};

typedef oscObjectVariable<oscLongitudinalParams *> oscLongitudinalParamsMember;
typedef oscObjectVariableArray<oscLongitudinalParams *> oscLongitudinalParamsArrayMember;


}

#endif //OSCLONGITUDINALPARAMS_H
