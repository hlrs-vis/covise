/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCLANECHANGETARGET_H
#define OSCLANECHANGETARGET_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscRelative.h"
#include "oscAbsolute.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscLaneChangeTarget : public oscObjectBase
{
public:
oscLaneChangeTarget()
{
        OSC_OBJECT_ADD_MEMBER(Relative, "oscRelative", 1);
        OSC_OBJECT_ADD_MEMBER(Absolute, "oscAbsolute", 1);
    };
        const char *getScope(){return "/OSCPrivateAction/Lateral/LaneChange";};
    oscRelativeMember Relative;
    oscAbsoluteMember Absolute;

};

typedef oscObjectVariable<oscLaneChangeTarget *> oscLaneChangeTargetMember;
typedef oscObjectVariableArray<oscLaneChangeTarget *> oscLaneChangeTargetArrayMember;


}

#endif //OSCLANECHANGETARGET_H
