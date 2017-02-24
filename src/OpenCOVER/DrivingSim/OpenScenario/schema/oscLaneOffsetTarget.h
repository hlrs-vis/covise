/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCLANEOFFSETTARGET_H
#define OSCLANEOFFSETTARGET_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscRelative.h"
#include "oscAbsolute.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscLaneOffsetTarget : public oscObjectBase
{
public:
oscLaneOffsetTarget()
{
        OSC_OBJECT_ADD_MEMBER(Relative, "oscRelative", 1);
        OSC_OBJECT_ADD_MEMBER(Absolute, "oscAbsolute", 1);
    };
        const char *getScope(){return "/OSCPrivateAction/Lateral/LaneOffset";};
    oscRelativeMember Relative;
    oscAbsoluteMember Absolute;

};

typedef oscObjectVariable<oscLaneOffsetTarget *> oscLaneOffsetTargetMember;
typedef oscObjectVariableArray<oscLaneOffsetTarget *> oscLaneOffsetTargetArrayMember;


}

#endif //OSCLANEOFFSETTARGET_H
