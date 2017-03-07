/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCRELATIVE_H
#define OSCRELATIVE_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscRelative : public oscObjectBase
{
public:
oscRelative()
{
        OSC_ADD_MEMBER(object, 0);
        OSC_ADD_MEMBER(value, 0);
    };
        const char *getScope(){return "/OSCPrivateAction/Lateral/LaneChange/LaneChangeTarget";};
    oscString object;
    oscDouble value;

};

typedef oscObjectVariable<oscRelative *> oscRelativeMember;
typedef oscObjectVariableArray<oscRelative *> oscRelativeArrayMember;


}

#endif //OSCRELATIVE_H
