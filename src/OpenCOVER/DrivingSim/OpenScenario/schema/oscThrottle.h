/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCTHROTTLE_H
#define OSCTHROTTLE_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscThrottle : public oscObjectBase
{
public:
oscThrottle()
{
        OSC_ADD_MEMBER(value, 0);
        OSC_ADD_MEMBER(active, 0);
    };
        const char *getScope(){return "/OSCPrivateAction/ActionController/Override";};
    oscDouble value;
    oscBool active;

};

typedef oscObjectVariable<oscThrottle *> oscThrottleMember;
typedef oscObjectVariableArray<oscThrottle *> oscThrottleArrayMember;


}

#endif //OSCTHROTTLE_H
