/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCOFFROAD_H
#define OSCOFFROAD_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscOffroad : public oscObjectBase
{
public:
oscOffroad()
{
        OSC_ADD_MEMBER(duration, 0);
    };
        const char *getScope(){return "/OSCCondition/ByEntity/EntityCondition";};
    oscDouble duration;

};

typedef oscObjectVariable<oscOffroad *> oscOffroadMember;
typedef oscObjectVariableArray<oscOffroad *> oscOffroadArrayMember;


}

#endif //OSCOFFROAD_H
