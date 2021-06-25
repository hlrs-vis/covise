/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCDATE_H
#define OSCDATE_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscDate : public oscObjectBase
{
public:
oscDate()
{
        OSC_ADD_MEMBER(day, 0);
        OSC_ADD_MEMBER(month, 0);
        OSC_ADD_MEMBER(year, 0);
    };
        const char *getScope(){return "/OSCCondition/ByValue/TimeOfDay";};
    oscUInt day;
    oscUInt month;
    oscUInt year;

};

typedef oscObjectVariable<oscDate *> oscDateMember;
typedef oscObjectVariableArray<oscDate *> oscDateArrayMember;


}

#endif //OSCDATE_H
