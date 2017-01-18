/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCSTANDSTILL_H
#define OSCSTANDSTILL_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscStandStill : public oscObjectBase
{
public:
oscStandStill()
{
        OSC_ADD_MEMBER(duration);
    };
    oscDouble duration;

};

typedef oscObjectVariable<oscStandStill *> oscStandStillMember;
typedef oscObjectVariableArray<oscStandStill *> oscStandStillArrayMember;


}

#endif //OSCSTANDSTILL_H
