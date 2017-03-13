/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCCURRENT_H
#define OSCCURRENT_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscCurrent : public oscObjectBase
{
public:
oscCurrent()
{
        OSC_ADD_MEMBER(object, 0);
    };
        const char *getScope(){return "/OSCPosition/Route/PositionRoute";};
    oscString object;

};

typedef oscObjectVariable<oscCurrent *> oscCurrentMember;
typedef oscObjectVariableArray<oscCurrent *> oscCurrentArrayMember;


}

#endif //OSCCURRENT_H
