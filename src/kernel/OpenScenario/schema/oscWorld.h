/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCWORLD_H
#define OSCWORLD_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscWorld : public oscObjectBase
{
public:
oscWorld()
{
        OSC_ADD_MEMBER(x, 0);
        OSC_ADD_MEMBER(y, 0);
        OSC_ADD_MEMBER_OPTIONAL(z, 0);
        OSC_ADD_MEMBER_OPTIONAL(h, 0);
        OSC_ADD_MEMBER_OPTIONAL(p, 0);
        OSC_ADD_MEMBER_OPTIONAL(r, 0);
    };
        const char *getScope(){return "/OSCPosition";};
    oscDouble x;
    oscDouble y;
    oscDouble z;
    oscDouble h;
    oscDouble p;
    oscDouble r;

};

typedef oscObjectVariable<oscWorld *> oscWorldMember;
typedef oscObjectVariableArray<oscWorld *> oscWorldArrayMember;


}

#endif //OSCWORLD_H
