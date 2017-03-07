/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCADD_H
#define OSCADD_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscPosition.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscAdd : public oscObjectBase
{
public:
oscAdd()
{
        OSC_OBJECT_ADD_MEMBER(Position, "oscPosition", 0);
    };
        const char *getScope(){return "/OSCGlobalAction/ActionEntity";};
    oscPositionMember Position;

};

typedef oscObjectVariable<oscAdd *> oscAddMember;
typedef oscObjectVariableArray<oscAdd *> oscAddArrayMember;


}

#endif //OSCADD_H
