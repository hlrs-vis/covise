/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCACQUIREPOSITION_H
#define OSCACQUIREPOSITION_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscPosition.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscAcquirePosition : public oscObjectBase
{
public:
oscAcquirePosition()
{
        OSC_OBJECT_ADD_MEMBER(Position, "oscPosition", 0);
    };
        const char *getScope(){return "/OSCPrivateAction/Routing";};
    oscPositionMember Position;

};

typedef oscObjectVariable<oscAcquirePosition *> oscAcquirePositionMember;
typedef oscObjectVariableArray<oscAcquirePosition *> oscAcquirePositionArrayMember;


}

#endif //OSCACQUIREPOSITION_H
