/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCACTIONENTITY_H
#define OSCACTIONENTITY_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscAdd.h"
#include "oscDelete.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscActionEntity : public oscObjectBase
{
public:
oscActionEntity()
{
        OSC_ADD_MEMBER(name, 0);
        OSC_OBJECT_ADD_MEMBER(Add, "oscAdd", 1);
        OSC_OBJECT_ADD_MEMBER(Delete, "oscDelete", 1);
    };
        const char *getScope(){return "/OSCGlobalAction";};
    oscString name;
    oscAddMember Add;
    oscDeleteMember Delete;

};

typedef oscObjectVariable<oscActionEntity *> oscActionEntityMember;
typedef oscObjectVariableArray<oscActionEntity *> oscActionEntityArrayMember;


}

#endif //OSCACTIONENTITY_H
