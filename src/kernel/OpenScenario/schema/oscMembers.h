/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCMEMBERS_H
#define OSCMEMBERS_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscMembersByEntity.h"
#include "oscByType.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscMembers : public oscObjectBase
{
public:
oscMembers()
{
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(ByEntity, "oscMembersByEntity", 1);
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(ByType, "oscByType", 1);
    };
        const char *getScope(){return "/OpenSCENARIO/Entities/Selection";};
    oscMembersByEntityArrayMember ByEntity;
    oscByTypeArrayMember ByType;

};

typedef oscObjectVariable<oscMembers *> oscMembersMember;
typedef oscObjectVariableArray<oscMembers *> oscMembersArrayMember;


}

#endif //OSCMEMBERS_H
