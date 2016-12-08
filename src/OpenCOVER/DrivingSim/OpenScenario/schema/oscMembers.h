/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCMEMBERS_H
#define OSCMEMBERS_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"
#include "oscObjectVariableArray.h"

#include "oscVariables.h"
#include "schema/oscByNamedEntity.h"
#include "schema/oscByType.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscMembers : public oscObjectBase
{
public:
oscMembers()
{
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(ByNamedEntity, "oscByNamedEntity");
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(ByType, "oscByType");
    };
    oscByNamedEntityArrayMember ByNamedEntity;
    oscByTypeArrayMember ByType;

};

typedef oscObjectVariable<oscMembers *> oscMembersMember;
typedef oscObjectVariableArray<oscMembers *> oscMembersArrayMember;


}

#endif //OSCMEMBERS_H
