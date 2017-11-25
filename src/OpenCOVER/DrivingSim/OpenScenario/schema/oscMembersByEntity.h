/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCMEMBERSBYENTITY_H
#define OSCMEMBERSBYENTITY_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscMembersByEntity : public oscObjectBase
{
public:
oscMembersByEntity()
{
        OSC_ADD_MEMBER(name, 0);
    };
        const char *getScope(){return "/OpenSCENARIO/Entities/Selection/Members";};
    oscString name;

};

typedef oscObjectVariable<oscMembersByEntity *> oscMembersByEntityMember;
typedef oscObjectVariableArray<oscMembersByEntity *> oscMembersByEntityArrayMember;


}

#endif //OSCMEMBERSBYENTITY_H
