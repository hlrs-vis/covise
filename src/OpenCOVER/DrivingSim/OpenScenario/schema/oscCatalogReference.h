/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCCATALOGREFERENCE_H
#define OSCCATALOGREFERENCE_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscParameterAssignment.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscCatalogReference : public oscObjectBase
{
public:
oscCatalogReference()
{
        OSC_ADD_MEMBER(catalogName, 0);
        OSC_ADD_MEMBER(entryName, 0);
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(OSCParameterAssignment, "oscParameterAssignment", 0);
    };
        const char *getScope(){return "";};
    oscString catalogName;
    oscString entryName;
    oscParameterAssignmentMember OSCParameterAssignment;

};

typedef oscObjectVariable<oscCatalogReference *> oscCatalogReferenceMember;
typedef oscObjectVariableArray<oscCatalogReference *> oscCatalogReferenceArrayMember;


}

#endif //OSCCATALOGREFERENCE_H
