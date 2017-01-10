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
#include "oscParameterList.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscCatalogReference : public oscObjectBase
{
public:
oscCatalogReference()
{
        OSC_ADD_MEMBER(catalog);
        OSC_ADD_MEMBER(name);
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(OSCParameterList, "oscParameterList");
    };
    oscString catalog;
    oscString name;
    oscParameterListMember OSCParameterList;

};

typedef oscObjectVariable<oscCatalogReference *> oscCatalogReferenceMember;
typedef oscObjectVariableArray<oscCatalogReference *> oscCatalogReferenceArrayMember;


}

#endif //OSCCATALOGREFERENCE_H
