/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCOBJECT_H
#define OSCOBJECT_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscCatalogReference.h"
#include "oscCtrl.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscObject : public oscObjectBase
{
public:
oscObject()
{
        OSC_ADD_MEMBER(name);
        OSC_OBJECT_ADD_MEMBER(CatalogReference, "oscCatalogReference");
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(Ctrl, "oscCtrl");
    };
    oscString name;
    oscCatalogReferenceMember CatalogReference;
    oscCtrlMember Ctrl;

};

typedef oscObjectVariable<oscObject *> oscObjectMember;
typedef oscObjectVariableArray<oscObject *> oscObjectArrayMember;


}

#endif //OSCOBJECT_H
