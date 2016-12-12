/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCENTITIES_H
#define OSCENTITIES_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"
#include "oscObjectVariableArray.h"

#include "oscVariables.h"
#include "schema/oscObject.h"
#include "schema/oscSelection.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscEntities : public oscObjectBase
{
public:
oscEntities()
{
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(Object, "oscObject");
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(Selection, "oscSelection");
    };
    oscObjectArrayMember Object;
    oscSelectionArrayMember Selection;

};

typedef oscObjectVariable<oscEntities *> oscEntitiesMember;
typedef oscObjectVariableArray<oscEntities *> oscEntitiesArrayMember;


}

#endif //OSCENTITIES_H
