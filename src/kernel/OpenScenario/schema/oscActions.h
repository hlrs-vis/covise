/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCACTIONS_H
#define OSCACTIONS_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscGlobalAction.h"
#include "oscUserDefinedAction.h"
#include "oscPrivate.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscActions : public oscObjectBase
{
public:
oscActions()
{
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(Global, "oscGlobalAction", 0);
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(UserDefined, "oscUserDefinedAction", 0);
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(Private, "oscPrivate", 0);
    };
        const char *getScope(){return "/OpenSCENARIO/Storyboard/Init";};
    oscGlobalActionArrayMember Global;
    oscUserDefinedActionArrayMember UserDefined;
    oscPrivateArrayMember Private;

};

typedef oscObjectVariable<oscActions *> oscActionsMember;
typedef oscObjectVariableArray<oscActions *> oscActionsArrayMember;


}

#endif //OSCACTIONS_H
