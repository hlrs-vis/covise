/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCACTION_H
#define OSCACTION_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscGlobalAction.h"
#include "oscUserDefinedAction.h"
#include "oscPrivateAction.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscAction : public oscObjectBase
{
public:
oscAction()
{
        OSC_ADD_MEMBER(name, 0);
        OSC_OBJECT_ADD_MEMBER(Global, "oscGlobalAction", 1);
        OSC_OBJECT_ADD_MEMBER(UserDefined, "oscUserDefinedAction", 1);
        OSC_OBJECT_ADD_MEMBER(Private, "oscPrivateAction", 1);
    };
    oscString name;
    oscGlobalActionMember Global;
    oscUserDefinedActionMember UserDefined;
    oscPrivateActionMember Private;

};

typedef oscObjectVariable<oscAction *> oscActionMember;
typedef oscObjectVariableArray<oscAction *> oscActionArrayMember;


}

#endif //OSCACTION_H
