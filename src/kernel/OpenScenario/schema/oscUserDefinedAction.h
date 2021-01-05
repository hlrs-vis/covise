/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCUSERDEFINEDACTION_H
#define OSCUSERDEFINEDACTION_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscStringElement.h"
#include "oscScript.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscUserDefinedAction : public oscObjectBase
{
public:
oscUserDefinedAction()
{
        OSC_OBJECT_ADD_MEMBER(Command, "oscStringElement", 1);
        OSC_OBJECT_ADD_MEMBER(Script, "oscScript", 1);
    };
        const char *getScope(){return "";};
    oscStringElementMember Command;
    oscScriptMember Script;

};

typedef oscObjectVariable<oscUserDefinedAction *> oscUserDefinedActionMember;
typedef oscObjectVariableArray<oscUserDefinedAction *> oscUserDefinedActionArrayMember;


}

#endif //OSCUSERDEFINEDACTION_H
