/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCUSERDEFINEDACTION_H
#define OSCUSERDEFINEDACTION_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "schema/xsd:string.h"
#include "schema/oscScript.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscUserDefinedAction : public oscObjectBase
{
public:
    oscUserDefinedAction()
    {
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(Command, "xsd:string");
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(Script, "oscScript");
    };
    xsd:stringMember Command;
    oscScriptMember Script;

};

typedef oscObjectVariable<oscUserDefinedAction *> oscUserDefinedActionMember;


}

#endif //OSCUSERDEFINEDACTION_H
