/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCSCRIPT_H
#define OSCSCRIPT_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscParameterAssignment.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT Enum_Script_executionType : public oscEnumType
{
public:
static Enum_Script_executionType *instance();
    private:
		Enum_Script_executionType();
	    static Enum_Script_executionType *inst; 
};
class OPENSCENARIOEXPORT oscScript : public oscObjectBase
{
public:
oscScript()
{
        OSC_ADD_MEMBER(name, 0);
        OSC_ADD_MEMBER(file, 0);
        OSC_ADD_MEMBER(execution, 0);
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(OSCParameterAssignment, "oscParameterAssignment", 0);
        execution.enumType = Enum_Script_executionType::instance();
    };
        const char *getScope(){return "/OSCUserDefinedAction";};
    oscString name;
    oscString file;
    oscEnum execution;
    oscParameterAssignmentMember OSCParameterAssignment;

    enum Enum_Script_execution
    {
single,
continuous,

    };

};

typedef oscObjectVariable<oscScript *> oscScriptMember;
typedef oscObjectVariableArray<oscScript *> oscScriptArrayMember;


}

#endif //OSCSCRIPT_H
