/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCSCRIPT_H
#define OSCSCRIPT_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "schema/oscParameterList.h"

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
        OSC_ADD_MEMBER(name);
        OSC_ADD_MEMBER(file);
        OSC_ADD_MEMBER(execution);
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(OSCParameterList, "oscParameterList");
    };
    oscString name;
    oscString file;
    oscEnum execution;
    oscParameterListMember OSCParameterList;

    enum Enum_Script_execution
    {
single,
continuous,

    };

};

typedef oscObjectVariable<oscScript *> oscScriptMember;


}

#endif //OSCSCRIPT_H
