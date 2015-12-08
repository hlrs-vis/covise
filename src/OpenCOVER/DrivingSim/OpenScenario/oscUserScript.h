/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_USER_SCRIPT_H
#define OSC_USER_SCRIPT_H
#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscVariables.h>
#include <oscFile.h>
#include <oscParameter.h>

namespace OpenScenario {

class OPENSCENARIOEXPORT executionType: public oscEnumType
{
public:
    static executionType *instance(); 
private:
    executionType();
    static executionType *inst;
};

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscUserScript: public oscObjectBase
{
public:
    oscUserScript()
    {
        OSC_ADD_MEMBER(name);
        OSC_OBJECT_ADD_MEMBER(file,"oscFile");
        OSC_OBJECT_ADD_MEMBER(parameter,"oscParameter");
		OSC_ADD_MEMBER(execution);
		execution.enumType = executionType::instance();  
    };
    oscString name;
    oscFileMember file;
    enum execution
    {
        fireAndForget,
        waitForTermination,
    };
    oscEnum execution;
    oscParameterMember parameter;
    
};

typedef oscObjectVariable<oscUserScript *> oscUserScriptMember;

}

#endif //OSC_USER_SCRIPT_H
