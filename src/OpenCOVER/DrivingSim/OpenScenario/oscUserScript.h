/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_USER_SCRIPT_H
#define OSC_USER_SCRIPT_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "oscFile.h"
#include "oscParameterListTypeB.h"


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
        OSC_OBJECT_ADD_MEMBER(file, "oscFile");
        OSC_OBJECT_ADD_MEMBER(parameterList, "oscParameterListTypeB");
        OSC_ADD_MEMBER(execution);

        execution.enumType = executionType::instance();
    };

    oscString name;
    oscFileMember file;
    oscEnum execution;
    oscParameterListTypeBArrayMember parameterList;

    enum execution
    {
        fireAndForget,
        waitForTermination,
    };
};

typedef oscObjectVariable<oscUserScript *> oscUserScriptMember;

}

#endif //OSC_USER_SCRIPT_H
