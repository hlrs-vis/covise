/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_USER_DEFINED_ACTION_H
#define OSC_USER_DEFINED_ACTION_H

#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>

#include <oscVariables.h>
#include <oscParameterListTypeB.h>


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscUserDefinedAction: public oscObjectBase
{
public:
    oscUserDefinedAction()
    {
        OSC_ADD_MEMBER(name);
        OSC_OBJECT_ADD_MEMBER(parameterList, "oscParameterListTypeB");
    };

    oscString name;
    oscParameterListTypeBArrayMember parameterList;
};

typedef oscObjectVariable<oscUserDefinedAction *> oscUserDefinedActionMember;

}

#endif //OSC_USER_DEFINED_ACTION_H
