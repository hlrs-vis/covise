/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_PARAMETER_LIST_H
#define OSC_PARAMETER_LIST_H
#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscVariables.h>
#include <oscParameters.h>

namespace OpenScenario {

class OpenScenarioBase;

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscParameterList: public oscObjectBase
{
public:
    oscParameterList()
    {
        OSC_ADD_MEMBER(parameter);
    };
    oscParametersMember parameter;
};

typedef oscObjectVariable<oscParameterList *> oscParameterListMember;

}

#endif //OSC_PARAMETER_LIST_H