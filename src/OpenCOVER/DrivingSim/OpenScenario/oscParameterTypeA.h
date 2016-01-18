/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_PARAMETER_TYPE_A_H
#define OSC_PARAMETER_TYPE_A_H

#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>

#include <oscVariables.h>


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscParameterTypeA: public oscObjectBase
{
public:
    oscParameterTypeA()
    {
        OSC_ADD_MEMBER(name);
        OSC_ADD_MEMBER(value);
    };

    oscString name;
    oscString value;
};

typedef oscObjectVariable<oscParameterTypeA *> oscParameterTypeAMember;

}

#endif //OSC_PARAMETER_TYPE_A_H
