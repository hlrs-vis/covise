/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_PARAMETER_H
#define OSC_PARAMETER_H
#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscVariables.h>

namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscParameter: public oscObjectBase
{
public:
    oscParameter()
    {
        OSC_ADD_MEMBER(name);
        OSC_ADD_MEMBER(value);
        OSC_ADD_MEMBER(type);
    };
    oscString name;
    oscString value;
    oscString type;
};

typedef oscObjectVariable<oscParameter *> oscParameterMember;

}

#endif //OSC_PARAMETER_H
