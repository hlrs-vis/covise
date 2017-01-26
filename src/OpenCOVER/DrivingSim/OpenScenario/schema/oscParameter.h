/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCPARAMETER_H
#define OSCPARAMETER_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscParameter : public oscObjectBase
{
public:
oscParameter()
{
        OSC_ADD_MEMBER(name, 0);
        OSC_ADD_MEMBER(value, 0);
        OSC_ADD_MEMBER_OPTIONAL(type, 0);
        OSC_ADD_MEMBER_OPTIONAL(scope, 0);
    };
    oscString name;
    oscString value;
    oscString type;
    oscString scope;

};

typedef oscObjectVariable<oscParameter *> oscParameterMember;
typedef oscObjectVariableArray<oscParameter *> oscParameterArrayMember;


}

#endif //OSCPARAMETER_H
