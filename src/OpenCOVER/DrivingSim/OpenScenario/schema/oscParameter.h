/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCPARAMETER_H
#define OSCPARAMETER_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscParameter : public oscObjectBase
{
public:
    oscParameter()
    {
        OSC_ADD_MEMBER(name);
        OSC_ADD_MEMBER(value);
        OSC_ADD_MEMBER_OPTIONAL(type);
        OSC_ADD_MEMBER_OPTIONAL(scope);
    };
    oscString name;
    oscString value;
    oscString type;
    oscString scope;

};

typedef oscObjectVariable<oscParameter *> oscParameterMember;


}

#endif //OSCPARAMETER_H
