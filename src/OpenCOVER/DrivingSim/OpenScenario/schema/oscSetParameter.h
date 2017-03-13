/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCSETPARAMETER_H
#define OSCSETPARAMETER_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscSetParameter : public oscObjectBase
{
public:
oscSetParameter()
{
        OSC_ADD_MEMBER(name, 0);
        OSC_ADD_MEMBER(value, 0);
    };
        const char *getScope(){return "/OSCParameterAssignment";};
    oscString name;
    oscString value;

};

typedef oscObjectVariable<oscSetParameter *> oscSetParameterMember;
typedef oscObjectVariableArray<oscSetParameter *> oscSetParameterArrayMember;


}

#endif //OSCSETPARAMETER_H
