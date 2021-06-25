/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCPROPERTY_H
#define OSCPROPERTY_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscProperty : public oscObjectBase
{
public:
oscProperty()
{
        OSC_ADD_MEMBER(name, 0);
        OSC_ADD_MEMBER(value, 0);
    };
        const char *getScope(){return "/OSCProperties";};
    oscString name;
    oscString value;

};

typedef oscObjectVariable<oscProperty *> oscPropertyMember;
typedef oscObjectVariableArray<oscProperty *> oscPropertyArrayMember;


}

#endif //OSCPROPERTY_H
