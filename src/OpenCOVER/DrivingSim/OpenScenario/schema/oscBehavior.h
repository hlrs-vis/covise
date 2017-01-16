/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCBEHAVIOR_H
#define OSCBEHAVIOR_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"
#include "oscObjectVariableArray.h"

#include "oscVariables.h"
#include "schema/oscParameter.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscBehavior : public oscObjectBase
{
public:
oscBehavior()
{
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(Parameter, "oscParameter");
    };
    oscParameterArrayMember Parameter;

};

typedef oscObjectVariable<oscBehavior *> oscBehaviorMember;
typedef oscObjectVariableArray<oscBehavior *> oscBehaviorArrayMember;


}

#endif //OSCBEHAVIOR_H
