/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCDRIVER_H
#define OSCDRIVER_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscDescription.h"
#include "oscBehavior.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscDriver : public oscObjectBase
{
public:
oscDriver()
{
        OSC_ADD_MEMBER(name);
        OSC_OBJECT_ADD_MEMBER(Description, "oscDescription");
        OSC_OBJECT_ADD_MEMBER(Behavior, "oscBehavior");
    };
    oscString name;
    oscDescriptionMember Description;
    oscBehaviorMember Behavior;

};

typedef oscObjectVariable<oscDriver *> oscDriverMember;
typedef oscObjectVariableArray<oscDriver *> oscDriverArrayMember;


}

#endif //OSCDRIVER_H
