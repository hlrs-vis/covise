/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCEVENTCONDITIONS_H
#define OSCEVENTCONDITIONS_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "schema/oscStart.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscEventConditions : public oscObjectBase
{
public:
    oscEventConditions()
    {
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(Start, "oscStart");
    };
    oscStartMember Start;

};

typedef oscObjectVariable<oscEventConditions *> oscEventConditionsMember;


}

#endif //OSCEVENTCONDITIONS_H
