/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCSIGNALS_H
#define OSCSIGNALS_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscSignalsController.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscSignals : public oscObjectBase
{
public:
oscSignals()
{
        OSC_ADD_MEMBER(name, 0);
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(Controller, "oscSignalsController", 0);
    };
        const char *getScope(){return "/OpenSCENARIO/RoadNetwork";};
    oscString name;
    oscSignalsControllerArrayMember Controller;

};

typedef oscObjectVariable<oscSignals *> oscSignalsMember;
typedef oscObjectVariableArray<oscSignals *> oscSignalsArrayMember;


}

#endif //OSCSIGNALS_H
