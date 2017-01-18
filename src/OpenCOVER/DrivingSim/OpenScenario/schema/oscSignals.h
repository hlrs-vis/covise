/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCSIGNALS_H
#define OSCSIGNALS_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"
#include "oscObjectVariableArray.h"

#include "oscVariables.h"
#include "schema/oscSignalController.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscSignals : public oscObjectBase
{
public:
oscSignals()
{
        OSC_ADD_MEMBER(name);
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(SignalController, "oscSignalController");
    };
    oscString name;
    oscSignalControllerArrayMember SignalController;

};

typedef oscObjectVariable<oscSignals *> oscSignalsMember;
typedef oscObjectVariableArray<oscSignals *> oscSignalsArrayMember;


}

#endif //OSCSIGNALS_H
