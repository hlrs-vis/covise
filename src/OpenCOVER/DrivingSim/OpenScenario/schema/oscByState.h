/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCBYSTATE_H
#define OSCBYSTATE_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscAtStart.h"
#include "oscAfterTermination.h"
#include "oscCommand.h"
#include "oscSignalState.h"
#include "oscController.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscByState : public oscObjectBase
{
public:
oscByState()
{
        OSC_OBJECT_ADD_MEMBER(AtStart, "oscAtStart", 1);
        OSC_OBJECT_ADD_MEMBER(AfterTermination, "oscAfterTermination", 1);
        OSC_OBJECT_ADD_MEMBER(Command, "oscCommand", 1);
        OSC_OBJECT_ADD_MEMBER(SignalState, "oscSignalState", 1);
        OSC_OBJECT_ADD_MEMBER(Controller, "oscController", 1);
    };
    oscAtStartMember AtStart;
    oscAfterTerminationMember AfterTermination;
    oscCommandMember Command;
    oscSignalStateMember SignalState;
    oscControllerMember Controller;

};

typedef oscObjectVariable<oscByState *> oscByStateMember;
typedef oscObjectVariableArray<oscByState *> oscByStateArrayMember;


}

#endif //OSCBYSTATE_H
