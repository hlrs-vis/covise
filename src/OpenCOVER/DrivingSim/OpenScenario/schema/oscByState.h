/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCBYSTATE_H
#define OSCBYSTATE_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"
#include "oscObjectVariableArray.h"

#include "oscVariables.h"
#include "schema/oscAtStart.h"
#include "schema/oscAfterTermination.h"
#include "schema/oscCommand.h"
#include "schema/oscSignalState.h"
#include "schema/oscController.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscByState : public oscObjectBase
{
public:
oscByState()
{
        OSC_OBJECT_ADD_MEMBER(AtStart, "oscAtStart");
        OSC_OBJECT_ADD_MEMBER(AfterTermination, "oscAfterTermination");
        OSC_OBJECT_ADD_MEMBER(Command, "oscCommand");
        OSC_OBJECT_ADD_MEMBER(SignalState, "oscSignalState");
        OSC_OBJECT_ADD_MEMBER(Controller, "oscController");
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
