/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCSIGNAL_H
#define OSCSIGNAL_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscSignal : public oscObjectBase
{
public:
oscSignal()
{
        OSC_ADD_MEMBER(name, 0);
        OSC_ADD_MEMBER(state, 0);
    };
        const char *getScope(){return "/OpenSCENARIO/RoadNetwork/Signals/SignalsController/Phase";};
    oscString name;
    oscString state;

};

typedef oscObjectVariable<oscSignal *> oscSignalMember;
typedef oscObjectVariableArray<oscSignal *> oscSignalArrayMember;


}

#endif //OSCSIGNAL_H
