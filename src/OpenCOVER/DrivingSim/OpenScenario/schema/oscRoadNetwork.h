/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCROADNETWORK_H
#define OSCROADNETWORK_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscFile.h"
#include "oscFile.h"
#include "oscSignals.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscRoadNetwork : public oscObjectBase
{
public:
oscRoadNetwork()
{
        OSC_OBJECT_ADD_MEMBER(Logics, "oscFile", 0);
        OSC_OBJECT_ADD_MEMBER(SceneGraph, "oscFile", 0);
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(Signals, "oscSignals", 0);
    };
        const char *getScope(){return "/OpenSCENARIO";};
    oscFileMember Logics;
    oscFileMember SceneGraph;
    oscSignalsMember Signals;

};

typedef oscObjectVariable<oscRoadNetwork *> oscRoadNetworkMember;
typedef oscObjectVariableArray<oscRoadNetwork *> oscRoadNetworkArrayMember;


}

#endif //OSCROADNETWORK_H
