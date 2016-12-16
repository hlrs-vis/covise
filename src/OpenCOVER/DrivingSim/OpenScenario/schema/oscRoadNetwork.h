/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCROADNETWORK_H
#define OSCROADNETWORK_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"
#include "oscObjectVariableArray.h"

#include "oscVariables.h"
#include "schema/oscLogics.h"
#include "schema/oscSceneGraph.h"
#include "schema/oscSignals.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscRoadNetwork : public oscObjectBase
{
public:
oscRoadNetwork()
{
        OSC_OBJECT_ADD_MEMBER(Logics, "oscLogics");
        OSC_OBJECT_ADD_MEMBER(SceneGraph, "oscSceneGraph");
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(Signals, "oscSignals");
    };
    oscLogicsMember Logics;
    oscSceneGraphMember SceneGraph;
    oscSignalsMember Signals;

};

typedef oscObjectVariable<oscRoadNetwork *> oscRoadNetworkMember;
typedef oscObjectVariableArray<oscRoadNetwork *> oscRoadNetworkArrayMember;


}

#endif //OSCROADNETWORK_H
