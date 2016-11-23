/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_ROAD_NETWORK_H
#define OSC_ROAD_NETWORK_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"


#include "oscLogics.h"
#include "oscSignals.h"
#include "oscSceneGraph.h"

namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscRoadNetwork: public oscObjectBase
{
public:
    oscRoadNetwork()
    {
        OSC_OBJECT_ADD_MEMBER(Logics, "oscLogics");
        OSC_OBJECT_ADD_MEMBER(SceneGraph, "oscSceneGraph");
        OSC_OBJECT_ADD_MEMBER(Signals, "oscSignals");
    };

    oscLogicsMember Logics;
    oscSceneGraphMember SceneGraph;
    oscSignalsMember Signals;
};

typedef oscObjectVariable<oscRoadNetwork *> oscRoadNetworkMember;

}

#endif //OSC_ROAD_NETWORK_H
