/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCTRAFFIC_H
#define OSCTRAFFIC_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscSource.h"
#include "oscSink.h"
#include "oscSwarm.h"
#include "oscJam.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscTraffic : public oscObjectBase
{
public:
oscTraffic()
{
        OSC_OBJECT_ADD_MEMBER(Source, "oscSource", 1);
        OSC_OBJECT_ADD_MEMBER(Sink, "oscSink", 1);
        OSC_OBJECT_ADD_MEMBER(Swarm, "oscSwarm", 1);
        OSC_OBJECT_ADD_MEMBER(Jam, "oscJam", 1);
    };
    oscSourceMember Source;
    oscSinkMember Sink;
    oscSwarmMember Swarm;
    oscJamMember Jam;

};

typedef oscObjectVariable<oscTraffic *> oscTrafficMember;
typedef oscObjectVariableArray<oscTraffic *> oscTrafficArrayMember;


}

#endif //OSCTRAFFIC_H
