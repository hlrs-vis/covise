/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCTRAFFIC_H
#define OSCTRAFFIC_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"
#include "oscObjectVariableArray.h"

#include "oscVariables.h"
#include "schema/oscSource.h"
#include "schema/oscSink.h"
#include "schema/oscSwarm.h"
#include "schema/oscJam.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscTraffic : public oscObjectBase
{
public:
oscTraffic()
{
        OSC_OBJECT_ADD_MEMBER(Source, "oscSource");
        OSC_OBJECT_ADD_MEMBER(Sink, "oscSink");
        OSC_OBJECT_ADD_MEMBER(Swarm, "oscSwarm");
        OSC_OBJECT_ADD_MEMBER(Jam, "oscJam");
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
