/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCSWARM_H
#define OSCSWARM_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscCentralObject.h"
#include "oscTrafficDefinition.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscSwarm : public oscObjectBase
{
public:
oscSwarm()
{
        OSC_ADD_MEMBER(semiMajorAxis, 0);
        OSC_ADD_MEMBER(semiMinorAxis, 0);
        OSC_ADD_MEMBER(innerRadius, 0);
        OSC_ADD_MEMBER(offset, 0);
        OSC_OBJECT_ADD_MEMBER(CentralObject, "oscCentralObject", 0);
        OSC_OBJECT_ADD_MEMBER(TrafficDefinition, "oscTrafficDefinition", 0);
    };
        const char *getScope(){return "/OSCGlobalAction/Traffic";};
    oscDouble semiMajorAxis;
    oscDouble semiMinorAxis;
    oscDouble innerRadius;
    oscDouble offset;
    oscCentralObjectMember CentralObject;
    oscTrafficDefinitionMember TrafficDefinition;

};

typedef oscObjectVariable<oscSwarm *> oscSwarmMember;
typedef oscObjectVariableArray<oscSwarm *> oscSwarmArrayMember;


}

#endif //OSCSWARM_H
