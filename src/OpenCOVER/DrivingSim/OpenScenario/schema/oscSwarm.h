/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCSWARM_H
#define OSCSWARM_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "schema/oscCentralObject.h"
#include "schema/oscTrafficDefinition.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscSwarm : public oscObjectBase
{
public:
    oscSwarm()
    {
        OSC_ADD_MEMBER(semiMajorAxis);
        OSC_ADD_MEMBER(semiMinorAxis);
        OSC_ADD_MEMBER(innerRadius);
        OSC_ADD_MEMBER(offset);
        OSC_OBJECT_ADD_MEMBER(CentralObject, "oscCentralObject");
        OSC_OBJECT_ADD_MEMBER(TrafficDefinition, "oscTrafficDefinition");
    };
    oscDouble semiMajorAxis;
    oscDouble semiMinorAxis;
    oscDouble innerRadius;
    oscDouble offset;
    oscCentralObjectMember CentralObject;
    oscTrafficDefinitionMember TrafficDefinition;

};

typedef oscObjectVariable<oscSwarm *> oscSwarmMember;


}

#endif //OSCSWARM_H
