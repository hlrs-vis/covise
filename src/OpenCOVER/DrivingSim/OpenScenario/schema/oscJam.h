/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCJAM_H
#define OSCJAM_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscPosition.h"
#include "oscTrafficDefinition.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscJam : public oscObjectBase
{
public:
oscJam()
{
        OSC_ADD_MEMBER(direction, 0);
        OSC_ADD_MEMBER(speed, 0);
        OSC_ADD_MEMBER(length, 0);
        OSC_OBJECT_ADD_MEMBER(Position, "oscPosition", 0);
        OSC_OBJECT_ADD_MEMBER(TrafficDefinition, "oscTrafficDefinition", 0);
    };
    oscString direction;
    oscDouble speed;
    oscDouble length;
    oscPositionMember Position;
    oscTrafficDefinitionMember TrafficDefinition;

};

typedef oscObjectVariable<oscJam *> oscJamMember;
typedef oscObjectVariableArray<oscJam *> oscJamArrayMember;


}

#endif //OSCJAM_H
