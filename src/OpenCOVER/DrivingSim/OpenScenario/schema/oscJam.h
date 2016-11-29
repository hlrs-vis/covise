/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCJAM_H
#define OSCJAM_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "schema/oscPosition.h"
#include "schema/oscTrafficDefinition.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscJam : public oscObjectBase
{
public:
    oscJam()
    {
        OSC_ADD_MEMBER(direction);
        OSC_ADD_MEMBER(speed);
        OSC_ADD_MEMBER(length);
        OSC_OBJECT_ADD_MEMBER(Position, "oscPosition");
        OSC_OBJECT_ADD_MEMBER(TrafficDefinition, "oscTrafficDefinition");
    };
    oscString direction;
    oscDouble speed;
    oscDouble length;
    oscPositionMember Position;
    oscTrafficDefinitionMember TrafficDefinition;

};

typedef oscObjectVariable<oscJam *> oscJamMember;


}

#endif //OSCJAM_H
