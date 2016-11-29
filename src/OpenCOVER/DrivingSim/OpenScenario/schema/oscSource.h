/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCSOURCE_H
#define OSCSOURCE_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "schema/oscPosition.h"
#include "schema/oscTrafficDefinition.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscSource : public oscObjectBase
{
public:
    oscSource()
    {
        OSC_ADD_MEMBER(rate);
        OSC_ADD_MEMBER(radius);
        OSC_OBJECT_ADD_MEMBER(Position, "oscPosition");
        OSC_OBJECT_ADD_MEMBER(TrafficDefinition, "oscTrafficDefinition");
    };
    oscDouble rate;
    oscDouble radius;
    oscPositionMember Position;
    oscTrafficDefinitionMember TrafficDefinition;

};

typedef oscObjectVariable<oscSource *> oscSourceMember;


}

#endif //OSCSOURCE_H
