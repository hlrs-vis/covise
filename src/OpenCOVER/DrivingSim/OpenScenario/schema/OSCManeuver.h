/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCMANEUVER_H
#define OSCMANEUVER_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "schema/oscEvent.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscManeuver : public oscObjectBase
{
public:
    oscManeuver()
    {
        OSC_ADD_MEMBER(name);
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(Event, "oscEvent");
    };
    oscString name;
    oscEventMember Event;

};

typedef oscObjectVariable<oscManeuver *> oscManeuverMember;


}

#endif //OSCMANEUVER_H
