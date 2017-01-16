/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCMANEUVER_H
#define OSCMANEUVER_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"
#include "oscObjectVariableArray.h"

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
        OSC_OBJECT_ADD_MEMBER(Event, "oscEvent");
    };
    oscString name;
    oscEventArrayMember Event;

};

typedef oscObjectVariable<oscManeuver *> oscManeuverMember;
typedef oscObjectVariableArray<oscManeuver *> oscManeuverArrayMember;


}

#endif //OSCMANEUVER_H
