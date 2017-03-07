/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCEVENT_H
#define OSCEVENT_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscAction.h"
#include "oscConditions.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT Enum_event_priorityType : public oscEnumType
{
public:
static Enum_event_priorityType *instance();
    private:
		Enum_event_priorityType();
	    static Enum_event_priorityType *inst; 
};
class OPENSCENARIOEXPORT oscEvent : public oscObjectBase
{
public:
oscEvent()
{
        OSC_ADD_MEMBER(name, 0);
        OSC_ADD_MEMBER(priority, 0);
        OSC_OBJECT_ADD_MEMBER(Action, "oscAction", 0);
        OSC_OBJECT_ADD_MEMBER(Conditions, "oscConditions", 0);
        priority.enumType = Enum_event_priorityType::instance();
    };
        const char *getScope(){return "/OSCManeuver";};
    oscString name;
    oscEnum priority;
    oscActionArrayMember Action;
    oscConditionsArrayMember Conditions;

    enum Enum_event_priority
    {
overwrite,
following,
skip,

    };

};

typedef oscObjectVariable<oscEvent *> oscEventMember;
typedef oscObjectVariableArray<oscEvent *> oscEventArrayMember;


}

#endif //OSCEVENT_H
