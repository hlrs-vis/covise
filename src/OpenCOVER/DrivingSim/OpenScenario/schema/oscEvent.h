/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCEVENT_H
#define OSCEVENT_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "schema/oscAction.h"
#include "schema/oscEventConditions.h"

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
        OSC_ADD_MEMBER(name);
        OSC_ADD_MEMBER(priority);
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(Action, "oscAction");
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(EventConditions, "oscEventConditions");
    };
    oscString name;
    oscEnum priority;
    oscActionMember Action;
    oscEventConditionsMember EventConditions;

    enum Enum_event_priority
    {
overwrite,
following,
skip,

    };

};

typedef oscObjectVariable<oscEvent *> oscEventMember;


}

#endif //OSCEVENT_H
