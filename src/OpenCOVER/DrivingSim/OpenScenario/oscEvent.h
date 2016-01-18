/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_EVENT_H
#define OSC_EVENT_H

#include <oscExport.h>
#include <oscPriority.h>
#include <oscObjectVariable.h>

#include <oscVariables.h>
#include <oscEventStartConditionGroup.h>
#include <oscAction.h>

namespace OpenScenario {


/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscEvent: public oscPriority
{
public:
    oscEvent()
    {
        OSC_ADD_MEMBER(name);
        OSC_OBJECT_ADD_MEMBER(startConditionGroup,"oscEventStartConditionGroup");
        OSC_OBJECT_ADD_MEMBER(action,"oscAction");
    };

    oscString name;
    oscEventStartConditionGroupArrayMember startConditionGroup;
    oscActionMember action;
};

typedef oscObjectVariable<oscEvent *> oscEventMember;

}

#endif //OSC_EVENT_H
