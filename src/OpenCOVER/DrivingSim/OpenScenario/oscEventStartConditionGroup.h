/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_Event_START_CONGDITION_GROUP_H
#define OSC_Event_START_CONGDITION_GROUP_H

#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectArrayVariable.h>

#include <oscEventStartCondition.h>


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscEventStartConditionGroup: public oscObjectBase
{
public:
    oscEventStartConditionGroup()
    {
        OSC_OBJECT_ADD_MEMBER(startCondition, "oscEventStartCondition");
    };

    oscEventStartConditionMember startCondition;
};

typedef oscObjectArrayVariable<oscEventStartConditionGroup *> oscEventStartConditionGroupArrayMember;

}

#endif //OSC_Event_START_CONGDITION_GROUP_H
