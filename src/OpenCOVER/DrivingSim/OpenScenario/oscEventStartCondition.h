/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_EVENT_START_CONDITION_H
#define OSC_EVENT_START_CONDITION_H

#include <oscExport.h>
#include <oscNamedObject.h>
#include <oscObjectVariable.h>

#include <oscVariables.h>
#include <oscCondition.h>


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscEventStartCondition: public oscNamedObject
{
public:
    oscEventStartCondition()
    {
        OSC_ADD_MEMBER(delayTime);
        OSC_OBJECT_ADD_MEMBER(condition,"oscCondition");
    };

    oscDouble delayTime;
    oscConditionMember condition;
};

typedef oscObjectVariable<oscEventStartCondition *> oscEventStartConditionMember;

}

#endif //OSC_EVENT_START_CONDITION_H
