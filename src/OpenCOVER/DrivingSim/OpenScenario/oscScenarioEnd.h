/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_SCENARIO_END_H
#define OSC_SCENARIO_END_H

#include <oscExport.h>
#include <oscObjectBase.h>

#include <oscEndConditionGroup.h>
#include <oscUserData.h>


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscScenarioEnd: public oscObjectBase
{
public:
    oscScenarioEnd()
    {
       OSC_OBJECT_ADD_MEMBER(endConditionGroup,"oscEndConditionGroup");
       OSC_OBJECT_ADD_MEMBER(userData,"oscUserData");
    };

    oscEndConditionGroupMember endConditionGroup;
    oscUserDataMember userData;
};

typedef oscObjectVariable<oscScenarioEnd *> oscScenarioEndMember;

}

#endif //OSC_SCENARIO_END_H
