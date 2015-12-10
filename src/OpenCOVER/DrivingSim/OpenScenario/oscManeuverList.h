/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_MANEUVER_LIST_H
#define OSC_MANEUVER_LIST_H
#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscStartConditionGroup.h>
#include <oscEndConditionGroup.h>
#include <oscCancelConditionGroup.h>
#include <oscManeuverRef.h>

namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscManeuverList: public oscObjectBase
{
public:
    oscManeuverList()
    {
       OSC_OBJECT_ADD_MEMBER(startConditionGroup,"oscStartConditionGroup");
	   OSC_OBJECT_ADD_MEMBER(endConditionGroup,"oscEndConditionGroup");
       OSC_OBJECT_ADD_MEMBER(cancelConditionGroup,"oscCancelConditionGroup");
       OSC_OBJECT_ADD_MEMBER(maneuverRef,"oscManeuverRef");
    };
    oscStartConditionGroupMember startConditionGroup;
    oscEndConditionGroupMember endConditionGroup;
    oscCancelConditionGroupMember cancelConditionGroup;
    oscManeuverRefMember maneuverRef;
};

typedef oscObjectVariable<oscManeuverList *>oscManeuverListMember;

}

#endif //OSC_MANEUVER_LIST_H