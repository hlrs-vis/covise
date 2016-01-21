/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_MANEUVER_GROUP_TYPE_AB_H
#define OSC_MANEUVER_GROUP_TYPE_AB_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscEndConditionsGroupsTypeA.h"
#include "oscCancelConditionsGroupsTypeA.h"
#include "oscStartConditionsGroupsTypeA.h"
#include "oscManeuversTypeB.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscManeuverGroupTypeAB: public oscObjectBase
{
public:
    oscManeuverGroupTypeAB()
    {
        OSC_OBJECT_ADD_MEMBER(startConditionGroups, "oscStartConditionsGroupsTypeA");
        OSC_OBJECT_ADD_MEMBER(endConditionGroups, "oscEndConditionsGroupsTypeA");
        OSC_OBJECT_ADD_MEMBER(cancelConditionGroups, "oscCancelConditionsGroupsTypeA");
        OSC_OBJECT_ADD_MEMBER(maneuvers, "oscManeuversTypeB");
    };

    oscStartConditionsGroupsTypeAArrayMember startConditionGroups;
    oscEndConditionsGroupsTypeAArrayMember endConditionGroups;
    oscCancelConditionsGroupsTypeAArrayMember cancelConditionGroups;
    oscManeuversTypeBArrayMember maneuvers;
};

typedef oscObjectVariable<oscManeuverGroupTypeAB *>oscManeuverGroupTypeABMember;

}

#endif //OSC_MANEUVER_GROUP_TYPE_AB_H
