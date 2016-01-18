/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_MANEUVER_TYPE_C_H
#define OSC_MANEUVER_TYPE_C_H

#include <oscExport.h>
#include <oscNamedPriority.h>
#include <oscObjectVariable.h>

#include <oscStartConditionTypeBGroup.h>
#include <oscEndConditionTypeBGroup.h>
#include <oscCancelConditionTypeBGroup.h>
#include <oscCatalogRef.h>
#include <oscManeuverTypeA.h>


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscManeuverTypeC: public oscNamedPriority
{
public:
    oscManeuverTypeC()
    {
        OSC_OBJECT_ADD_MEMBER(startCondition, "oscStartConditionTypeBGroup");
        OSC_OBJECT_ADD_MEMBER(endCondition, "oscEndConditionTypeBGroup");
        OSC_OBJECT_ADD_MEMBER(cancelCondition, "oscCancelConditionTypeBGroup");
        OSC_OBJECT_ADD_MEMBER(catalogRef, "oscCatalogRef");
        OSC_OBJECT_ADD_MEMBER(maneuver, "oscManeuverTypeA");
    };

    oscStartConditionTypeBGroupArrayMember startCondition;
    oscEndConditionTypeBGroupArrayMember endCondition;
    oscCancelConditionTypeBGroupArrayMember cancelCondition;
    oscCatalogRefMember catalogRef;
    oscManeuverTypeAMember maneuver;
};

typedef oscObjectVariable<oscManeuverTypeC *>oscManeuverTypeCMember;

}

#endif //OSC_MANEUVER_TYPE_C_H
