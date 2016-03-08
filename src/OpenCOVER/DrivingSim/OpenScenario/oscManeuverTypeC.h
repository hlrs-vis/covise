/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_MANEUVER_TYPE_C_H
#define OSC_MANEUVER_TYPE_C_H

#include "oscExport.h"
#include "oscNamePriority.h"
#include "oscObjectVariable.h"

#include "oscStartConditionsTypeB.h"
#include "oscEndConditionsTypeB.h"
#include "oscCancelConditionsTypeB.h"
#include "oscCatalogReferenceTypeA.h"
#include "oscManeuverTypeA.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscManeuverTypeC: public oscNamePriority
{
public:
    oscManeuverTypeC()
    {
        OSC_OBJECT_ADD_MEMBER(startConditions, "oscStartConditionsTypeB");
        OSC_OBJECT_ADD_MEMBER(endConditions, "oscEndConditionsTypeB");
        OSC_OBJECT_ADD_MEMBER(cancelConditions, "oscCancelConditionsTypeB");
        OSC_OBJECT_ADD_MEMBER_CHOICE(catalogReference, "oscCatalogReferenceTypeA");
        OSC_OBJECT_ADD_MEMBER_CHOICE(maneuver, "oscManeuverTypeA");
    };

    oscStartConditionsTypeBMemberArray startConditions;
    oscEndConditionsTypeBMemberArray endConditions;
    oscCancelConditionsTypeBMemberArray cancelConditions;
    oscCatalogReferenceTypeAMember catalogReference;
    oscManeuverTypeAMember maneuver;
};

typedef oscObjectVariable<oscManeuverTypeC *>oscManeuverTypeCMember;

}

#endif //OSC_MANEUVER_TYPE_C_H
