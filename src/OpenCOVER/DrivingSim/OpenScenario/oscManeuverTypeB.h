/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_MANEUVER_TYPE_B_H
#define OSC_MANEUVER_TYPE_B_H

#include "oscExport.h"
#include "oscNamePriority.h"
#include "oscObjectVariable.h"

#include "oscRefActorsTypeB.h"
#include "oscCatalogReferenceTypeA.h"
#include "oscManeuverTypeA.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscManeuverTypeB: public oscNamePriority
{
public:
    oscManeuverTypeB()
    {
        OSC_OBJECT_ADD_MEMBER(refActors, "oscRefActorsTypeB");
        OSC_OBJECT_ADD_MEMBER(catalogReference, "oscCatalogReferenceTypeA");
        OSC_OBJECT_ADD_MEMBER(maneuver, "oscManeuverTypeA");
    };

    oscRefActorsTypeBMemberArray refActors;
    oscCatalogReferenceTypeAMember catalogReference;
    oscManeuverTypeAMember maneuver;
};

typedef oscObjectVariable<oscManeuverTypeB *>oscManeuverTypeBMember;

}

#endif //OSC_MANEUVER_TYPE_B_H
