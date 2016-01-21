/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_MANEUVER_TYPE_B_H
#define OSC_MANEUVER_TYPE_B_H

#include "oscExport.h"
#include "oscNamedPriority.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "oscRefActorsTypeB.h"
#include "oscCatalogRef.h"
#include "oscManeuverTypeA.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscManeuverTypeB: public oscNamedPriority
{
public:
    oscManeuverTypeB()
    {
        OSC_OBJECT_ADD_MEMBER(refActors, "oscRefActorsTypeB");
        OSC_OBJECT_ADD_MEMBER(catalogRef, "oscCatalogRef");
        OSC_OBJECT_ADD_MEMBER(maneuver, "oscManeuverTypeA");
    };

    oscRefActorsTypeBArrayMember refActors;
    oscCatalogRefMember catalogRef;
    oscManeuverTypeAMember maneuver;
};

typedef oscObjectVariable<oscManeuverTypeB *>oscManeuverTypeBMember;

}

#endif //OSC_MANEUVER_TYPE_B_H
