/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_MANEUVER_TYPE_B_H
#define OSC_MANEUVER_TYPE_B_H

#include <oscExport.h>
#include <oscNamedPriority.h>
#include <oscObjectVariable.h>

#include <oscVariables.h>
#include <oscCatalogRef.h>
#include <oscManeuverTypeA.h>
#include <oscRefActorTypeBList.h>


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscManeuverTypeB: public oscNamedPriority
{
public:
    oscManeuverTypeB()
    {
        OSC_OBJECT_ADD_MEMBER(refActorList, "oscRefActorTypeBList");
        OSC_OBJECT_ADD_MEMBER(catalogRef, "oscCatalogRef");
        OSC_OBJECT_ADD_MEMBER(maneuver, "oscManeuverTypeA");
    };

    oscRefActorTypeBListArrayMember refActorList;
    oscCatalogRefMember catalogRef;
    oscManeuverTypeAMember maneuver;
};

typedef oscObjectVariable<oscManeuverTypeB *>oscManeuverTypeBMember;

}

#endif //OSC_MANEUVER_TYPE_B_H
