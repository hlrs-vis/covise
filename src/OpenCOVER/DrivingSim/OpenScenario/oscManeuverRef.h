/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_MANEUVER_REF_H
#define OSC_MANEUVER_REF_H
#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscVariables.h>
#include <oscNamedObject.h>
#include <oscRefActor.h>
#include <oscCatalogRef.h>
#include <oscManeuver.h>


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscManeuverRef: public oscNamedObject
{
public:
    oscManeuverRef()
    {
       OSC_OBJECT_ADD_MEMBER(refActor,"oscRefActor");
	   OSC_ADD_MEMBER(priority);
       OSC_ADD_MEMBER(numberOfExecutions);
       OSC_OBJECT_ADD_MEMBER(catalogRef,"oscCatalogRef");
       OSC_OBJECT_ADD_MEMBER(maneuver,"oscManeuver");
    };
    oscRefActorMember refActor;
    oscString priority;
    oscString numberOfExecutions;
    oscCatalogRefMember catalogRef;
    oscManeuverMember maneuver;
};

typedef oscObjectVariable<oscManeuverRef *>oscManeuverRefMember;

}

#endif //OSC_MANEUVER_REF_H
