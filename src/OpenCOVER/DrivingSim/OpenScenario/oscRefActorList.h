/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_REF_ACTOR_LIST_H
#define OSC_REF_ACTOR_LIST_H
#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscVariables.h>
#include <oscManeuverRefActor.h>

namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscRefActorList: public oscObjectBase
{
public:
    oscRefActorList()
    {
        OSC_ADD_MEMBER(name);
		OSC_OBJECT_ADD_MEMBER(maneuver,"oscManeuverRefActor");
    };
    oscString name;
	oscManeuverRefActorMember maneuver;
};

typedef oscObjectVariable<oscRefActorList *> oscRefActorListMember;

}

#endif //OSC_REF_ACTOR_LIST_H
