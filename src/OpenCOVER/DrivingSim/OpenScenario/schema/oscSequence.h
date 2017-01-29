/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCSEQUENCE_H
#define OSCSEQUENCE_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscActors.h"
#include "oscCatalogReference.h"
#include "oscManeuver.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscSequence : public oscObjectBase
{
public:
oscSequence()
{
        OSC_ADD_MEMBER(numberOfExecutions, 0);
        OSC_ADD_MEMBER(name, 0);
        OSC_OBJECT_ADD_MEMBER(Actors, "oscActors", 0);
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(CatalogReference, "oscCatalogReference", 0);
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(Maneuver, "oscManeuver", 0);
    };
    oscUInt numberOfExecutions;
    oscString name;
    oscActorsMember Actors;
    oscCatalogReferenceArrayMember CatalogReference;
    oscManeuverArrayMember Maneuver;

};

typedef oscObjectVariable<oscSequence *> oscSequenceMember;
typedef oscObjectVariableArray<oscSequence *> oscSequenceArrayMember;


}

#endif //OSCSEQUENCE_H
