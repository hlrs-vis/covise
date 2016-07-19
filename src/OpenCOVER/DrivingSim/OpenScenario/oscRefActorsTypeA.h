/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_REFACTORS_TYPE_A_H
#define OSC_REFACTORS_TYPE_A_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariableArray.h"

#include "oscRefActorTypeA.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscRefActorsTypeA: public oscObjectBase
{
public:
    oscRefActorsTypeA()
    {
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(refActor, "oscRefActorTypeA");
    };

    oscRefActorTypeAMember refActor;
};

typedef oscObjectVariableArray<oscRefActorsTypeA *> oscRefActorsTypeAArrayMember;

}

#endif /* OSC_REFACTORS_TYPE_A_H */
