/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_REFACTOR_TYPE_A_LIST_H
#define OSC_REFACTOR_TYPE_A_LIST_H

#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectArrayVariable.h>

#include <oscRefActorTypeA.h>


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscRefActorTypeAList: public oscObjectBase
{
public:
    oscRefActorTypeAList()
    {
        OSC_OBJECT_ADD_MEMBER(refActor, "oscRefActorTypeA");
    };

    oscRefActorTypeAMember refActor;
};

typedef oscObjectArrayVariable<oscRefActorTypeAList *> oscRefActorTypeAListArrayMember;

}

#endif /* OSC_REFACTOR_TYPE_A_LIST_H */
