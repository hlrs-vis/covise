/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_REFACTOR_TYPE_B_LIST_H
#define OSC_REFACTOR_TYPE_B_LIST_H

#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectArrayVariable.h>

#include <oscRefActorTypeB.h>


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscRefActorTypeBList: public oscObjectBase
{
public:
    oscRefActorTypeBList()
    {
        OSC_OBJECT_ADD_MEMBER(refActor, "oscRefActorTypeB");
    };

    oscRefActorTypeBMember refActor;
};

typedef oscObjectArrayVariable<oscRefActorTypeBList *> oscRefActorTypeBListArrayMember;

}

#endif /* OSC_REFACTOR_TYPE_B_LIST_H */
