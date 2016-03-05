/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_CHOICE_OBJECT_H
#define OSC_CHOICE_OBJECT_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscNameRefId.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscChoiceObject: public oscObjectBase
{
public:
    oscChoiceObject()
    {
        OSC_OBJECT_ADD_MEMBER(vehicle, "oscNameRefId");
        OSC_OBJECT_ADD_MEMBER(pedestrian, "oscNameRefId");
        OSC_OBJECT_ADD_MEMBER(miscObject, "oscNameRefId");
    };

    oscNameRefIdMember vehicle;
    oscNameRefIdMember pedestrian;
    oscNameRefIdMember miscObject;
};

typedef oscObjectVariable<oscChoiceObject *> oscChoiceObjectMember;

}

#endif //OSC_CHOICE_OBJECT_H
