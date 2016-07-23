/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_OBJECTS_H
#define OSC_OBJECTS_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariableArray.h"

#include "oscObject.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscObjects: public oscObjectBase
{
public:
    oscObjects()
    {
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(object, "oscObject");
    };

    oscObjectMember object;
};

typedef oscObjectVariableArray<oscObjects *> oscObjectsArrayMember;

}

#endif /* OSC_OBJECTS_H */
