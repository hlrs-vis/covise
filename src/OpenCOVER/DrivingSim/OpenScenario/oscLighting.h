/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_LIGHTING_H
#define OSC_LIGHTING_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariableArray.h"

#include "oscLights.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscLighting : public oscObjectBase
{
public:
    oscLighting()
    {
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(light, "oscLights");
    };

    oscLightsMember light;
};

typedef oscObjectVariableArray<oscLighting *> oscLightingMemberArray;

}

#endif //OSC_LIGHTING_H
