/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCPROPERTIES_H
#define OSCPROPERTIES_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscProperty.h"
#include "oscFile.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscProperties : public oscObjectBase
{
public:
oscProperties()
{
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(Property, "oscProperty", 0);
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(File, "oscFile", 0);
    };
        const char *getScope(){return "";};
    oscPropertyArrayMember Property;
    oscFileArrayMember File;

};

typedef oscObjectVariable<oscProperties *> oscPropertiesMember;
typedef oscObjectVariableArray<oscProperties *> oscPropertiesArrayMember;


}

#endif //OSCPROPERTIES_H
