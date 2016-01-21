/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_START_CONDITIONS_GROUP_TYPE_C_H
#define OSC_START_CONDITIONS_GROUP_TYPE_C_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscStartConditionsTypeC.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscStartConditionsGroupTypeC: public oscObjectBase
{
public:
    oscStartConditionsGroupTypeC()
    {
        OSC_OBJECT_ADD_MEMBER(startConditions, "oscStartConditionsTypeC");
    };

    oscStartConditionsTypeCArrayMember startConditions;
};

typedef oscObjectVariable<oscStartConditionsGroupTypeC *> oscStartConditionsGroupTypeCMember;

}

#endif //OSC_START_CONDITIONS_GROUP_TYPE_C_H
