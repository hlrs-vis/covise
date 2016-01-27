/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_START_CONDITIONS_GROUPS_TYPE_A_H
#define OSC_START_CONDITIONS_GROUPS_TYPE_A_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectArrayVariable.h"

#include "oscStartConditionsGroupTypeA.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscStartConditionsGroupsTypeA: public oscObjectBase
{
public:
    oscStartConditionsGroupsTypeA()
    {
        OSC_OBJECT_ADD_MEMBER(startConditionGroup, "oscStartConditionsGroupTypeA");
    };

    oscStartConditionsGroupTypeAMember startConditionGroup;
};

typedef oscObjectArrayVariable<oscStartConditionsGroupsTypeA *> oscStartConditionsGroupsTypeAArrayMember;

}

#endif /* OSC_START_CONDITIONS_GROUPS_TYPE_A_H */
