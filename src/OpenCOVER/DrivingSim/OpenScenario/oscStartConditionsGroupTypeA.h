/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_START_CONDITIONS_GROUP_TYPE_A_H
#define OSC_START_CONDITIONS_GROUP_TYPE_A_H

#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>

#include <oscStartConditionsTypeA.h>


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscStartConditionsGroupTypeA: public oscObjectBase
{
public:
    oscStartConditionsGroupTypeA()
    {
        OSC_OBJECT_ADD_MEMBER(startConditions, "oscStartConditionsTypeA");
    };

    oscStartConditionsTypeAArrayMember startConditions;
};

typedef oscObjectVariable<oscStartConditionsGroupTypeA *> oscStartConditionsGroupTypeAMember;

}

#endif //OSC_START_CONDITIONS_GROUP_TYPE_A_H
