/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_END_CONDITION_TYPE_B_H
#define OSC_END_CONDITION_TYPE_B_H

#include <oscExport.h>
#include <oscConditionTypeB.h>
#include <oscObjectVariable.h>

#include <oscVariables.h>


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscEndConditionTypeB: public oscConditionTypeB
{
public:
    oscEndConditionTypeB()
    {

    };

};

typedef oscObjectVariable<oscEndConditionTypeB *> oscEndConditionTypeBMember;

}

#endif //OSC_END_CONDITION_TYPE_B_H

