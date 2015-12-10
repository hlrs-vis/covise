/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_END_CONDITION_H
#define OSC_END_CONDITION_H
#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscConditionObject.h>

namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscEndCondition: public oscConditionObject
{
public:
    oscEndCondition()
    {
    };
};

typedef oscObjectVariable<oscEndCondition *> oscEndConditionMember;

}

#endif //OSC_END_CONDITION_H
