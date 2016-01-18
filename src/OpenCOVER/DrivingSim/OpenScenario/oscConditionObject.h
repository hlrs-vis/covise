/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_CONDITION_OBJECT_H
#define OSC_CONDITION_OBJECT_H

#include <oscExport.h>
#include <oscNamedObject.h>
#include <oscObjectVariable.h>

#include <oscVariables.h>
#include <oscCondition.h>


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscConditionObject: public oscNamedObject
{
public:
    oscConditionObject()
    {
        OSC_ADD_MEMBER(counter);
        OSC_OBJECT_ADD_MEMBER(condition,"oscCondition");
    };

    oscInt counter;
    oscConditionMember condition;
};

typedef oscObjectVariable<oscConditionObject *> oscConditionObjectMember;

}

#endif //OSC_CONDITION_OBJECT_H
