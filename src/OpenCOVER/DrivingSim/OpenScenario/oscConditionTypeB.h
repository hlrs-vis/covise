/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_CONDITION_TYPE_B_H
#define OSC_CONDITION_TYPE_B_H

#include <oscExport.h>
#include <oscNamedObject.h>
#include <oscObjectVariable.h>

#include <oscVariables.h>
#include <oscConditionBase.h>


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscConditionTypeB: public oscNamedObject
{
public:
    oscConditionTypeB()
    {
        OSC_ADD_MEMBER(iid);
        OSC_ADD_MEMBER(groupId);
        OSC_ADD_MEMBER(counter);
        OSC_OBJECT_ADD_MEMBER(condition, "oscConditionBase");
    };

    oscInt iid;
    oscInt groupId;
    oscInt counter;
    oscConditionBaseMember condition;
};

typedef oscObjectVariable<oscConditionTypeB *> oscConditionTypeBMember;

}

#endif //OSC_CONDITION_TYPE_B_H
