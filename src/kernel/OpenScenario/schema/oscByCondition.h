/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCBYCONDITION_H
#define OSCBYCONDITION_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT Enum_ByCondition_actorType : public oscEnumType
{
public:
static Enum_ByCondition_actorType *instance();
    private:
		Enum_ByCondition_actorType();
	    static Enum_ByCondition_actorType *inst; 
};
class OPENSCENARIOEXPORT oscByCondition : public oscObjectBase
{
public:
oscByCondition()
{
        OSC_ADD_MEMBER(actor, 0);
        actor.enumType = Enum_ByCondition_actorType::instance();
    };
        const char *getScope(){return "/OpenSCENARIO/Storyboard/Story/Act/Sequence/Actors";};
    oscEnum actor;

    enum Enum_ByCondition_actor
    {
triggeringEntity,
anyEntity,

    };

};

typedef oscObjectVariable<oscByCondition *> oscByConditionMember;
typedef oscObjectVariableArray<oscByCondition *> oscByConditionArrayMember;


}

#endif //OSCBYCONDITION_H
