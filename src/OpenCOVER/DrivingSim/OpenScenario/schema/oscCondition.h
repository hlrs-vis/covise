/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCCONDITION_H
#define OSCCONDITION_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscByEntity.h"
#include "oscByState.h"
#include "oscByValue.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT Enum_Condition_edgeType : public oscEnumType
{
public:
static Enum_Condition_edgeType *instance();
    private:
		Enum_Condition_edgeType();
	    static Enum_Condition_edgeType *inst; 
};
class OPENSCENARIOEXPORT oscCondition : public oscObjectBase
{
public:
oscCondition()
{
        OSC_ADD_MEMBER(name, 0);
        OSC_ADD_MEMBER(delay, 0);
        OSC_ADD_MEMBER(edge, 0);
        OSC_OBJECT_ADD_MEMBER(ByEntity, "oscByEntity", 1);
        OSC_OBJECT_ADD_MEMBER(ByState, "oscByState", 1);
        OSC_OBJECT_ADD_MEMBER(ByValue, "oscByValue", 1);
        edge.enumType = Enum_Condition_edgeType::instance();
    };
        const char *getScope(){return "";};
    oscString name;
    oscDouble delay;
    oscEnum edge;
    oscByEntityMember ByEntity;
    oscByStateMember ByState;
    oscByValueMember ByValue;

    enum Enum_Condition_edge
    {
rising,
falling,
any,

    };

};

typedef oscObjectVariable<oscCondition *> oscConditionMember;
typedef oscObjectVariableArray<oscCondition *> oscConditionArrayMember;


}

#endif //OSCCONDITION_H
