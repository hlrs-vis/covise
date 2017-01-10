/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCTIMEHEADWAY_H
#define OSCTIMEHEADWAY_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT Enum_ruleType : public oscEnumType
{
public:
static Enum_ruleType *instance();
    private:
		Enum_ruleType();
	    static Enum_ruleType *inst; 
};
class OPENSCENARIOEXPORT oscTimeHeadway : public oscObjectBase
{
public:
oscTimeHeadway()
{
        OSC_ADD_MEMBER(entity);
        OSC_ADD_MEMBER(value);
        OSC_ADD_MEMBER(freespace);
        OSC_ADD_MEMBER(alongRoute);
        OSC_ADD_MEMBER(rule);
        rule.enumType = Enum_ruleType::instance();
    };
    oscString entity;
    oscDouble value;
    oscBool freespace;
    oscBool alongRoute;
    oscEnum rule;

    enum Enum_rule
    {
greater_than,
less_than,
equal_to,

    };

};

typedef oscObjectVariable<oscTimeHeadway *> oscTimeHeadwayMember;
typedef oscObjectVariableArray<oscTimeHeadway *> oscTimeHeadwayArrayMember;


}

#endif //OSCTIMEHEADWAY_H
