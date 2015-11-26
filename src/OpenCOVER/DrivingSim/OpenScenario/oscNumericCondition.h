/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_NUMERIC_CONDITION_H
#define OSC_NUMERIC_CONDITION_H
#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscVariables.h>

namespace OpenScenario {


class OPENSCENARIOEXPORT ruleType: public oscEnumType
{
public:
    static ruleType *instance(); 
private:
    ruleType();
    static ruleType *inst;
};

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscNumericCondition: public oscObjectBase
{
public:
	oscString name;
	oscInt value;
    enum rule
    {
        equal,
		notEqual,
		lessEqual,
		less,
		greaterEqual,
		greater,
    };
    oscNumericCondition()
    {
		OSC_ADD_MEMBER(name);
		OSC_ADD_MEMBER(value);
		OSC_ADD_MEMBER(rule);
		rule.enumType = ruleType::instance();
    };
	oscEnum rule;
};

typedef oscObjectVariable<oscNumericCondition *> oscNumericConditionMember;

}

#endif //OSC_NUMERIC_CONDITION_H
