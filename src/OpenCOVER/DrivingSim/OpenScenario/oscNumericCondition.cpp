/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#include "oscNumericCondition.h"


using namespace OpenScenario;


ruleType::ruleType()
{
    addEnum("==", oscNumericCondition::equal);
    addEnum("!=", oscNumericCondition::notEqual);
    addEnum("<=", oscNumericCondition::lessEqual);
    addEnum("<", oscNumericCondition::less);
    addEnum(">=", oscNumericCondition::greaterEqual);
    addEnum(">", oscNumericCondition::greater);
}

ruleType *ruleType::instance()
{
    if(inst == NULL)
    {
        inst = new ruleType();
    }
    return inst;
}

ruleType *ruleType::inst = NULL;
