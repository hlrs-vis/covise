/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#include <oscNumericCondition.h>


using namespace OpenScenario;


ruleType::ruleType()
{
    addEnum("equal", oscNumericCondition::equal);
    addEnum("notEqual", oscNumericCondition::notEqual);
    addEnum("lessEqual", oscNumericCondition::lessEqual);
    addEnum("less", oscNumericCondition::less);
    addEnum("greaterEqual", oscNumericCondition::greaterEqual);
    addEnum("greater", oscNumericCondition::greater);
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
