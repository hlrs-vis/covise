/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#include "oscConditionChoiceTypeA.h"


using namespace OpenScenario;


referenceType::referenceType()
{
    addEnum("relative", oscConditionChoiceTypeA::relative);
    addEnum("absolute", oscConditionChoiceTypeA::absolute);
}

referenceType *referenceType::instance()
{
    if(inst == NULL)
    {
        inst = new referenceType();
    }
    return inst;
}

referenceType *referenceType::inst = NULL;
