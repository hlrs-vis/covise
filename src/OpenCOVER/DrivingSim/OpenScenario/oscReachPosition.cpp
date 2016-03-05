/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#include "oscReachPosition.h"


using namespace OpenScenario;


conditionReachPositionType::conditionReachPositionType()
{
    addEnum("exceed", oscReachPosition::exceed);
    addEnum("deceed", oscReachPosition::deceed);
}

conditionReachPositionType *conditionReachPositionType::instance()
{
    if(inst == NULL)
    {
        inst = new conditionReachPositionType();
    }
    return inst;
}

conditionReachPositionType *conditionReachPositionType::inst = NULL;
