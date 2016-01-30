/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#include "oscReachPosition.h"


using namespace OpenScenario;


reachPosConditionType::reachPosConditionType()
{
    addEnum("exceed", oscReachPosition::exceed);
    addEnum("deceed", oscReachPosition::deceed);
}

reachPosConditionType *reachPosConditionType::instance()
{
    if(inst == NULL)
    {
        inst = new reachPosConditionType();
    }
    return inst;
}

reachPosConditionType *reachPosConditionType::inst = NULL;
