/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#include "oscReferenceHandling.h"


using namespace OpenScenario;


refHandlConditionType::refHandlConditionType()
{
    addEnum("starts", oscReferenceHandling::starts);
    addEnum("ends", oscReferenceHandling::ends);
    addEnum("cancels", oscReferenceHandling::cancels);
}

refHandlConditionType *refHandlConditionType::instance()
{
    if(inst == NULL)
    {
        inst = new refHandlConditionType();
    }
    return inst;
}

refHandlConditionType *refHandlConditionType::inst = NULL;
