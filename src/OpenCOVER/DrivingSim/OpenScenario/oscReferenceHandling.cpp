/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#include "oscReferenceHandling.h"


using namespace OpenScenario;


conditionReferenceHandlingType::conditionReferenceHandlingType()
{
    addEnum("starts", oscReferenceHandling::starts);
    addEnum("ends", oscReferenceHandling::ends);
    addEnum("cancels", oscReferenceHandling::cancels);
}

conditionReferenceHandlingType *conditionReferenceHandlingType::instance()
{
    if(inst == NULL)
    {
        inst = new conditionReferenceHandlingType();
    }
    return inst;
}

conditionReferenceHandlingType *conditionReferenceHandlingType::inst = NULL;
