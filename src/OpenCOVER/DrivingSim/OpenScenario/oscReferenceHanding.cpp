/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#include "oscReferenceHanding.h"


using namespace OpenScenario;


conditionRefType::conditionRefType()
{
    addEnum("starts", oscReferenceHanding::starts);
    addEnum("ends", oscReferenceHanding::ends);
    addEnum("cancels", oscReferenceHanding::cancels);
}

conditionRefType *conditionRefType::instance()
{
    if(inst == NULL)
    {
        inst = new conditionRefType();
    }
    return inst;
}

conditionRefType *conditionRefType::inst = NULL;
