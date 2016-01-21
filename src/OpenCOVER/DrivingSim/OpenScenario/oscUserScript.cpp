/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#include "oscUserScript.h"


using namespace OpenScenario;


executionType::executionType()
{
    addEnum("fireAndForget", oscUserScript::fireAndForget);
    addEnum("waitForTermination", oscUserScript::waitForTermination);
}

executionType *executionType::instance()
{
    if(inst == NULL)
    {
        inst = new executionType();
    }
    return inst;
}

executionType *executionType::inst = NULL;
