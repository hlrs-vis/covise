/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#include "oscControlPoint.h"


using namespace OpenScenario;


statusType::statusType()
{
    addEnum("to be defined", oscControlPoint::toBeDefined);
}

statusType *statusType::instance()
{
    if(inst == NULL)
    {
        inst = new statusType();
    }
    return inst;
}

statusType *statusType::inst = NULL;



