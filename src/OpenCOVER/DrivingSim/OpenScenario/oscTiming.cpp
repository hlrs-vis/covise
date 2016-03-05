/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#include "oscTiming.h"


using namespace OpenScenario;


domainType::domainType()
{
    addEnum("absolute", oscTiming::absolute);
    addEnum("relative", oscTiming::relative);
}

domainType *domainType::instance()
{
    if(inst == NULL)
    {
        inst = new domainType();
    }
    return inst;
    }

domainType *domainType::inst = NULL;
