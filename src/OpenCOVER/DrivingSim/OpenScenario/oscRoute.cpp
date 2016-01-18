/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#include <oscRoute.h>


using namespace OpenScenario;


strategyType::strategyType()
{
    addEnum("fastest", oscRoute::fastest);
    addEnum("shortest", oscRoute::shortest);
    addEnum("leastIntersections", oscRoute::leastIntersections);
    addEnum("random", oscRoute::random);
}

strategyType *strategyType::instance()
{
    if(inst == NULL)
    {
        inst = new strategyType();
    }
    return inst;
}

strategyType *strategyType::inst = NULL;
