/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#include "oscLateral.h"


using namespace OpenScenario;


purposeLateralType::purposeLateralType()
{
    addEnum("position", oscLateral::position);
    addEnum("steering", oscLateral::steering);
    addEnum("navigation", oscLateral::navigation);
}

purposeLateralType *purposeLateralType::instance()
{
    if(inst == NULL)
    {
        inst = new purposeLateralType();
    }
    return inst;
}

purposeLateralType *purposeLateralType::inst = NULL;
