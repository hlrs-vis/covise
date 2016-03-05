/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#include "oscShape.h"


using namespace OpenScenario;


purposeShapeType::purposeShapeType()
{
    addEnum("steering", oscShape::steering);
    addEnum("positioning", oscShape::positioning);
}

purposeShapeType *purposeShapeType::instance()
{
    if(inst == NULL)
    {
        inst = new purposeShapeType();
    }
    return inst;
}

purposeShapeType *purposeShapeType::inst = NULL;
