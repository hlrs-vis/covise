/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Shape.h"
#include <assert.h>

// delete list of shapes
void Shape::deleteShapeList(Shape **list)
{
    if (list)
    {
        Shape **ptr = list;
        while (*ptr)
            delete *ptr++;
        delete list;
    }
}
