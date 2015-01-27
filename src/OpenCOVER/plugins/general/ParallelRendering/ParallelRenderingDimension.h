/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef PARALLELRENDERING_DIMENSION_H
#define PARALLELRENDERING_DIMENSION_H

#include "ParallelRenderingDefines.h"

struct ParallelRenderingDimension
{

    int width;
    int height;

    ParallelRenderingDimension(int w, int h)
    {
        width = w;
        height = h;
    }
    ParallelRenderingDimension()
    {
        width = 0;
        height = 0;
    }
};
#endif
