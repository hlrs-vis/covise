/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <include/cov.h>
#include <include/points.h>

struct covise_info *AllocCoviseInfo()
{
    struct covise_info *ci;

    if ((ci = (struct covise_info *)calloc(1, sizeof(struct covise_info))) != NULL)
    {

        ci->p = AllocPointStruct();
        ci->cpol = AllocIlistStruct(50);
        ci->cvx = AllocIlistStruct(150);
        ci->pol = AllocIlistStruct(300);
        ci->vx = AllocIlistStruct(900);
        ci->lpol = AllocIlistStruct(100);
        ci->lvx = AllocIlistStruct(300);
    }
    else
    {
        fprintf(stderr, "not enough space to allocate ci");
    }
    return ci;
}

void FreeCoviseInfo(struct covise_info *ci)
{
    if (ci)
    {
        FreePointStruct(ci->p);
        FreeIlistStruct(ci->cpol);
        FreeIlistStruct(ci->cvx);
        FreeIlistStruct(ci->pol);
        FreeIlistStruct(ci->vx);
        FreeIlistStruct(ci->lpol);
        FreeIlistStruct(ci->lvx);

        free(ci);
    }
}
