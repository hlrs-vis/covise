/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <include/points.h>
#include <include/ilist.h>

struct covise_info
{
    struct Point *p;

    struct Ilist *cpol;
    struct Ilist *cvx;

    // this is for the entry surface
    int bcinnumPoints;
    struct Ilist *bcinpol;
    struct Ilist *bcinvx;

    // this is the geometry
    struct Ilist *pol;
    struct Ilist *vx;

    // this is the geometry (in line format)
    struct Ilist *lpol;
    struct Ilist *lvx;
};

extern struct covise_info *AllocCoviseInfo();
extern void FreeCoviseInfo(struct covise_info *ci);
