/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _LIC_UTIL_H
#define _LIC_UTIL_H

#include <api/coModule.h>
using namespace covise;

#include "nrutil.h"
#include "Triangles.h"
//#include "RungeKutta.h"

/*****************************\ 
 *                           *
 *  place the typedefs here  *
 *                           *
\*****************************/

//see W.H. Press et al.: Numerical Recipes in C, 2nd edition, pg. 282
//period > 2*10^8 (or 2*10**8 for "FORTRANers)
float random2(long *idum);
int ran2int(float number, short psize);

//function unfortunately used though obsolte, see "nrutil.h"
inline float max(float a, float b)
{
    return ((a > b) ? a : b);
};

void triPack2polygons(coOutputPort **packageOutPort, coDoPolygons **package, trivec &triangles);

void heapSort(fvec &height, ivec &index, int tsize);

void siftDown(fvec &height, ivec &index, int root, int bottom);

void reverse(fvec &height, ivec &index, int tsize);
#endif
