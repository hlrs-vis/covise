/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _IP_LIC_UTIL_TRY_H
#define _IP_LIC_UTIL_TRY_H

////////////////////////////////////////////////////////////////////////

#include <api/coModule.h>
using namespace covise;

#include "nrutil.h"
#include "IPTriangles.h"

////////////////////////////////////////////////////////////////////////

const int SEED = -37; //for random2(...);
const int NHITS = 4; //fuer hitcount[][]
const float FLIMIT = 1e-12;

////////////////////////////////////////////////////////////////////////

void dimension(int *width, int *height, int nt);

//matrix(3,5) - col 4 -> row_scaling, col 5 -> col_pivoting !!
f2ten gauss3D(f2ten matrix);

//returns vector lambda
//bary(point, coord) = sum( i=1...3, lambda[i]*coord[i] )
//for vectors v1, v2, v3 defined on the three vertices create tensor
//v=(v1,v2,v3); then bary(point, v) = sum( i=1...3, lambda[i]*v[i] )
fvec bary(const fvec &point, const f2ten &coord);

fvec doLambda(const fvec &point, const f2ten &coord);

//returns vector of size 2
fvec vec(const fvec &lambda, const f2ten v);

//returns vector of size 2
fvec pstep(Triangles *triangle, fvec pos, float h);

void doFieldLine(trivec &triangles, const i2ten &neighbours, f3ten &accum, i2ten &hitcount, int res, float delta, int ii, int jj, int hh, int ww, long *seed);

fvec pix2pos(int i, int j, float delta);

ivec pos2pix(fvec pos, float delta);

void accum2pixels(const f3ten &accum, const i2ten &hitcount, char **image, int img_size);

int tInd(int ii, int jj, int ww, int hh, int res);

int getTri(trivec &triangles, const i2ten &neighbours, int tri, int first, int second, int third, fvec &posIn, fvec posOut, float tLength, float delta);
#endif
