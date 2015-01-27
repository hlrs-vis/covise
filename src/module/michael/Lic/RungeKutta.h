/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _RUNGE_KUTTA_H
#define _RUNGE_KUTTA_H

//see W.H. Press et al, Numerical Recipes in C, 2nd Edition
//Chap. 16.2: Adaptive Stepsize Control for Runge-Kutta, pg. 719-720

//#include <math.h>
//#include <iostream.h>
//#include <stdlib.h>
//#include <stdio.h>
//#include <ctype.h>
//#include <vector.h>
//#include <string.h>
#include "nrutil.h"

const float SAFETY = 0.9;
const float PGROW = -0.2;
const float PSHRNK = -0.25;
const float ERRCON = 1.89e-4;
//ERRCON = pow( (5/SAFETY), (1/PGROW) )

fvec derivative(float t, const fvec &s, const fvec &v);
void derivs(float t, const fvec &s, const fvec &ds);

void rkqs(fvec s, fvec ds, int n, float *t, float htry, float eps, fvec sscal, float *hdid, float *hnext);
void rkck(fvec s, fvec v, int n, float t, float h, fvec sout, fvec serr);
#endif
