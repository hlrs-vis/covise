/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __NUMERIC_H__
#define __NUMERIC_H__

double Romberg(double (*fkt)(double), double a, double b, int N);
double Simpson(double (*fkt)(double), double t1, double t2, int N);

double HlFresnelSinus(double t);
double HlFresnelCosinus(double t);

double HlErf(double t);

double HlSinc(double t);
double HlSi(double t);

double HlSign(double z);
double HlSignum(double z, int n);
double HlBetrag(double z, int n);

#endif // __NUMERIC_H__
