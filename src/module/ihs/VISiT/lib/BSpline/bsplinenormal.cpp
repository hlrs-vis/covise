#include <stdio.h>
#include <stdlib.h>
#ifdef WIN32
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#endif
#include <math.h>
#include "../General/include/points.h"
#include "../General/include/flist.h"
#include "../General/include/fatal.h"
#include "include/bspline.h"

// ***   IMPORTANT   ***   IMPORTANT   ***   IMPORTANT ***   IMPORTANT   ***
//
// BSplineNormal() calculates the normal vector to the spline described
// by the sampling points in (struct Point*)d and the matching knot vector
// in (struct Flist*)t for the x-y-PLANE only (z coord = 0) !!!
//
// ***   IMPORTANT   ***   IMPORTANT   ***   IMPORTANT ***   IMPORTANT   ***

void BSplineNormal(int deg, struct Point *d, struct Flist *t, float t0, float *grad)
{
   const float eps = 0.001f;
   float p1[3], p2[3];
   float mag, temp;

   // two very close spline points
   BSplinePoint(deg, d, t, t0, &p1[0]);
   BSplinePoint(deg, d, t, (t0+eps), &p2[0]);
   // gradient at t0
   grad[0] = p2[0] - p1[0];
   grad[1] = p2[1] - p1[1];
   grad[2] = p2[2] - p1[2];
   // normalize
   mag      = float(sqrt(pow(grad[0], 2) + pow(grad[1], 2) + pow(grad[2], 2)));
   grad[0] /= mag;
   grad[1] /= mag;
   grad[2] /= mag;
   // normal to gradient (+M_PI/2 rotated)
   temp    =  grad[0];
   grad[0] = -grad[1];
   grad[1] =  temp;

   return;
}


void BSplineNormalXZ(int deg, struct Point *d, struct Flist *t, float t0, float *grad)
{
   const float eps = 0.001f;
   float p1[3], p2[3];
   float mag, temp;

   // two very close spline points
   BSplinePoint(deg, d, t, t0, p1);
   BSplinePoint(deg, d, t, (t0+eps), p2);
   // gradient at t0
   grad[0] = p2[0] - p1[0];
   grad[1] = p2[1] - p1[1];
   grad[2] = p2[2] - p1[2];
   // normalize
   mag      = float(sqrt(pow(grad[0], 2) + pow(grad[1], 2) + pow(grad[2], 2)));
   grad[0] /= mag;
   grad[1] /= mag;
   grad[2] /= mag;
   // normal to gradient (+M_PI/2 rotated)
   temp    =  grad[0];
   grad[0] = -grad[2];
   grad[2] =  temp;

   return;
}
