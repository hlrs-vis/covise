#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "include/coordtrans.h"

#ifdef WIN32
#pragma warning (disable : 4244)
#endif

int CalcCylindricalCoords(float *x)
{
   double r;

   r    = sqrt(pow(x[0], 2) + pow(x[1], 2));
   if (x[1] >= 0.0)
   {
      x[1] = acos(x[0] / r);
   }
   else
   {
      x[1] = -acos(x[0] / r);
   }
   x[0] = r;
   x[2] = x[2];

   return 1;
}


int CalcCylindricalCoords2(float *x, float r)
{
   x[2]  = x[1];
   x[1]  = x[0]/r;
   x[0]  = r;

   return 1;
}


int CalcCartesianCoords(float *x)
{
   float r;

   r    = x[0];
   x[0] = r * cos(x[1]);
   x[1] = r * sin(x[1]);
   x[2] = x[2];

   return 1;
}
