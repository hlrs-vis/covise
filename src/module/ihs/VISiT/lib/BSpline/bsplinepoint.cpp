#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../General/include/log.h"
#include "../General/include/points.h"
#include "../General/include/flist.h"
#include "../General/include/fatal.h"
#include "include/bspline.h"

void BSplinePoint(int deg, struct Point *d, struct Flist *t, float t0, float *x)
{
   int   i;
   int   intvl = d->nump - 1;
   float **D;
   char  buf[200];

   // knot interval of t0
   for (i = deg-1; i <= d->nump; i++)
   {
      if (t->list[i] > t0)
      {
         intvl = i-1;
         break;
      }
   }

   // deBoor algorithm for all coordinates
   if ((D = (float **)calloc((d->nump+1), sizeof(float *))) == NULL)
      fatal("Error allocating memory: (float *)D");
   for (i = 0; i < (d->nump+1); i++)
   {
      if ((D[i] = (float *)calloc(deg, sizeof(float))) == NULL)
      {
         sprintf(buf, "Error allocating memory: (float)D[i] = %d", i);
         fatal(buf);
      }
   }

   x[0] = deBoor(d->x, t, deg, t0, intvl, D);
   x[1] = deBoor(d->y, t, deg, t0, intvl, D);
   x[2] = deBoor(d->z, t, deg, t0, intvl, D);

   for (i = 0; i < (d->nump+1); i++)
      free(D[i]);
   free(D);
}


float deBoor(float *coord, struct Flist *t, int deg, float t0, int intvl, float **D)
{
   int   i,j;
   float alpha;

   // calculate spline value at t0
   for (i = (intvl-deg+1); i <= intvl; i++)
   {
      D[i][0] = coord[i];
   }
   for (j = 1; j < deg; j++)
   {
      for (i = (intvl-deg+1+j); i <= intvl; i++)
      {
         alpha   = (t0 - t->list[i])/(t->list[i+deg-j] - t->list[i]);
         D[i][j] = (1 - alpha)*D[i-1][j-1] + alpha*D[i][j-1];
         dprintf (6, "(1-a): %7.4f  D[%d][%d]: %7.4f\n", (1-alpha), i-1,
            j-1, D[i-1][j-1]);
         dprintf (6, "   a : %7.4f  D[%d][%d]: %7.4f\n", alpha, i, j-1,
            D[i][j-1]);
         dprintf (6, "                D[%d][%d]: %7.4f\n", i, j, D[i][j]);
      }
   }
   dprintf (6, "             => D[%d][%d]: %7.4f\n\n", intvl, deg-1,
      D[intvl][deg-1]);
   return(D[intvl][deg-1]);
}
