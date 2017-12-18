#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../General/include/fatal.h"
#include "../General/include/log.h"
#include "../General/include/points.h"
#include "../General/include/flist.h"
#include "include/bspline.h"

struct Flist *BSplineKnot(struct Point *d, int deg)
{
   int   i, deg_mod, deg_int;
   float len, dist;
   float *para;
   float tol_abs = 0.000001f;
   struct Flist *t;

   // determine length of coordinate polygon
   if ((para = (float *)calloc(d->nump, sizeof(float))) == NULL)
      fatal("Error allocating memory: (float *)para");
   para[0] = len = 0.0;
   for (i = 1; i < d->nump; i++)
   {
      dist  = 0.0;
      dist += float(pow((d->x[i] - d->x[i-1]), 2));
      dist += float(pow((d->y[i] - d->y[i-1]), 2));
      dist += float(pow((d->z[i] - d->z[i-1]), 2));
      len  += float(sqrt(dist));
      para[i] = len;
   }

   // error message, if len less than tolerance
   if (len < tol_abs)
   {
      dprintf (0, "*** ERROR calculating polygon length.\n");
      dprintf (0, "*** Line %d in File %si.\n\n", __LINE__, __FILE__);
      dprintf (0, " Polygon length is less than tolerance:\n");
      dprintf (0, " length:    %8.4f\n", len);
      dprintf (0, " tolerance: %8.6f\n", tol_abs);
      exit(1);
   }

   // parametrization of control point sequence: 0..1
   for (i = 0; i < d->nump; i++)
      para[i] /= len;

   // assign bspline knots: 1st deg knots = 0, last deg knots = 1
   deg_mod = deg%2;
   deg_int = (int)(deg/2);

   t = AllocFlistStruct(d->nump+deg);
   for (i = 0; i < deg; i++)
      Add2Flist(t, para[0]);
   for (i = deg; i < d->nump; i++)
   {
      if (deg_mod)
         Add2Flist(t, (0.5f*(para[i-deg_int]+para[i-1-deg_int])));
      else
         Add2Flist(t, para[i-deg_int]);
   }
   for (i = d->nump; i < (d->nump+deg); i++)
      Add2Flist(t, para[d->nump-1]);

   free(para);
   return(t);
}
