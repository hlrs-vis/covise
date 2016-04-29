#include <stdio.h>
#include <stdlib.h>
#include "include/fatal.h"
#include "include/ihs_cfg.h"
#include "include/iso.h"

#define GLOBAL    "[global]"
#define NO_MEASRD "no_measured"
#define ISO_CURVE "[measured%d]"
#define ISO_VAL      "isoval"
#define ISO_NOP      "no_points"
#define ISO_POINT "point%d"

struct Isofield * ReadMeasured(char *fn)
{
   int i, j;
   int niso = 0, np = 0;
   float iv, off, val1, val2;
   struct Isofield *isof;
   char *buf;
   char curve[100], point[100];

   if (!fn)
      fatal("Missing data file for isovalues.");

   // read global section: no_isocurves
   if ((buf = IHS_GetCFGValue(fn, GLOBAL, NO_MEASRD)) != NULL)
   {
      sscanf(buf, "%d", &niso);
      free(buf);
   }
   else
      fprintf(stderr, "WARNING: No measured values !! (ReadMeasured)\n");

   // read measured data
   isof = AllocIsofieldStruct();
   for (i = 0; i < niso; i++)
   {
      sprintf(curve, ISO_CURVE, i+1);
      if ((buf = IHS_GetCFGValue(fn, curve, ISO_VAL)) != NULL)
      {
         sscanf(buf, "%f", &iv);
         AddIsocurve(isof, iv);
         free(buf);
      }
      if ((buf = IHS_GetCFGValue(fn, curve, ISO_NOP)) != NULL)
      {
         sscanf(buf, "%d", &np);
         for (j = 0; j < np; j++)
         {
            sprintf(point, ISO_POINT, j+1);
            if ((buf = IHS_GetCFGValue(fn, curve, point)) != NULL)
            {
               sscanf(buf, "%f, %f, %f", &off, &val1, &val2);
               AddIsotupel(isof->ic[i], off, &val1, &val2);
               free(buf);
            }
         }
      }
   }
   return isof;
}
