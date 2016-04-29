#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include "include/fatal.h"
#include "include/ihs_cfg.h"
#include "include/iso.h"

#define GLOBAL "[global]"
#define DOPLOT "plot"

struct Isofield * IsoRead(char *fn)
{
   int i, j;
   int ncalc;
   float *calc;
   struct Isofield *iso;
   struct stat st;

   if (stat((const char *)fn, &st))
      return NULL;

   // read measured data
   iso = ReadMeasured(fn);

   // read chosen calc data
   ncalc = ReadCalc(fn, &calc);

   // prepare calculation choice
   for (i = 0; i < ncalc; i++)
      for (j = 0; j < iso->num; j++)
   {
      if (calc[i] < iso->ic[j]->isoval)
      {
         CalcIsocurve(calc[i], iso);
         SortIsofield(iso);
         break;
      }
      else if (calc[i] == iso->ic[j]->isoval)
      {
         iso->ic[j]->calc = 1;
         break;
      }
      else if (j == iso->num-1)
      {
         CalcIsocurve(calc[i], iso);
         SortIsofield(iso);
      }
   }
   return iso;
}


int IsoGetIndex(struct Isofield *iso, float val)
{
   int i;

   for (i = 0; i < iso->num; i++)
      if (iso->ic[i]->isoval == val)
         return i;
   return -1;                                     // error ...
}
