#include <stdio.h>
#include <stdlib.h>
#include "include/fatal.h"
#include "include/ihs_cfg.h"
#include "include/iso.h"
#include "include/plot.h"

#define GLOBAL "[global]"
#define DOPLOT "plot"

int main(int argc, char **argv)
{
   int i, j;
   int ncalc, doplot;
   float *calc;
   struct Isofield *isof;
   char *fn, *buf;

   // check, if an accessible cfg file is entered
   if (argc < 2)
   {
      fprintf(stderr, "Data file missing:\nUsage: inter <data file>\n");
      exit(1);
   }
   else
      fn = argv[1];
   if (fopen(fn, "r") == NULL)
   {
      fprintf (stderr, "Error opening file %s\n", fn);
      exit(1);
   }

   // read measured data
   isof = ReadMeasured(fn);
   SortIsofield(isof);

   // read chosen calc data
   ncalc = ReadCalc(fn, &calc);

   // prepare calculation choice
   for (i = 0; i < ncalc; i++)
      for (j = 0; j < isof->num; j++)
   {
      if (calc[i] < isof->ic[j]->isoval)
      {
         CalcIsocurve(calc[i], isof);
         SortIsofield(isof);
         break;
      }
      else if (calc[i] == isof->ic[j]->isoval)
      {
         isof->ic[j]->calc = 1;
         break;
      }
      else if (j == isof->num-1)
      {
         CalcIsocurve(calc[i], isof);
         SortIsofield(isof);
      }
   }

   // plot all ?
   if ((buf = IHS_GetCFGValue(fn, GLOBAL, DOPLOT)) != NULL)
   {
      sscanf(buf, "%d", &doplot);
      free(buf);
   }

   // plot to screen or via gnuplot
   if (doplot)
      PlotIsofield(isof);
   else for (i = 0; i < isof->num; i++)
   {
#ifdef DEBUG
      DumpIsocurve(isof->ic[i], NULL);
      printf("\n");
#endif
   }

   FreeIsofieldStruct(isof);

   return 0;
}
