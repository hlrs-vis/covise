#include <stdio.h>
#include <stdlib.h>
#include "include/fatal.h"
#include "include/ihs_cfg.h"
#include "include/iso.h"

#define GLOBAL    "[global]"
#define NO_CALC      "no_calculate"
#define ISO_CALC  "[calculate]"
#define CALC_VAL  "calc%d"

int ReadCalc(char *fn, float **calc)
{
   int i;
   int ncalc = 0;
   char *buf;
   char text[100];

   if ((buf = IHS_GetCFGValue(fn, GLOBAL, NO_CALC)) != NULL)
   {
      sscanf(buf, "%d", &ncalc);
      free(buf);
   }
   else
   {
      fprintf(stderr, "WARNING: No calculated values !! (ReadCalc)\n");
      ncalc = 0;
   }
   if ((*calc = (float *)calloc(ncalc, sizeof(float))) == NULL)
      fatal("Error allocating memory: (float *)calc");
   for (i = 0; i < ncalc; i++)
   {
      sprintf(text, CALC_VAL, i+1);
      if ((buf = IHS_GetCFGValue(fn, ISO_CALC, text)) != NULL)
      {
         sscanf(buf, "%f", calc[0]+i);
         free(buf);
      }
   }

#ifdef DEBUG
   for (i = 0; i < ncalc; i++)
      printf("calc[%d] = %8.4f\n", i, calc[0][i]);
#endif                                         // DEBUG

   return ncalc;
}
