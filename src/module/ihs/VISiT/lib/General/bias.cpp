#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "include/bias.h"
#include "include/flist.h"
#include "include/fatal.h"

struct Flist *CalcBladeElementBias(int nodes, float t1, float t2, int type, float ratio)
{
   int i, odd;
   float mstep, s1, s2, ds, step;
   float parval = 0;
   struct Flist *bias = NULL;

   bias = AllocFlistStruct(nodes+1);
   odd = nodes % 2;
   mstep = (t2 - t1)/(nodes - 1);
   if (ratio < 0) ratio = -1/ratio;

   s1 = s2 = 0.0;
   switch (type)
   {
      case 0:                                     // no bias
         break;
      case 1:                                     // one-way bias
         s1 = 2 * mstep * ratio/(ratio + 1);
         s2 = 2 * mstep/(1 + ratio);
         break;
      case 2:                                     // two-way bias, odd/even number of nodes
         if (odd)
         {
            s1 = 2 * mstep * ratio/(ratio + 1);
            s2 = 2 * mstep/(1 + ratio);
         }
         else
         {
            s1 = 2 * (t2 - t1) * ratio/(ratio * nodes + nodes - 2);
            s2 = 2 * (t2 - t1)/(ratio * nodes + nodes - 2);
         }
         break;
      default:                                    // undefined bias type
         fatal((char *)"undefined bias type");
         break;
   }

   ds = s2 - s1;
   Add2Flist(bias, t1);
   parval = t1;
   for (i = 1; i < nodes; i++)
   {
      switch (type)
      {
         case 1:                                  // one-way bias
            step = s1 + ds * (i - 1)/(nodes-2);
            break;
         case 2:                                  // two-way bias, odd/even number of nodes
            if (odd)
            {
               if (i <= (nodes - 1)/2)
                  step = s1 + ds * 2 * (i - 1)/(nodes - 3);
               else
                  step = s2 - ds * 2 * (i - 1 - 0.5f * (nodes - 1))/(nodes - 3);
            }
            else
            {
               if (i <= (nodes/2))
                  step = s1 + ds * 2 * (i - 1)/(nodes - 2);
               else
                  step = s2 - ds * 2 * (i - 0.5f * nodes)/(nodes - 2);
            }
            break;
         default:                                 // equidistant spacing
            step = mstep;
      }
      parval += step;
      Add2Flist(bias, parval);
   }
   bias->list[nodes-1] = t2;
   return bias;
}


struct Flist *Add2Bias(struct Flist *bias, int nodes, float t1, float t2,
int type, float ratio, int first)
{
   int i;
   struct Flist *addtl;

   addtl = CalcBladeElementBias(nodes, t1, t2, type, ratio);

   for(i = first; i < nodes; i++)
   {
      Add2Flist(bias, addtl->list[i]);
   }

   FreeFlistStruct(addtl);
   return bias;
}
