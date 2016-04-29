#include <stdio.h>
#include "include/geo.h"
#include "include/common.h"

static const char *gt_type[] =
{
   "none",
   "tube",
   "radial runner",
   "diagonal runner",
   "axial runner",
   "gate"
};

char *GT_Type(int ind)
{
   if (ind >= 0 && ind < Num_GT_Type())
      return (char *)gt_type[ind];
   return NULL;
}


int Num_GT_Type(void)
{
   return DIM(gt_type);
}
