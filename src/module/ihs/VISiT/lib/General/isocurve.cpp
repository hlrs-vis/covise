#include <stdlib.h>
#include "include/fatal.h"
#include "include/iso.h"

#define NP 10                                     // number of interpolation points

struct Isofield * AllocIsofieldStruct(void)
{
   return ((struct Isofield *)calloc(1, sizeof(struct Isofield)));
}


int AddIsocurve(struct Isofield *isof, float isoval)
{
   if ((isof->num+1) >= isof->max)
   {
      isof->max += 10;
      if ((isof->ic = (struct Isocurve **)realloc(isof->ic, isof->max*sizeof(struct Isocurve *))) == NULL)
         fatal("Space in AddIsocurve(): (struct Isocurve *)isof->ic");
   }
   if ((isof->ic[isof->num] = (struct Isocurve *)calloc(1, sizeof(struct Isocurve))) == NULL)
      fatal("Space in AddIsocurve(): (struct Isocurve)isof->ic[i]");
   isof->ic[isof->num]->isoval = isoval;
   isof->ic[isof->num]->measured = 1;
   isof->ic[isof->num]->calc = 0;
   isof->num++;
   return (isof->num-1);
}


int AddIsotupel(struct Isocurve *ic, float offset, float *val1, float *val2)
{
   if ((ic->num+1) >= ic->max)
   {
      ic->max += 100;
      if ((ic->offset = (float *)realloc(ic->offset, ic->max*sizeof(float))) == NULL)
         fatal("Space in AddIsopair(): ic->offset");
      if ((ic->val1 = (float *)realloc(ic->val1, ic->max*sizeof(float))) == NULL)
         fatal("Space in AddIsopair(): ic->val1");
      if ((ic->val2 = (float *)realloc(ic->val2, ic->max*sizeof(float))) == NULL)
         fatal("Space in AddIsopair(): ic->val2");
   }
   ic->offset[ic->num] = offset;
   ic->val1[ic->num] = *val1;
   ic->val2[ic->num] = *val2;
   ic->num++;
   return (ic->num-1);
}


void FreeIsocurveStruct(struct Isocurve *ic)
{
   if (ic->num && ic->offset) free(ic->offset);
   if (ic->num && ic->val1)   free(ic->val1);
   if (ic->num && ic->val2)   free(ic->val2);
   free(ic);
}


void FreeIsofieldStruct(struct Isofield *isof)
{
   int i;

   for (i = 0; i < isof->num; i++)
      FreeIsocurveStruct(isof->ic[i]);
   free(isof);
}


int CalcIsocurve(float isoval, struct Isofield *isof)
{
   int i, num;
   float off;
   float val1, val2;
   float off_min = 1.0e+6;
   float off_max = -1.0e+6;
   struct Isocurve *ic_min, *ic_max;

   for (i = 1; i < isof->num; i++)
   {
      if (isof->ic[i]->isoval >= isoval)
      {
         ic_min = isof->ic[i-1];
         ic_max = isof->ic[i];
         break;
      }
      else if (i == isof->num-1)
      {
         ic_min = isof->ic[isof->num-2];
         ic_max = isof->ic[isof->num-1];
      }
   }

   // determine min/max offset
   if (ic_min->offset[0] > ic_max->offset[0])
      off_min = ic_max->offset[0];
   else
      off_min = ic_min->offset[0];

   if (ic_min->offset[ic_min->num-1] < ic_max->offset[ic_max->num-1])
      off_max = ic_max->offset[ic_max->num-1];
   else
      off_max = ic_min->offset[ic_min->num-1];

   num = AddIsocurve(isof, isoval);
   isof->ic[num]->measured = 0;
   isof->ic[num]->calc = 1;

   // interpolate new isocurve with NP points
   for (i = 0; i < NP; i++)
   {
      off = off_min + (off_max - off_min)/(NP - 1)*i;
      CalcIsotupel(ic_min, ic_max, off, isoval, &val1, &val2);
      AddIsotupel(isof->ic[num], off, &val1, &val2);
   }
   return num;
}


void CalcIsotupel(struct Isocurve *ic_min, struct Isocurve *ic_max, float off, float isoval, float *rv1, float *rv2)
{
   int i, imin, imax;
   float val1[2], val2[2];

   for (i = 0; i < ic_min->num; i++)
   {
      if (ic_min->offset[0] >= off)
      {
         imin = 0;
         imax = 1;
         break;
      }
      else if (ic_min->offset[ic_min->num-1] <= off)
      {
         imin = ic_min->num-2;
         imax = ic_min->num-1;
         break;
      }
      else if (ic_min->offset[i] >= off)
      {
         imin = i-1;
         imax = i;
         break;
      }
   }
   val1[0]  = (ic_min->val1[imin] - ic_min->val1[imax]);
   val1[0] /= (ic_min->offset[imin] - ic_min->offset[imax]);
   val1[0] *= (off - ic_min->offset[imax]);
   val1[0] += ic_min->val1[imax];

   val2[0]  = (ic_min->val2[imin] - ic_min->val2[imax]);
   val2[0] /= (ic_min->offset[imin] - ic_min->offset[imax]);
   val2[0] *= (off - ic_min->offset[imax]);
   val2[0] += ic_min->val2[imax];

   for (i = 0; i < ic_max->num; i++)
   {
      if (ic_max->offset[0] >= off)
      {
         imin = 0;
         imax = 1;
         break;
      }
      else if (ic_max->offset[ic_max->num-1] <= off)
      {
         imin = ic_max->num-2;
         imax = ic_max->num-1;
         break;
      }
      else if (ic_max->offset[i] >= off)
      {
         imin = i-1;
         imax = i;
         break;
      }
   }
   val1[1]  = (ic_max->val1[imin] - ic_max->val1[imax]);
   val1[1] /= (ic_max->offset[imin] - ic_max->offset[imax]);
   val1[1] *= (off - ic_max->offset[imax]);
   val1[1] += ic_max->val1[imax];

   val2[1]  = (ic_max->val2[imin] - ic_max->val2[imax]);
   val2[1] /= (ic_max->offset[imin] - ic_max->offset[imax]);
   val2[1] *= (off - ic_max->offset[imax]);
   val2[1] += ic_max->val2[imax];

   *rv1  = (val1[0] - val1[1]);
   *rv1 /= (ic_min->isoval - ic_max->isoval);
   *rv1 *= (isoval - ic_max->isoval);
   *rv1 += val1[1];

   *rv2  = (val2[0] - val2[1]);
   *rv2 /= (ic_min->isoval - ic_max->isoval);
   *rv2 *= (isoval - ic_max->isoval);
   *rv2 += val2[1];

   return;
}


void SortIsofield(struct Isofield *isof)
{
   qsort(isof->ic, isof->num, sizeof(struct Isocurve *), isocmp);
}


int isocmp(const void *ae, const void  *be)
{
   struct Isocurve **a;
   struct Isocurve **b;

   a = (struct Isocurve **)ae;
   b = (struct Isocurve **)be;

   if ((*a)->isoval > (*b)->isoval)
      return 1;
   else if ((*a)->isoval < (*b)->isoval)
      return -1;
   else
      return 0;
}


#ifdef   DEBUG
void DumpIsocurve(struct Isocurve *ic, FILE *fp)
{
   int j, lc;

   if (fp)
      fprintf(fp, "# isoval =%10.6f (%s)\n", ic->isoval, (ic->measured) ? "measured" : "interpolated");
   fprintf(stderr, "isoval =%10.6f (%s)\n", ic->isoval, (ic->measured) ? "measured" : "interpolated");
   for (j = 0, lc = 0; j < ic->num; j++)
   {
      if (fp)  fprintf(fp, "%10.6f %10.6f %10.6f\n", ic->offset[j], ic->val1[j], ic->val2[j]);
      fprintf(stderr, "j =%3d: offset =%10.6f val1 =%10.6f val2=%10.6f\n", j, ic->offset[j], ic->val1[j], ic->val2[j]);
   }
}
#endif
