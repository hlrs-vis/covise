#include <stdio.h>
#include <stdlib.h>
#ifndef WIN32
#include <strings.h>
#endif 
#ifdef WIN32
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#endif
#include <math.h>
#include "include/cfg.h"
#include "include/geo.h"
#include "include/profile.h"
#include "include/common.h"
#include "include/log.h"
#include "include/fatal.h"
#include "include/v.h"

#define BP_NACA      "naca style"
#define BP_STAT      "stat%d"

/* WARNING: if "realloc" is called in AddProfilePoint memory
   errors will occur somewhere later in the program! This has
   to be examined! F.L.10/2004 */

struct profile *AllocBladeProfile(void)
{
   struct profile *bp;

   if ((bp = (struct profile*)calloc(1, sizeof(struct profile))) != NULL)
   {
      bp->max = 45;
      if ((bp->c = (float *)calloc(bp->max, sizeof(float))) == NULL)
         fatal("memory for bp->c (float)");
      if ((bp->t = (float *)calloc(bp->max, sizeof(float))) == NULL)
         fatal("memory for bp->t (float)");
   }
   else
      fatal("memory for struct profile");

   return(bp);
}


int AddProfilePoint(struct profile *p, float c, float t)
{
   if ((p->num+1) >= p->max)
   {
      p->max += 10;
      if ((p->c = (float *)realloc(p->c, p->max*sizeof(float))) == NULL)
         fatal("Space in AddPoint(): p->c");
      if ((p->t = (float *)realloc(p->t, p->max*sizeof(float))) == NULL)
         fatal("Space in AddPoint(): p->t");
   }
   p->c[p->num] = c;
   p->t[p->num] = t;
   p->num++;
   return (p->num-1);
}


void FreeBladeProfile(struct profile *bp)
{
   if (bp->num && bp->c)   free(bp->c);
   if (bp->num && bp->t)   free(bp->t);
   free(bp);
}


int ReadBladeProfile(struct profile *bp, const char *sec, const char *fn)
{
   int i;
   float c, t, norm;
   char *tmp;
   char buf[100];

   for (i = 0; ; i++)
   {
      if (bp->num+1 > bp->max)
      {
         bp->max += 10;
         if ((bp->c = (float *)realloc(bp->c, bp->max*sizeof(float))) == NULL)
            fatal("memory for bp->c (float)");
         if ((bp->t = (float *)realloc(bp->t, bp->max*sizeof(float))) == NULL)
            fatal("memory for bp->t (float)");
      }
      else
      {
         sprintf(buf, BP_STAT, i);
         if ((tmp = IHS_GetCFGValue(fn, sec, buf)) != NULL)
         {
            sscanf(tmp, "%f, %f", &c, &t);
            free(tmp);
            bp->c[bp->num] = c;
            bp->t[bp->num] = t;
            if (bp->num && (bp->t[bp->num] > bp->t[bp->num-1])) bp->t_sec = bp->num;
            bp->num++;
         }
         else
            break;
      }
   }
   // profile data style, normalize if naca style
   if ((tmp = IHS_GetCFGValue(fn, sec, BP_NACA)) != NULL)
   {
      sscanf(tmp, "%d", &bp->naca);
      free(tmp);
   }
   if (bp->naca)
   {
      norm = bp->t[bp->t_sec];
      for (i = 0; i < bp->num; i++)
      {
         bp->c[i] /= bp->c[bp->num-1];
         bp->t[i] /= norm;
      }
   }

   return(bp->num);
}


int AssignBladeProfile(struct profile *bp1, struct profile *bp2)
{
   int i;

   bp2->num   = bp1->num;
   bp2->t_sec = bp1->t_sec;
   bp2->naca  = bp1->naca;
   for (i = 0; i < bp1->num; i++)
   {
      bp2->c[i] = bp1->c[i];
      bp2->t[i] = bp1->t[i];
   }

   return(bp2->num);
}


int ShiftBladeProfile(struct profile *bp, float shift)
{
   int i;

   // shift blade profile chord:
   // (shift > 1.0): shift to leading edge
   // (shift = 1.0): no shift
   // (shift < 1.0): shift to trailing edge
   for (i = 0; i < bp->num; i++)
      bp->c[i] = float(pow(bp->c[i], shift));

   return(0);
}


void DumpBladeProfile(struct profile *bp)
{
   int i;

   dprintf(1, "Entering DumpBladeProfile()\n");
   dprintf(5, "bp->num   = %d\n", bp->num);
   dprintf(5, "bp->max   = %d\n", bp->max);
   dprintf(5, "bp->t_sec = %d\n", bp->t_sec);
   dprintf(5, "bp->naca  = %d\n", bp->naca);
   for (i = 0; i < bp->num; i++)
      dprintf(5, "   bp->c[%02d] = %8.4f | bp->t[%02d] = %8.4f\n", i, bp->c[i], i, bp->t[i]);
   dprintf(1, "Leaving DumpBladeProfile()\n");
}
