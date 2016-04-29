#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "include/cov.h"
#include "include/points.h"
#include "include/fatal.h"

#define  P_POINTS 1000
#define  P_VERTEX 4000
#define  P_POLYGONS  P_VERTEX/4

#define  P_POINTS_MIN    P_POINTS/10
#define  P_VERTEX_MIN    P_VERTEX/10
#define  P_POLYGONS_MIN  P_POLYGONS/10

struct ci_cs *AllocCiCsStruct(void)
{
   struct ci_cs *ci_cs;

   if ((ci_cs = (struct ci_cs *)calloc(1, sizeof(struct ci_cs))) != NULL)
   {
      ci_cs->p       = AllocPointStruct();
      ci_cs->cvx     = AllocIlistStruct(30);
   }
   return ci_cs;
}


void FreeCiCsStruct(struct ci_cs *c)
{
   FreePointStruct(c->p);
   FreeIlistStruct(c->cvx);
}


struct covise_info *AllocCoviseInfo(int cs_num)
{
   struct covise_info *ci;

   if ((ci = (struct covise_info *)calloc(1, sizeof(struct covise_info))) != NULL)
   {

      ci->p       = AllocPointStruct();
      ci->bcinpol = AllocIlistStruct(10);
      ci->bcinvx  = AllocIlistStruct(30);
      ci->cpol    = AllocIlistStruct(50);
      ci->cvx     = AllocIlistStruct(150);
      if (cs_num)
         ci->ci_cs   = (struct ci_cs **)calloc(cs_num, sizeof(struct ci_cs *));
      ci->num_cs  = cs_num;
      ci->pol     = AllocIlistStruct(300);
      ci->vx      = AllocIlistStruct(900);

      ci->lpol    = AllocIlistStruct(100);
      ci->lvx     = AllocIlistStruct(300);

   }
   else
   {
      fatal((char *)"Space ci");
   }
   return ci;
}


void FreeCoviseInfo(struct covise_info *ci)
{
   int i;

   if (ci)
   {
      FreePointStruct(ci->p);
#ifdef   DRAFT_TUBE
      FreeIlistStruct(ci->bcinpol);
      FreeIlistStruct(ci->bcinvx);
#endif
      FreeIlistStruct(ci->cpol);
      FreeIlistStruct(ci->cvx);
      FreeIlistStruct(ci->pol);
      FreeIlistStruct(ci->vx);
#if (defined(AXIAL_RUNNER)||defined(RADIAL_RUNNER)||defined(DIAGONAL_RUNNER))
      FreeIlistStruct(ci->lpol);
      FreeIlistStruct(ci->lvx);
#endif
      if (ci->ci_cs && ci->num_cs)
      {
         for (i = 0; i < ci->num_cs; i++)
            FreeCiCsStruct(ci->ci_cs[i]);
         free(ci->ci_cs);
      }
      free(ci);
   }
}
