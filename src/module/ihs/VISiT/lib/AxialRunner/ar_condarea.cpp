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
#include "../General/include/geo.h"
#include "../General/include/log.h"
#include "../General/include/plane_geo.h"
#include "../General/include/points.h"
#include "../General/include/flist.h"
#include "../General/include/curve.h"
#include "../General/include/curvepoly.h"
#include "../General/include/profile.h"
#include "../General/include/parameter.h"
#include "../General/include/bias.h"
#include "../General/include/coordtrans.h"
#include "../General/include/common.h"
#include "../General/include/v.h"
#include "include/axial.h"
#include "include/ar_condarea.h"

int CreateAR_ConduitAreas(struct axial *ar)
// caller: CreateAR_Contours()
{
   int i, j;
   float area, rad;
   static int ncall = 0;
   char fname[255];
   char *fn;
   FILE *fp=NULL;

   sprintf(fname, "ar_area_%02d.txt", ncall++);
   fn = DebugFilename(fname);
   if(fn)
   fp = fopen(fn, "w");

   if (ar->area)
   {
      FreeFlistStruct(ar->area);
   }
   ar->area = AllocFlistStruct(INIT_PORTION);

#ifdef AREA_REGION_WISE
   for (i = 0; i < ar->p_hinlet->nump; i++)
   {
      area  = pow((ar->p_hinlet->x[i] - ar->p_sinlet->x[i]), 2);
      area += pow((ar->p_hinlet->y[i] - ar->p_sinlet->y[i]), 2);
      area += pow((ar->p_hinlet->z[i] - ar->p_sinlet->z[i]), 2);
      area  = sqrt(area);
      rad   = 0.5 * (ar->p_hinlet->x[i] + ar->p_sinlet->x[i]);
      area *= 2.0 * M_PI * rad;
      Add2Flist(ar->area, area);
   }
   for (i = 0; i < ar->p_hbend->nump; i++)
   {
      area  = pow((ar->p_hbend->x[i] - ar->p_sbend->x[i]), 2);
      area += pow((ar->p_hbend->y[i] - ar->p_sbend->y[i]), 2);
      area += pow((ar->p_hbend->z[i] - ar->p_sbend->z[i]), 2);
      area  = sqrt(area);
      rad   = 0.5 * (ar->p_hbend->x[i] + ar->p_sbend->x[i]);
      area *= 2.0 * M_PI * rad;
      Add2Flist(ar->area, area);
   }
   for (i = 0; i < ar->p_hcore->nump; i++)
   {
      area  = pow((ar->p_hcore->x[i] - ar->p_score->x[i]), 2);
      area += pow((ar->p_hcore->y[i] - ar->p_score->y[i]), 2);
      area += pow((ar->p_hcore->z[i] - ar->p_score->z[i]), 2);
      area  = sqrt(area);
      rad   = 0.5 * (ar->p_hcore->x[i] + ar->p_score->x[i]);
      area *= 2.0 * M_PI * rad;
      Add2Flist(ar->area, area);
   }
   for (i = 0; i < ar->p_houtlet->nump; i++)
   {
      area  = pow((ar->p_houtlet->x[i] - ar->p_soutlet->x[i]), 2);
      area += pow((ar->p_houtlet->y[i] - ar->p_soutlet->y[i]), 2);
      area += pow((ar->p_houtlet->z[i] - ar->p_soutlet->z[i]), 2);
      area  = sqrt(area);
      rad   = 0.5 * (ar->p_houtlet->x[i] + ar->p_soutlet->x[i]);
      area *= 2.0 * M_PI * rad;
      Add2Flist(ar->area, area);
   }
#else                                          // AREA_FROM_ENTIRE_CURVE
   for(i = 0; i < ar->me[0]->ml->p->nump; i++)
   {
      area  = float(pow((ar->me[0]->ml->p->x[i] - ar->me[ar->be_num-1]->ml->p->x[i]), 2));
      area += float(pow((ar->me[0]->ml->p->y[i] - ar->me[ar->be_num-1]->ml->p->y[i]), 2));
      area += float(pow((ar->me[0]->ml->p->z[i] - ar->me[ar->be_num-1]->ml->p->z[i]), 2));
      area  = float(sqrt(area));
      rad   = 0.5f * (ar->me[0]->ml->p->x[i] + ar->me[ar->be_num-1]->ml->p->x[i]);
      area *= float(2.0f * M_PI * rad);
      for (j = 0; j < ar->be_num; j++)
         Add2Flist(ar->me[j]->area, area);
   }
#endif                                         // AREA_REGION_WISE

   if (fp)
   {
#ifdef AREA_REGION_WISE
      for (i = 0; i < ar->area->num; i++)
      {
         fprintf(fp, "%d  ", i);
         fprintf(fp, "%f  ", (float)i/(ar->area->num-1));
         fprintf(fp, "%f\n", ar->area->list[i]);
      }
#else                                       // AREA_FROM_ENTIRE_CURVE
      for (i = 0; i < ar->me[0]->area->num; i++)
      {
         fprintf(fp, "%d  ", i);
         fprintf(fp, "%f  ", (float)i/(ar->me[0]->area->num-1));
         fprintf(fp, "%f\n", ar->me[0]->area->list[i]);
      }
#endif
      fclose(fp);
   }

   return 1;
}
