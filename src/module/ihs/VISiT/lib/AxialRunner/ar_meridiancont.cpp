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
#include "../General/include/fatal.h"
#include "../General/include/v.h"
#include "../BSpline/include/bspline.h"
#include "include/axial.h"
#include "include/ar_meridiancont.h"

int CreateAR_MeridianContours(struct axial *ar)
// caller: CreateAR_Contours()
{
   float vec[3], base[3], p[3];
   int i;
   int j;
   int count;
   char *fn;
   FILE *fp1 = NULL;
   FILE *fp2 = NULL;

   fn =DebugFilename("ar_meridians.txt");
   if (fn && *fn)
   {
      if ((fp1 = fopen(fn, "w")) == NULL)
         dprintf(0, "CreateAR_MeridianContours(): file %s could not be opened\n", fn);
   }

   fn = DebugFilename("ar_meridianpar.txt");
   if (fn && *fn)
   {
      if ((fp2 = fopen(fn, "w")) == NULL)
         dprintf(0, "CreateAR_MeridianContours(): file %s could not be opened\n", fn);
   }

   // delete old data
   for (i = 1; i < ar->be_num-1; i++)
   {
      if (ar->me[i]->ml)
      {
         FreeCurveStruct(ar->me[i]->ml);
      }
      ar->me[i]->ml = AllocCurveStruct();
   }

   for (i = 0; i < ar->me[0]->ml->p->nump; i++)
   {
      base[0] = ar->me[0]->ml->p->x[i];
      base[1] = ar->me[0]->ml->p->y[i];
      base[2] = ar->me[0]->ml->p->z[i];
      vec[0]  = ar->me[ar->be_num-1]->ml->p->x[i] - base[0];
      vec[1]  = ar->me[ar->be_num-1]->ml->p->y[i] - base[1];
      vec[2]  = ar->me[ar->be_num-1]->ml->p->z[i] - base[2];
      for (j = 1; j < ar->be_num-1; j++ )
      {
         p[0] = base[0] + ar->me[j]->para * vec[0];
         p[1] = base[1] + ar->me[j]->para * vec[1];
         p[2] = base[2] + ar->me[j]->para * vec[2];
         AddCurvePoint(ar->me[j]->ml, p[0], p[1], p[2], 0.0, 0.0);
      }
      count = 0;
      if (!(i % 4) && fp2)
      {
         fprintf(fp2, "# meridian parameter line %d (index %d)\n", i, count);
         fprintf(fp2, "%f  ", ar->me[0]->ml->p->x[i]/ar->ref);
         fprintf(fp2, "%f  ", ar->me[0]->ml->p->y[i]/ar->ref);
         fprintf(fp2, "%f\n", ar->me[0]->ml->p->z[i]/ar->ref);
         fprintf(fp2, "%f  ", ar->me[ar->be_num-1]->ml->p->x[i]/ar->ref);
         fprintf(fp2, "%f  ", ar->me[ar->be_num-1]->ml->p->y[i]/ar->ref);
         fprintf(fp2, "%f  ", ar->me[ar->be_num-1]->ml->p->z[i]/ar->ref);
         fprintf(fp2, "\n\n");
         count++;
      }
   }

   for (i = 0; i < ar->be_num; i++)
      CalcCurveArclen(ar->me[i]->ml);

   for (i = 0; fp1 && i < ar->be_num; i++)
   {
      fprintf(fp1, "# blade element %d (index)\n", i);
      for (j = 0; j < ar->me[i]->ml->p->nump; j++)
      {
         fprintf(fp1, "%f  ", ar->me[i]->ml->p->x[j]/ar->ref);
         fprintf(fp1, "%f  ", ar->me[i]->ml->p->y[j]/ar->ref);
         fprintf(fp1, "%f  ", ar->me[i]->ml->p->z[j]/ar->ref);
         fprintf(fp1, "%f  ", ar->me[i]->ml->len[j]/ar->ref);
         fprintf(fp1, "%f\n", ar->me[i]->ml->par[j]);
      }
      fprintf(fp1, "\n\n");
   }
   if (fp1) fclose(fp1);
   if (fp2) fclose(fp2);

   return 1;
}
