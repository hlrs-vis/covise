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
#include "../General/include/points.h"
#include "../General/include/flist.h"
#include "../General/include/curve.h"
#include "../General/include/bias.h"
#include "../General/include/coordtrans.h"
#include "../General/include/common.h"
#include "../General/include/v.h"
#include "include/axial.h"
#include "include/ar_arclenbe.h"

int ArclenAR_BladeElements(struct axial *ar)
//caller: ReadAxialRunner()
{
   int i, num;
   float rmin, rmax, rad;
   float xsec, xedge, bl_arc, lec, tec;
   const float coeff = 0.4f;                       // coefficient for constriction calculation

   bl_arc = 2.0f * (float)M_PI / ar->nob * ar->enlace;
   num    = ar->be_num-1;
   rmin   = ar->be[0]->rad;
   rmax   = ar->be[ar->be_num-1]->rad;

   for (i = 0; i < ar->be_num; i++)
   {
      xsec = ar->be[i]->para;
      rad  = ar->be[i]->rad;
      // leading edge constriction
      if (ar->le->nocon != 0)
      {
         if (xsec <= ar->le->nocon)
         {
            xedge = 1 - xsec/ar->le->nocon;
            lec   = ar->le->con[0] * bl_arc;
         }
         else                                     // (xsec > ar->le->nocon)
         {
            xedge = (xsec - ar->le->nocon)/(1.0 - ar->le->nocon);
            lec   = ar->le->con[1] * bl_arc;
         }
         lec *= (coeff * pow(xedge, 3) + (1.0 - coeff) * pow(xedge, 2));
      }
      else                                        // (ar->le->nocon == 0)
      {
         lec  = bl_arc / rad;
         lec *= (rmin * ar->le->con[0]  + (rmax * ar->le->con[1] - rmin * ar->le->con[0]) * xsec);
      }
      ar->be[i]->lec = lec;

      // trailing edge constriction
      if ((ar->te->nocon != 0) && (ar->te->nocon != 1.0))
      {
         if (xsec <= ar->te->nocon)
         {
            xedge = 1.0 - xsec / ar->te->nocon;
            tec   = ar->te->con[0] * bl_arc;
         }
         else                                     // (xsec > ar->te->nocon)
         {
            xedge = (xsec - ar->te->nocon) / (1.0 - ar->te->nocon);
            tec   = ar->te->con[1] * bl_arc;
         }
         tec *= (coeff * pow(xedge, 3) + (1.0 - coeff) * pow(xedge, 2));
      }
      else                                        // ((ar->te->nocon == 0) || (ar->te->nocon == 1))
      {
         tec  = bl_arc / rad;
         tec *= (rmin * ar->te->con[0]  + (rmax * ar->te->con[1] - rmin * ar->te->con[0]) * xsec);
      }
      ar->be[i]->tec = tec;

      // resulting arc angle
      ar->be[i]->bl_wrap = bl_arc - (lec + tec);
   }
   return 0;
}
