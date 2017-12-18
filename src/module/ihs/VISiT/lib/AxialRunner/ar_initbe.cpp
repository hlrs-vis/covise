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
#include "../General/include/log.h"
#include "../BSpline/include/bspline.h"
#include "include/axial.h"
#include "include/ar_initbe.h"

#define INIT_PORTION    25

int InitAR_BladeElements(struct axial *ar)
// caller: ReadAxialRunner()
{
   int i;
   struct Flist *bias;

   bias = CalcBladeElementBias(ar->be_num, 0.0, 1.0, ar->be_type, ar->be_bias);
   dprintf(6, "reiFE: ar->hubcap->x[0]=%f\n", ar->p_hubcap->x[0]);

#ifdef DEBUG
   DumpFlist(bias);
#endif
   // initialize blade elements (cylindrical planes)
   if ((ar->be = (struct be **)calloc(ar->be_num, sizeof(struct be *))) == NULL)
      fatal("memory for (struct be*)");
   for (i = 0; i < ar->be_num; i++)
   {
      dprintf(6, "reiFE: ar->hubcap->x[0]=%f (i=%d)\n", ar->p_hubcap->x[0], i);
      if ((ar->be[i] = (struct be *)calloc(1, sizeof(struct be))) == NULL)
         fatal("memory for (struct be)");
      // assign bias and abs. radius to blade elements
      ar->be[i]->para = bias->list[i];
      ar->be[i]->rad  = 0.5f * ar->ref * (ar->diam[0] + (1.0f - ar->diam[0]) * ar->be[i]->para);
      // interpolate parameter set data to blade elements
      ar->be[i]->pivot        = ar->piv;
      ar->be[i]->angle[0]     = InterpolateParameterSet(ar->iang, ar->be[i]->para, ar->extrapol);
      ar->be[i]->mod_angle[0] = InterpolateParameterSet(ar->mod_iang, ar->be[i]->para, ar->extrapol);
      ar->be[i]->angle[1]     = InterpolateParameterSet(ar->oang, ar->be[i]->para, ar->extrapol);
      ar->be[i]->mod_angle[1] = InterpolateParameterSet(ar->mod_oang, ar->be[i]->para, ar->extrapol);
      ar->be[i]->p_thick      = InterpolateParameterSet(ar->t, ar->be[i]->para, ar->extrapol);
      ar->be[i]->te_thick     = InterpolateParameterSet(ar->tet, ar->be[i]->para, ar->extrapol);
      ar->be[i]->camb         = InterpolateParameterSet(ar->camb, ar->be[i]->para, ar->extrapol);
      ar->be[i]->camb_pos     = InterpolateParameterSet(ar->camb_pos, ar->be[i]->para, ar->extrapol);
      ar->be[i]->bp_shift     = InterpolateParameterSet(ar->bps, ar->be[i]->para, ar->extrapol);

      // prepare blade profile data
      ar->be[i]->bp = AllocBladeProfile();
      AssignBladeProfile(ar->bp, ar->be[i]->bp);
      ShiftBladeProfile(ar->be[i]->bp, ar->be[i]->bp_shift);
   }

   // prepare meridional data
   if ((ar->me = (struct meridian **)calloc(ar->be_num, sizeof(struct meridian *))) == NULL)
      fatal("memory for (struct meridian *)");
   for (i = 0; i < ar->be_num; i++)
   {
      if ((ar->me[i] = (struct meridian *)calloc(1, sizeof(struct meridian))) == NULL)
         fatal("memory for (struct meridian)");
      // assign bias
      dprintf(6, "reiFE: ar->hubcap->x[0]=%f (i=%d)\n", ar->p_hubcap->x[0], i);
      ar->me[i]->para = bias->list[i];
      // memory for meridional lines
      ar->me[i]->ml = AllocCurveStruct();
      // memory for conduit areas
      ar->me[i]->area = AllocFlistStruct(INIT_PORTION);
   }
   ar->be_single = 1;
   for(i = 0; i < NUM_PARLOCK; i++) ar->parlock[i] = 0;
   //FreeFlistStruct(bias);
   return 0;
}
