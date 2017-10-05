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
#include "../BSpline/include/bspline.h"
#include "include/axial.h"

#define NUM_BEND_POINTS 15

extern int CreateAR_InletArbitrary(struct axial *ar)
{
   int i;

   float delta_r, delta_z, delta_w, alpha_krit, arb_angle, ratio;
   float p1[3], p2[3], p3[3], v1[3], v3[3], x[3];

   struct Point *poly=NULL;
   struct Flist *knot=NULL;

   // **************************************************
   // inits and dels
   // reset bend curves
   if(ar->p_sbend) ar->p_sbend->nump = 0;
   if(ar->p_hbend) ar->p_hbend->nump = 0;

   ar->mod->inl = 0; ar->mod->bend = 1;

   // **************************************************
   // shroud contour
   if((delta_r = ar->d_inl_ext*0.5f - ar->p_score->x[0]) <= 0.0f)
      return INLET_RADIUS_ERR;
   if((delta_z = ar->h_inl_ext*0.5f - ar->p_score->z[0]) <= 0.0f)
      return INLET_HEIGHT_ERR;
   // inlet width delta_w supposed to be constant
   delta_w = ar->p_score->x[0] - ar->p_hcore->x[0];

   alpha_krit = float(atan(delta_r/delta_z));
   arb_angle  = RAD(ar->arb_angle);
   if(arb_angle <= alpha_krit)
   {
      dprintf(0," WARNING: Inlet pitch angle results in straight inlet!\n");
      dprintf(0,"	     Critical angle = %f\n\n",alpha_krit*180.0/M_PI);
   }

   p3[0] = p1[0] = ar->p_score->x[0];
   p3[1] = p1[1] = 0.0;
   p3[2] = p1[2] = ar->p_score->z[0];
   p1[0] += delta_r;
   p1[2] += delta_z;
   // shroud: straight line of spline
   if(arb_angle <= alpha_krit)
   {
      arb_angle = alpha_krit;
      v1[0] = p3[0] - p1[0];
      v1[1] = 0.0;
      v1[2] = p3[2] - p1[2];
      for(i = 0; i < NUM_BEND_POINTS; i++)
      {
         // fl: it is correct to leave out ratio == 1.0!!!
         // fl: --> otherwise double point score[0]!
         ratio = ((float)(i)) / ((float) (NUM_BEND_POINTS));
         x[0] = p1[0] + ratio*v1[0];
         x[1] = 0.0;
         x[2] = p1[2] + ratio*v1[2];
         AddVPoint(ar->p_sbend,x);
      }
   }
   // spline
   else
   {
      v1[1] =  v3[0] =  v3[1] =  0.0;
      v1[2] =  v3[2] = -1.0;
      v1[0] = float(-sin(arb_angle));
      LineIntersectXZ(p1, v1, p3, v3, p2);
      poly = CurvePolygon(p1,p2,p3,ar->arb_part[0], ar->arb_part[1]);
      knot = BSplineKnot(poly, BSPLN_DEGREE);
      for(i = 0; i < NUM_BEND_POINTS; i++)
      {
         ratio = ((float)(i)) / ((float) (NUM_BEND_POINTS));
         BSplinePoint(BSPLN_DEGREE, poly, knot, ratio, x);
         AddVPoint(ar->p_sbend,x);
      }
   }
   if(poly) FreePointStruct(poly);
   if(knot) FreeFlistStruct(knot);

   // **************************************************
   // hub, different alpha_krit
   delta_r = float(ar->p_sbend->x[0] - delta_w*cos(arb_angle) - ar->p_hcore->x[0]);
   delta_z = float(ar->p_sbend->z[0] + delta_w*sin(arb_angle) - ar->p_hcore->z[0]);
   alpha_krit = (float)atan(delta_r/delta_z);

   p3[0] = p1[0] = ar->p_hcore->x[0];
   p3[1] = p1[1] = 0.0;
   p3[2] = p1[2] = ar->p_hcore->z[0];
   p1[0] += delta_r;
   p1[2] += delta_z;
   // hub: straight line of spline
   if(arb_angle <= alpha_krit)
   {
      arb_angle = alpha_krit;
      v1[0] = p3[0] - p1[0];
      v1[1] = 0.0;
      v1[2] = p3[2] - p1[2];
      for(i = 0; i < NUM_BEND_POINTS; i++)
      {
         ratio = ((float)(i)) / ((float) (NUM_BEND_POINTS));
         x[0] = p1[0] + ratio*v1[0];
         x[1] = 0.0;
         x[2] = p1[2] + ratio*v1[2];
         AddVPoint(ar->p_hbend,x);
      }
   }
   // spline
   else
   {
      v1[1] =  v3[0] =  v3[1] =  0.0;
      v1[2] =  v3[2] = -1.0;
      v1[0] = float(-sin(arb_angle));
      LineIntersectXZ(p1, v1, p3, v3, p2);
      poly = CurvePolygon(p1,p2,p3,ar->arb_part[0], ar->arb_part[1]);
      knot = BSplineKnot(poly, BSPLN_DEGREE);
      for(i = 0; i < NUM_BEND_POINTS; i++)
      {
         ratio = ((float)(i)) / ((float) (NUM_BEND_POINTS));
         BSplinePoint(BSPLN_DEGREE, poly, knot, ratio, x);
         AddVPoint(ar->p_hbend,x);
      }
   }
   if(poly) FreePointStruct(poly);
   if(knot) FreeFlistStruct(knot);
   // **************************************************

   return 0;
}
