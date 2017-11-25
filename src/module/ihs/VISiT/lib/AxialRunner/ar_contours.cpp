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
#include "BSpline/include/bspline.h"
#include "AxialRunner/include/axial.h"
#include "AxialRunner/include/ar_contours.h"
#include "include/ar_inletarbitrary.h"

#ifndef MAX
#define MAX(a,b) ( (a) >= (b) ? (a) : (b))
#endif

int CreateAR_Contours(struct axial *ar)
// caller: ReadAxialRunner()
{
   int i, j, err = 0;
   int hbend_nop     = 0;
   int npoin_contour = NPOIN_SPLN_CORE;
   static int p_hub_bend_calculated = 0;
   float d, d_hemi, h_arc, x[3], angle, length;
   float s[NPOIN_MAX][3], h[NPOIN_MAX][3];
   float sec, base[3], vec[3], cent1[3], cent2[3], cent3[3], cent4[3], base2[3], vec2[3];
   float a, b, n, m, bias;
   struct Point *p_shroud_arc  = NULL;
   struct Point *p_counter_arc = NULL;
   struct Point *p_hub_arc     = NULL;
   struct Point *p_sbend_arc1  = NULL;
   struct Point *p_sbend_arc2  = NULL;
   struct Point *p_hbend_arc   = NULL;
   struct Point *poly          = NULL;
   struct Flist *knot          = NULL;
   struct Flist *hbend_bias    = NULL;
   char *fn;
   FILE *fp, *fp2;

   // inits
   for (i = 0; i < NPOIN_MAX; i++)
   {
      s[i][0] = s[i][1] = s[i][2] = 0.0;
      h[i][0] = h[i][1] = h[i][2] = 0.0;
   }
   x[0]  = x[1]  = x[2]  = 0.0;
   cent1[0] = cent1[1] = cent1[2] = 0.0;
   cent2[0] = cent2[1] = cent2[2] = 0.0;
   cent3[0] = cent3[1] = cent3[2] = 0.0;
   cent4[0] = cent4[1] = 0.0;
   cent4[2] = -ar->h_run;
   ar->hub_sphere = ar->shroud_sphere = 0;

   // SHROUD STAGING POINTS
   if (ar->p_shroud)
   {
      FreePointStruct(ar->p_shroud);
   }
   ar->p_shroud  = AllocPointStruct();
   // S1
   s[0][0] = 0.5f * ar->d_inl_ext;
   s[0][1] = 0.0f;
   s[0][2] = -0.5f * ar->h_inl_ext;
   AddVPoint(ar->p_shroud, s[0]);
   // translation for hemisphere contour
   d_hemi = 0.5f * (ar->d_shroud_sphere - ar->diam[1]);
   // calculate composite shroud bend, then translate
   // to shroud diametre:
   // centre of shroud corner start arc
   cent1[0] = s[0][0];
   cent1[1] = 0.0;
   cent1[2] = s[0][2] - ar->r_shroud[0];
   // end point start arc (S3)
   angle   = RAD(-ar->ang_shroud);
   s[2][0]  = float((s[0][0] - cent1[0]) * cos(angle));
   s[2][0] += float((s[0][2] - cent1[2]) * sin(angle) + cent1[0]);
   s[2][1]  = 0.0;
   s[2][2]  = float((s[0][0] - cent1[0]) * -sin(angle));
   s[2][2] += float((s[0][2] - cent1[2]) * cos(angle) + cent1[2]);
   // centre of shroud end arc
   cent2[0] = s[2][0] + (cent1[0] - s[2][0]) * ar->r_shroud[1] / ar->r_shroud[0];
   cent2[1] = 0.0;
   cent2[2] = s[2][2] + (cent1[2] - s[2][2]) * ar->r_shroud[1] / ar->r_shroud[0];
   // end point bend (S4)
   s[3][0] = cent2[0] - ar->r_shroud[1];
   s[3][1] = 0.0;
   s[3][2] = cent2[2];
   dprintf(5,"cent2 = [%f  %f   %f]\n",cent2[0], cent2[1], cent2[2]);
   // distance to shroud diametre, translation (S2)
   d = 0.5f * ar->diam[1] - s[3][0];
   if (ar->shroud_hemi)
   {
      d += d_hemi;
   }
   s[3][0]  += d;
   s[2][0]  += d;
   s[1][0]   = s[0][0] + d;
   s[1][1]   = 0.0;
   s[1][2]   = s[0][2];
   cent1[0] += d;
   cent2[0] += d;
   AddVPoint(ar->p_shroud, s[1]);
   AddVPoint(ar->p_shroud, cent1);
   AddVPoint(ar->p_shroud, s[2]);
   AddVPoint(ar->p_shroud, cent2);
   AddVPoint(ar->p_shroud, s[3]);
   // shroud bend arc points
   if (p_sbend_arc1)
   {
      FreePointStruct(p_sbend_arc1);
   }
   p_sbend_arc1 = ArcSegmentsXZ(cent1, s[2], s[1], NPOIN_LINEAR-1);
   if (p_sbend_arc2)
   {
      FreePointStruct(p_sbend_arc2);
   }
   p_sbend_arc2 = ArcSegmentsXZ(cent2, s[3], s[2], (ar->hub_nos+1)*(NPOIN_LINEAR-1));
   // shroud sphere start/end point
   if (ar->d_shroud_sphere > ar->diam[1])
   {
      ar->shroud_sphere = 1;
      h_arc = float(0.5f * sqrt(pow(ar->d_shroud_sphere, 2) - pow(ar->diam[1], 2)));
      // S5
      s[4][0] = 0.5f * ar->diam[1];
      s[4][1] = 0.0f;
      s[4][2] = - ar->h_run + h_arc;
      if (ar->shroud_hemi)
      {
         s[4][0] += d_hemi;
      }
      AddVPoint(ar->p_shroud, s[4]);
      // S6 (shroud arc middle)
      s[5][0] = 0.5f * ar->d_shroud_sphere;
      s[5][1] = 0.0f;
      s[5][2] = -ar->h_run;
      AddVPoint(ar->p_shroud, s[5]);
      // S7
      s[6][0] = 0.5f * ar->diam[1];
      s[6][1] = 0.0f;
      s[6][2] = - ar->h_run - h_arc;
      AddVPoint(ar->p_shroud, s[6]);
      if (p_shroud_arc)
      {
         FreePointStruct(p_shroud_arc);
      }
      if (ar->shroud_hemi)
      {
         p_shroud_arc = ArcSegmentsXZ(cent4, s[5], s[6], NOS_SPHERE_ARC/2);
      }
      else
      {
         p_shroud_arc = ArcSegmentsXZ(cent4, s[4], s[6], NOS_SPHERE_ARC);
      }
   }
   else
   {
      if (ar->shroud_counter_rad)
      {
         ar->shroud_counter_rad = 0;
         dprintf(0, "\n *** WARNING ***\n");
         dprintf(0, "Runner cannot have counter radius without\n");
         dprintf(0, "shroud sphere!\n");
         dprintf(0, "Flag shroud_counter_rad is set to 0, runner\n");
         dprintf(0, "is modelled without counter radius contour.\n\n");
      }
   }
   // draft tube start point (new S8)
   s[7][0] = 0.5f * ar->d_draft;
   s[7][1] = 0.0f;
   s[7][2] = - ar->h_run - ar->h_draft;
   if (!ar->shroud_sphere)
   {
      s[7][0] = 0.5f * ar->diam[1];
   }
   AddVPoint(ar->p_shroud, s[7]);
   if (ar->shroud_counter_rad)
   {
      // centre of counter radius:
      // intersection of line [M_run-S7] (base, vec) and
      // perpendicular bisector to line [S7-S8] (base2, vec2)
      base[0]  = 0.0f;
      base[1]  = 0.0f;
      base[2]  = - ar->h_run;
      vec[0]   = s[6][0] - base[0];
      vec[1]   = 0.0f;
      vec[2]   = s[6][2] - base[2];
      base2[0] = 0.5f * (s[6][0] + s[7][0]);
      base2[1] = 0.0f;
      base2[2] = 0.5f * (s[6][2] + s[7][2]);
      // normal to line [S6-S7]
      vec2[0]  = s[7][2] - s[6][2];
      vec2[1]  = 0.0;
      vec2[2]  = s[6][0] - s[7][0];
      // intersection is centre of counter radius
      LineIntersectXZ(base, vec, base2, vec2, cent3);
      // counter radius arc points
      if (p_counter_arc)
      {
         FreePointStruct(p_counter_arc);
      }
      if (ar->counter_nos)
      {
         p_counter_arc = ArcSegmentsXZ(cent3, s[7], s[6], ar->counter_nos);
      }
      else
      {
         p_counter_arc = ArcSegmentsXZ(cent3, s[7], s[6], SMOOTH_COUNTER);
      }
   }
   // outlet extension start point (hub cap height, new S9)
   length  = ar->h_run + ar->p_hubcap->z[ar->cap_nop-1] + s[7][2];
   s[8][0] = float(s[7][0] + length * tan(RAD(ar->ang_draft)));
   s[8][1] = 0.0;
   s[8][2] = s[7][2] - length;
   AddVPoint(ar->p_shroud, s[8]);
   // draft tube end point (new S10)
   length  = ar->p_hubcap->z[ar->cap_nop-1];
   s[9][0] = float(s[7][0] + length * tan(RAD(ar->ang_draft)));
   s[9][1] = 0.0;
   s[9][2] = s[7][2] - length;
   AddVPoint(ar->p_shroud, s[9]);

   // SHROUD STAGING CONTOUR SEGMENTS
   // inlet extension shroud
   if (ar->p_sinlet)
   {
      FreePointStruct(ar->p_sinlet);
   }
   ar->p_sinlet = AllocPointStruct();
   base[0] = s[0][0];
   base[1] = s[0][1];
   base[2] = s[0][2];
   vec[0]  = s[1][0] - s[0][0];
   vec[1]  = s[1][1] - s[0][1];
   vec[2]  = s[1][2] - s[0][2];
   for (i = 0; i < NPOIN_LINEAR; i++)
   {
      sec  = (float)i / (NPOIN_LINEAR - 1);
      x[0] = base[0] + sec * vec[0];
      x[1] = base[1] + sec * vec[1];
      x[2] = base[2] + sec * vec[2];
      AddVPoint(ar->p_sinlet, x);
   }
   // bend region shroud
   if (ar->p_sbend)
   {
      FreePointStruct(ar->p_sbend);
   }
   ar->p_sbend = AllocPointStruct();
   for (i = p_sbend_arc1->nump-1; i > 0; i--)
   {
      x[0] = p_sbend_arc1->x[i];
      x[1] = p_sbend_arc1->y[i];
      x[2] = p_sbend_arc1->z[i];
      AddVPoint(ar->p_sbend, x);
   }
   for (i = p_sbend_arc2->nump-1; i >= 0; i--)
   {
      x[0] = p_sbend_arc2->x[i];
      x[1] = p_sbend_arc2->y[i];
      x[2] = p_sbend_arc2->z[i];
      AddVPoint(ar->p_sbend, x);
   }
   // core region shroud
   if (ar->p_score)
   {
      FreePointStruct(ar->p_score);
   }
   ar->p_score = AllocPointStruct();
   if (ar->shroud_sphere)
   {
      // pre-arc part
      base[0] = s[3][0];
      base[1] = s[3][1];
      base[2] = s[3][2];
      if (ar->shroud_hemi)
      {
         vec[0]  = s[5][0] - s[3][0];
         vec[1]  = s[5][1] - s[3][1];
         vec[2]  = s[5][2] - s[3][2];
      }
      else
      {
         vec[0]  = s[4][0] - s[3][0];
         vec[1]  = s[4][1] - s[3][1];
         vec[2]  = s[4][2] - s[3][2];
      }
      for (i = 0; i < NPOIN_LINEAR; i++)
      {
         sec  = (float)i / (NPOIN_LINEAR - 1);
         x[0] = base[0] + sec * vec[0];
         x[1] = base[1] + sec * vec[1];
         x[2] = base[2] + sec * vec[2];
         AddVPoint(ar->p_score, x);
      }
      // arc part (shroud sphere)
      for (i = 1; i < p_shroud_arc->nump; i++)
      {
         x[0] = p_shroud_arc->x[i];
         x[1] = p_shroud_arc->y[i];
         x[2] = p_shroud_arc->z[i];
         AddVPoint(ar->p_score, x);
      }
      // post-arc part, counter arc
      if (ar->shroud_counter_rad)
      {
         for (i = p_counter_arc->nump-1; i > 0; i--)
         {
            x[0] = p_counter_arc->x[i];
            x[1] = p_counter_arc->y[i];
            x[2] = p_counter_arc->z[i];
            AddVPoint(ar->p_score, x);
         }
      }
      else                                        // post-arc part, no counter arc
      {
         base[0] = s[6][0];
         base[1] = s[6][1];
         base[2] = s[6][2];
         vec[0]  = s[7][0] - s[6][0];
         vec[1]  = s[7][1] - s[6][1];
         vec[2]  = s[7][2] - s[6][2];
         for (i = 1; i < NPOIN_LINEAR; i++)
         {
            sec  = (float)i / (NPOIN_LINEAR - 1);
            x[0] = base[0] + sec * vec[0];
            x[1] = base[1] + sec * vec[1];
            x[2] = base[2] + sec * vec[2];
            AddVPoint(ar->p_score, x);
         }
      }
   }
   else
   {
      base[0] = s[3][0];
      base[1] = s[3][1];
      base[2] = s[3][2];
      vec[0]  = s[7][0] - s[3][0];
      vec[1]  = s[7][1] - s[3][1];
      vec[2]  = s[7][2] - s[3][2];
      for (i = 0; i < NPOIN_LINEAR; i++)
      {
         sec  = (float)i / (NPOIN_LINEAR - 1);
         x[0] = base[0] + sec * vec[0];
         x[1] = base[1] + sec * vec[1];
         x[2] = base[2] + sec * vec[2];
         AddVPoint(ar->p_score, x);
      }
   }
   // draft tube inlet
   base[0] = s[7][0];
   base[1] = s[7][1];
   base[2] = s[7][2];
   vec[0]  = s[8][0] - s[7][0];
   vec[1]  = s[8][1] - s[7][1];
   vec[2]  = s[8][2] - s[7][2];
   for (i = 1; i < NPOIN_LINEAR; i++)
   {
      sec  = (float)i / (NPOIN_LINEAR - 1);
      x[0] = base[0] + sec * vec[0];
      x[1] = base[1] + sec * vec[1];
      x[2] = base[2] + sec * vec[2];
      AddVPoint(ar->p_score, x);
   }
   // outlet extension shroud
   if (ar->p_soutlet)
   {
      FreePointStruct(ar->p_soutlet);
   }
   ar->p_soutlet = AllocPointStruct();

   base[0] = s[8][0];
   base[1] = s[8][1];
   base[2] = s[8][2];
   vec[0]  = s[9][0] - s[8][0];
   vec[1]  = s[9][1] - s[8][1];
   vec[2]  = s[9][2] - s[8][2];
   for (i = 0; i < NPOIN_OUTLET; i++)
   {
      sec  = (float)i / (NPOIN_OUTLET - 1);
      x[0] = base[0] + sec * vec[0];
      x[1] = base[1] + sec * vec[1];
      x[2] = base[2] + sec * vec[2];
      AddVPoint(ar->p_soutlet, x);
   }

   // HUB STAGING POINTS
   if (ar->p_hub)
   {
      FreePointStruct(ar->p_hub);
   }
   ar->p_hub = AllocPointStruct();
   // H1
   h[0][0] = 0.5f * ar->d_inl_ext;
   h[0][1] = 0.0f;
   h[0][2] = 0.5f * ar->h_inl_ext;
   AddVPoint(ar->p_hub, h[0]);
   // H2
   h[1][0] = s[1][0];
   h[1][1] = 0.0f;
   h[1][2] = 0.5f * ar->h_inl_ext;
   AddVPoint(ar->p_hub, h[1]);
   // H3
   h[2][0] = 0.5f * ar->diam[0] + ar->b_hub;
   h[2][1] = 0.0f;
   h[2][2] = 0.5f * ar->h_inl_ext;
   AddVPoint(ar->p_hub, h[2]);
   // H4
   h[3][0] = 0.5f * ar->diam[0];
   h[3][1] = 0.0f;
   h[3][2] = 0.5f * ar->h_inl_ext - ar->a_hub;
   AddVPoint(ar->p_hub, h[3]);
   // H5
   h[4][0] = 0.5f * ar->diam[0];
   h[4][1] = 0.0f;
   h[4][2] = s[3][2];
   AddVPoint(ar->p_hub, h[4]);
   // hub sphere start/end point
   if (ar->d_hub_sphere > ar->diam[0])
   {
      ar->hub_sphere = 1;
      h_arc = float(0.5f * sqrt(pow(ar->d_hub_sphere, 2) - pow(ar->diam[0], 2)));
      // H6
      h[5][0] = 0.5f * ar->diam[0];
      h[5][1] = 0.0f;
      h[5][2] = - ar->h_run + h_arc;
      AddVPoint(ar->p_hub, h[5]);
      // H7
      h[6][0] = 0.5f * ar->diam[0];
      h[6][1] = 0.0f;
      h[6][2] = - ar->h_run - h_arc;
      AddVPoint(ar->p_hub, h[6]);
      if (p_hub_arc)
      {
         FreePointStruct(p_hub_arc);
      }
      p_hub_arc = ArcSegmentsXZ(cent4, h[5], h[6], NOS_SPHERE_ARC);
   }
   // H8 ... H[7+cap_nop]
   for (i = 0; i < ar->cap_nop; i++)
   {
      h[7+i][0] = 0.5f * ar->p_hubcap->x[i];
      h[7+i][1] = 0.0;
      h[7+i][2] = - ar->h_run - ar->p_hubcap->z[i];
      if (!ar->hub_sphere && !i)
      {
         h[7+i][0] = 0.5f * ar->diam[0];
      }
      AddVPoint(ar->p_hub, h[7+i]);
      dprintf(6, "h[7+i(=%d)][0] = %f", i, h[7+i][0]);
      dprintf(6, "  h[7+i(=%d)][1] = %f", i, h[7+i][1]);
      dprintf(6, "  h[7+i(=%d)][2] = %f\n", i, h[7+i][2]);
   }
   // base for outlet extension spline (H[7+cap_nop])
   base[0] = 0.5f * ar->p_hubcap->x[ar->cap_nop-1];
   base[1] = 0.0;
   base[2] = - ar->h_run - ar->p_hubcap->z[ar->cap_nop-1];
   // hub cap middle point (H[8+cap_nop])
   h[7+ar->cap_nop][0] = 0.0;
   h[7+ar->cap_nop][1] = 0.0;
   h[7+ar->cap_nop][2] = - ar->h_run - ar->p_hubcap->z[ar->cap_nop-1];
   AddVPoint(ar->p_hub, h[7+ar->cap_nop]);
   // outlet extension end point (H[9+cap_nop])
   vec[0] = ar->p_hub->x[ar->p_hub->nump-2] - ar->p_hub->x[ar->p_hub->nump-3];
   vec[1] = ar->p_hub->y[ar->p_hub->nump-2] - ar->p_hub->y[ar->p_hub->nump-3];
   vec[2] = ar->p_hub->z[ar->p_hub->nump-2] - ar->p_hub->z[ar->p_hub->nump-3];

   h[8+ar->cap_nop][1]  = 0.0;
   h[8+ar->cap_nop][2]  = s[9][2];
   sec    = (h[8+ar->cap_nop][2] - base[2]) / vec[2];
   h[8+ar->cap_nop][0]  = base[0] + sec * vec[0];
   h[8+ar->cap_nop][0] *= 2.0;

   if(vec[2] < (10.0*vec[0]))
      h[8+ar->cap_nop][0] = 0.5f*ar->p_hubcap->x[ar->cap_nop-1];
   else
      h[8+ar->cap_nop][0] = POST_HUB_CORE*0.5f*ar->p_hubcap->x[ar->cap_nop-1];
   AddVPoint(ar->p_hub, h[8+ar->cap_nop]);
   // arc part (ellipse)
   if (p_hbend_arc)
   {
      FreePointStruct(p_hbend_arc);
   }
   p_hbend_arc = AllocPointStruct();
   // hub bend points directly modified (COVISE)
   if (ar->hub_bmodpoints && p_hub_bend_calculated)
   {
      // copy global hub bend points to local array
      // (input from COVISE module)
      for (i = 0; i < ar->p_hbpoints->nump; i++)
      {
         x[0] = ar->p_hbpoints->x[i];
         x[1] = ar->p_hbpoints->y[i];
         x[2] = ar->p_hbpoints->z[i];
         AddVPoint(p_hbend_arc, x);
      }
   }
   else                                           // calculate new points on ellipse
   {
      if (ar->p_hbpoints)
      {
         FreePointStruct(ar->p_hbpoints);
      }
      ar->p_hbpoints = AllocPointStruct();
      a = ar->a_hub;
      b = ar->b_hub;
      m = h[2][0];
      n = h[3][2];
      AddVPoint(p_hbend_arc, h[2]);
      bias = ar->a_hub / ar->b_hub * ELLIPSE_BIAS;
      if (ar->hub_nos)
      {
         hbend_nop = ar->hub_nos + 1;
      }
      else
      {
         hbend_nop = SMOOTH_HBEND + 1;
      }
      hbend_bias = CalcBladeElementBias(hbend_nop, 0.0, 1.0, 1, bias);
      for (i = 1; i < hbend_nop-1; i++)
      {
         sec  = hbend_bias->list[i];
         x[0] = m - sec * b;
         x[1] = 0.0;
         x[2] = float(n + a * sqrt(1.0 - pow((x[0]-m), 2)/pow(b, 2)));
         AddVPoint(p_hbend_arc, x);
      }
      if (!ar->hub_nos)
      {
         x[0] = 0.5f * (h[3][0] + p_hbend_arc->x[p_hbend_arc->nump-1]);
         x[1] = 0.5f * (h[3][1] + p_hbend_arc->y[p_hbend_arc->nump-1]);
         x[2] = 0.5f * (h[3][2] + p_hbend_arc->z[p_hbend_arc->nump-1]);
         AddVPoint(p_hbend_arc, x);
      }
      AddVPoint(p_hbend_arc, h[3]);
      // copy to global hub bend point array
      for (i = 0; i < p_hbend_arc->nump; i++)
      {
         x[0] = p_hbend_arc->x[i];
         x[1] = p_hbend_arc->y[i];
         x[2] = p_hbend_arc->z[i];
         AddVPoint(ar->p_hbpoints, x);
      }
      p_hub_bend_calculated = 1;
   }

   // HUB STAGING CONTOUR SEGMENTS
   // inlet extension hub
   if (ar->p_hinlet)
   {
      FreePointStruct(ar->p_hinlet);
   }
   ar->p_hinlet = AllocPointStruct();
   base[0] = h[0][0];
   base[1] = h[0][1];
   base[2] = h[0][2];
   vec[0]  = h[1][0] - h[0][0];
   vec[1]  = h[1][1] - h[0][1];
   vec[2]  = h[1][2] - h[0][2];
   for (i = 0; i < NPOIN_LINEAR; i++)
   {
      sec  = (float)i / (NPOIN_LINEAR - 1);
      x[0] = base[0] + sec * vec[0];
      x[1] = base[1] + sec * vec[1];
      x[2] = base[2] + sec * vec[2];
      AddVPoint(ar->p_hinlet, x);
   }
   // bend region hub
   if (ar->p_hbend)
   {
      FreePointStruct(ar->p_hbend);
   }
   ar->p_hbend = AllocPointStruct();
   // pre-arc part
   base[0] = h[1][0];
   base[1] = h[1][1];
   base[2] = h[1][2];
   vec[0]  = h[2][0] - h[1][0];
   vec[1]  = h[2][1] - h[1][1];
   vec[2]  = h[2][2] - h[1][2];
   for (i = 0; i < NPOIN_LINEAR; i++)
   {
      sec  = (float)i / (NPOIN_LINEAR - 1);
      x[0] = base[0] + sec * vec[0];
      x[1] = base[1] + sec * vec[1];
      x[2] = base[2] + sec * vec[2];
      AddVPoint(ar->p_hbend, x);
   }
   // ellipse part
   if (ar->hub_nos)
   {
      for (i = 0; i < ar->hub_nos; i++)
      {
         base[0] = p_hbend_arc->x[i];
         base[1] = p_hbend_arc->y[i];
         base[2] = p_hbend_arc->z[i];
         vec[0]  = p_hbend_arc->x[i+1] - p_hbend_arc->x[i];
         vec[1]  = p_hbend_arc->y[i+1] - p_hbend_arc->y[i];
         vec[2]  = p_hbend_arc->z[i+1] - p_hbend_arc->z[i];
         for (j = 1; j < NPOIN_LINEAR; j++)
         {
            sec  = (float)j / (NPOIN_LINEAR - 1);
            x[0] = base[0] + sec * vec[0];
            x[1] = base[1] + sec * vec[1];
            x[2] = base[2] + sec * vec[2];
            AddVPoint(ar->p_hbend, x);
         }
      }
   }
   else                                           // smooth hub bend contour
   {
      for (i = 0; i < p_hbend_arc->nump; i++)
      {
         x[0] = p_hbend_arc->x[i];
         x[1] = p_hbend_arc->y[i];
         x[2] = p_hbend_arc->z[i];
         AddVPoint(ar->p_hbend, x);
      }
   }
   // post-arc part
   base[0] = h[3][0];
   base[1] = h[3][1];
   base[2] = h[3][2];
   vec[0]  = h[4][0] - h[3][0];
   vec[1]  = h[4][1] - h[3][1];
   vec[2]  = h[4][2] - h[3][2];
   for (i = 1; i < NPOIN_LINEAR; i++)
   {
      sec  = (float)i / (NPOIN_LINEAR - 1);
      x[0] = base[0] + sec * vec[0];
      x[1] = base[1] + sec * vec[1];
      x[2] = base[2] + sec * vec[2];
      AddVPoint(ar->p_hbend, x);
   }
   // core region hub
   if (ar->p_hcore)
   {
      FreePointStruct(ar->p_hcore);
   }
   ar->p_hcore = AllocPointStruct();
   if (ar->hub_sphere)
   {
      // pre-arc part
      base[0] = h[4][0];
      base[1] = h[4][1];
      base[2] = h[4][2];
      vec[0]  = h[5][0] - h[4][0];
      vec[1]  = h[5][1] - h[4][1];
      vec[2]  = h[5][2] - h[4][2];
      for (i = 0; i < NPOIN_LINEAR; i++)
      {
         sec  = (float)i / (NPOIN_LINEAR - 1);
         x[0] = base[0] + sec * vec[0];
         x[1] = base[1] + sec * vec[1];
         x[2] = base[2] + sec * vec[2];
         AddVPoint(ar->p_hcore, x);
         dprintf(5,"x = [%f  %f  %f]\n",x[0],x[1],x[2]);
      }
      // arc part (hub sphere)
      for (i = 1; i < p_hub_arc->nump; i++)
      {
         x[0] = p_hub_arc->x[i];
         x[1] = p_hub_arc->y[i];
         x[2] = p_hub_arc->z[i];
         AddVPoint(ar->p_hcore, x);
      }
      // segment to cap start
      base[0] = h[6][0];
      base[1] = h[6][1];
      base[2] = h[6][2];
      vec[0]  = h[7][0] - h[6][0];
      vec[1]  = h[7][1] - h[6][1];
      vec[2]  = h[7][2] - h[6][2];
      for (i = 1; i < NPOIN_LINEAR; i++)
      {
         sec  = (float)i / (NPOIN_LINEAR - 1);
         x[0] = base[0] + sec * vec[0];
         x[1] = base[1] + sec * vec[1];
         x[2] = base[2] + sec * vec[2];
         AddVPoint(ar->p_hcore, x);
      }
   }
   else                                           // no hub sphere
   {
      base[0] = h[4][0];
      base[1] = h[4][1];
      base[2] = h[4][2];
      vec[0]  = h[7][0] - h[4][0];
      vec[1]  = h[7][1] - h[4][1];
      vec[2]  = h[7][2] - h[4][2];
      for (i = 0; i < NPOIN_LINEAR; i++)
      {
         sec  = (float)i / (NPOIN_LINEAR - 1);
         x[0] = base[0] + sec * vec[0];
         x[1] = base[1] + sec * vec[1];
         x[2] = base[2] + sec * vec[2];
         AddVPoint(ar->p_hcore, x);
      }
   }
   // post-arc, cap
   for (i = 0; i < ar->cap_nop-1; i++)
   {
      base[0] = h[7+i][0];
      base[1] = h[7+i][1];
      base[2] = h[7+i][2];
      vec[0]  = h[8+i][0] - h[7+i][0];
      vec[1]  = h[8+i][1] - h[7+i][1];
      vec[2]  = h[8+i][2] - h[7+i][2];
      for (j = 1; j < NPOIN_LINEAR; j++)
      {
         sec  = (float)j / (NPOIN_LINEAR - 1);
         x[0] = base[0] + sec * vec[0];
         x[1] = base[1] + sec * vec[1];
         x[2] = base[2] + sec * vec[2];
         AddVPoint(ar->p_hcore, x);
      }
   }
   // outlet extension hub
   if (ar->p_houtlet)
   {
      FreePointStruct(ar->p_houtlet);
   }
   ar->p_houtlet = AllocPointStruct();
   base[0]  = h[6+ar->cap_nop][0];
   base[1]  = h[6+ar->cap_nop][1];
   base[2]  = h[6+ar->cap_nop][2];
   vec[0]   = h[6+ar->cap_nop][0] - h[5+ar->cap_nop][0];
   vec[1]   = h[6+ar->cap_nop][1] - h[5+ar->cap_nop][1];
   vec[2]   = h[6+ar->cap_nop][2] - h[5+ar->cap_nop][2];
   base2[0] = h[8+ar->cap_nop][0];
   base2[1] = h[8+ar->cap_nop][1];
   base2[2] = h[8+ar->cap_nop][2];
   vec2[0]  = 0.0;
   vec2[1]  = 0.0;
   vec2[2]  = 1.0;
   LineIntersectXZ(base, vec, base2, vec2, x);
   poly = CurvePolygon(base, x, base2, 0.4f, 0.4f);
   knot = BSplineKnot(poly, BSPLN_DEGREE);
   for (i = 0; i < NPOIN_OUTLET; i++)
   {
      sec = (float)i / (NPOIN_OUTLET - 1);
      BSplinePoint(BSPLN_DEGREE, poly, knot, sec, &x[0]);
      AddVPoint(ar->p_houtlet, x);
   }

   // if arbitrary inlet (spline) modify sbend & hbend, delete h/sinlet
   // this could be more elegant, but it was the easiest way to do it
   // like this. All the other stuff already exists.
   if(ar->mod->arbitrary)
      if((err = CreateAR_InletArbitrary(ar))) return err;

   // ENTIRE MODELLED SHROUD AND HUB CONTOUR CURVES
   if (ar->me[ar->be_num-1]->ml)
   {
      FreeCurveStruct(ar->me[ar->be_num-1]->ml);
   }
   ar->me[ar->be_num-1]->ml = AllocCurveStruct();
   if (ar->me[0]->ml)
   {
      FreeCurveStruct(ar->me[0]->ml);
   }
   ar->me[0]->ml = AllocCurveStruct();

   if (ar->mod->inl)                              // inlet region
   {
      // shroud
      for (i = 0; i < ar->p_sinlet->nump-1; i++)
      {
         x[0] = ar->p_sinlet->x[i];
         x[1] = ar->p_sinlet->y[i];
         x[2] = ar->p_sinlet->z[i];
         AddCurvePoint(ar->me[ar->be_num-1]->ml, x[0], x[1], x[2], 0.0, 0.0);
      }
      // hub
      npoin_contour += ar->p_hinlet->nump - 1;
      for (i = 0; i < ar->p_hinlet->nump-1; i++)
      {
         x[0] = ar->p_hinlet->x[i];
         x[1] = ar->p_hinlet->y[i];
         x[2] = ar->p_hinlet->z[i];
         AddCurvePoint(ar->me[0]->ml, x[0], x[1], x[2], 0.0, 0.0);
      }
   }

   if (ar->mod->bend)                             // bend region
   {
      // shroud
      for (i = 0; i < ar->p_sbend->nump-1; i++)
      {
         x[0] = ar->p_sbend->x[i];
         x[1] = ar->p_sbend->y[i];
         x[2] = ar->p_sbend->z[i];
         AddCurvePoint(ar->me[ar->be_num-1]->ml, x[0], x[1], x[2], 0.0, 0.0);
      }
      // hub
      npoin_contour += ar->p_hbend->nump - 1;
      for (i = 0; i < ar->p_hbend->nump-1; i++)
      {
         x[0] = ar->p_hbend->x[i];
         x[1] = ar->p_hbend->y[i];
         x[2] = ar->p_hbend->z[i];
         AddCurvePoint(ar->me[0]->ml, x[0], x[1], x[2], 0.0, 0.0);
      }
   }

   // core region shroud (mandatory)
   if (poly)
   {
      FreePointStruct(poly);
      FreeFlistStruct(knot);
   }
   poly = AllocPointStruct();
   // fl: necessary? realloced later, in BSplineKnot
   //knot = AllocFlistStruct(INIT_PORTION);
   for (i = 0; i < ar->p_score->nump; i++)
   {
      x[0] = ar->p_score->x[i];
      x[1] = ar->p_score->y[i];
      x[2] = ar->p_score->z[i];
      AddVPoint(poly, x);
   }
   knot = BSplineKnot(poly, BSPLN_DEGREE);
   for (i = 0; i < NPOIN_SPLN_CORE; i++)
   {
      sec = (float)i / (NPOIN_SPLN_CORE - 1);
      BSplinePoint(BSPLN_DEGREE, poly, knot, sec, &x[0]);
      AddCurvePoint(ar->me[ar->be_num-1]->ml, x[0], x[1], x[2], 0.0, sec);
   }
   // core region hub (mandatory)
   if (poly)
   {
      FreePointStruct(poly);
      FreeFlistStruct(knot);
   }
   poly = AllocPointStruct();
   // fl: necessary? realloced later, in BSplineKnot
   //knot = AllocFlistStruct(INIT_PORTION);
   for (i = 0; i < ar->p_hcore->nump; i++)
   {
      x[0] = ar->p_hcore->x[i];
      x[1] = ar->p_hcore->y[i];
      x[2] = ar->p_hcore->z[i];
      AddVPoint(poly, x);
   }
   knot = BSplineKnot(poly, BSPLN_DEGREE);
   for (i = 0; i < NPOIN_SPLN_CORE; i++)
   {
      sec = (float)i / (NPOIN_SPLN_CORE - 1);
      BSplinePoint(BSPLN_DEGREE, poly, knot, sec, &x[0]);
      AddCurvePoint(ar->me[0]->ml, x[0], x[1], x[2], 0.0, sec);
   }

   if (ar->mod->outl)                             // outlet region
   {
      // shroud
      for (i = 1; i < ar->p_soutlet->nump; i++)
      {
         x[0] = ar->p_soutlet->x[i];
         x[1] = ar->p_soutlet->y[i];
         x[2] = ar->p_soutlet->z[i];
         AddCurvePoint(ar->me[ar->be_num-1]->ml, x[0], x[1], x[2], 0.0, 0.0);
      }
      // hub
      for (i = 1; i < ar->p_houtlet->nump; i++)
      {
         x[0] = ar->p_houtlet->x[i];
         x[1] = ar->p_houtlet->y[i];
         x[2] = ar->p_houtlet->z[i];
         AddCurvePoint(ar->me[0]->ml, x[0], x[1], x[2], 0.0, 0.0);
      }
   }

   // output 'ar_contourpoints.txt' for gnuplot check:
   // shroud and hub staging points and arcs

   fn = DebugFilename((char *)"ar_contourpoints.txt");
   if (fn && *fn && (fp = fopen(fn, "w")) != NULL)
   {
      fprintf(fp, "# p_shroud\n");
      for (i = 0; i < ar->p_shroud->nump; i++)
      {
         fprintf(fp, "%f  ", ar->p_shroud->x[i]);
         fprintf(fp, "%f  ", ar->p_shroud->y[i]);
         fprintf(fp, "%f\n", ar->p_shroud->z[i]);
      }
      fprintf(fp, "\n\n");
      fprintf(fp, "# p_sbend_arc1\n");
      for (i = 0; i < p_sbend_arc1->nump; i++)
      {
         fprintf(fp, "%f  ", p_sbend_arc1->x[i]);
         fprintf(fp, "%f  ", p_sbend_arc1->y[i]);
         fprintf(fp, "%f\n", p_sbend_arc1->z[i]);
      }
      fprintf(fp, "# p_sbend_arc2\n");
      for (i = 0; i < p_sbend_arc2->nump; i++)
      {
         fprintf(fp, "%f  ", p_sbend_arc2->x[i]);
         fprintf(fp, "%f  ", p_sbend_arc2->y[i]);
         fprintf(fp, "%f\n", p_sbend_arc2->z[i]);
      }
      fprintf(fp, "\n\n");
      fprintf(fp, "# p_hub\n");
      for (i = 0; i < ar->p_hub->nump; i++)
      {
         fprintf(fp, "%f  ", ar->p_hub->x[i]);
         fprintf(fp, "%f  ", ar->p_hub->y[i]);
         fprintf(fp, "%f\n", ar->p_hub->z[i]);
      }
      fprintf(fp, "\n\n");
      fprintf(fp, "# p_hbend_arc\n");
      for (i = 0; i < p_hbend_arc->nump; i++)
      {
         fprintf(fp, "%f  ", p_hbend_arc->x[i]);
         fprintf(fp, "%f  ", p_hbend_arc->y[i]);
         fprintf(fp, "%f\n", p_hbend_arc->z[i]);
      }
      fprintf(fp, "\n\n");
      if (ar->shroud_sphere)
      {
         fprintf(fp, "# p_shroud_arc\n");
         for (i = 0; i < p_shroud_arc->nump; i++)
         {
            fprintf(fp, "%f  ", p_shroud_arc->x[i]);
            fprintf(fp, "%f  ", p_shroud_arc->y[i]);
            fprintf(fp, "%f\n", p_shroud_arc->z[i]);
         }
         fprintf(fp, "\n\n");
         if (ar->shroud_counter_rad)
         {
            fprintf(fp, "# p_counter_arc\n");
            for (i = 0; i < p_counter_arc->nump; i++)
            {
               fprintf(fp, "%f  ", p_counter_arc->x[i]);
               fprintf(fp, "%f  ", p_counter_arc->y[i]);
               fprintf(fp, "%f\n", p_counter_arc->z[i]);
            }
            fprintf(fp, "\n\n");
         }
      }
      if (ar->hub_sphere)
      {
         fprintf(fp, "# p_hub_arc\n");
         for (i = 0; i < p_hub_arc->nump; i++)
         {
            fprintf(fp, "%f  ", p_hub_arc->x[i]);
            fprintf(fp, "%f  ", p_hub_arc->y[i]);
            fprintf(fp, "%f\n", p_hub_arc->z[i]);
         }
      }
      fclose(fp);
   }
   // output 'ar_contourpoints2.txt' for gnuplot check:
   // shroud and hub staging contour segments
   fn = DebugFilename("ar_contourpoints2.txt");
   if (fn && *fn && (fp2 = fopen(fn, "w")) != NULL)
   {
      fprintf(fp2, "# shroud inlet\n");
      for (i = 0; i < ar->p_sinlet->nump; i++)
      {
         fprintf(fp2, "%f  ", ar->p_sinlet->x[i]);
         fprintf(fp2, "%f  ", ar->p_sinlet->y[i]);
         fprintf(fp2," %f\n", ar->p_sinlet->z[i]);
      }
      fprintf(fp2, "\n\n");
      fprintf(fp2, "# shroud bend\n");
      for (i = 0; i < ar->p_sbend->nump; i++)
      {
         fprintf(fp2, "%f  ", ar->p_sbend->x[i]);
         fprintf(fp2, "%f  ", ar->p_sbend->y[i]);
         fprintf(fp2," %f\n", ar->p_sbend->z[i]);
      }
      fprintf(fp2, "\n\n");
      fprintf(fp2, "# shroud core curve\n");
      for (i = 0; i < ar->p_score->nump; i++)
      {
         fprintf(fp2, "%f  ", ar->p_score->x[i]);
         fprintf(fp2, "%f  ", ar->p_score->y[i]);
         fprintf(fp2," %f\n", ar->p_score->z[i]);
      }
      fprintf(fp2, "\n\n");
      fprintf(fp2, "# shroud outlet curve\n");
      for (i = 0; i < ar->p_soutlet->nump; i++)
      {
         fprintf(fp2, "%f  ", ar->p_soutlet->x[i]);
         fprintf(fp2, "%f  ", ar->p_soutlet->y[i]);
         fprintf(fp2," %f\n", ar->p_soutlet->z[i]);
      }
      fprintf(fp2, "\n\n");
      // hub staging contour segments
      fprintf(fp2, "# hub inlet curve\n");
      for (i = 0; i < ar->p_hinlet->nump; i++)
      {
         fprintf(fp2, "%f  ", ar->p_hinlet->x[i]);
         fprintf(fp2, "%f  ", ar->p_hinlet->y[i]);
         fprintf(fp2," %f\n", ar->p_hinlet->z[i]);
      }
      fprintf(fp2, "\n\n");
      fprintf(fp2, "# hub bend\n");
      for (i = 0; i < ar->p_hbend->nump; i++)
      {
         fprintf(fp2, "%f  ", ar->p_hbend->x[i]);
         fprintf(fp2, "%f  ", ar->p_hbend->y[i]);
         fprintf(fp2," %f\n", ar->p_hbend->z[i]);
      }
      fprintf(fp2, "\n\n");
      fprintf(fp2, "# hub core curve\n");
      for (i = 0; i < ar->p_hcore->nump; i++)
      {
         fprintf(fp2, "%f  ", ar->p_hcore->x[i]);
         fprintf(fp2, "%f  ", ar->p_hcore->y[i]);
         fprintf(fp2," %f\n", ar->p_hcore->z[i]);
      }
      fprintf(fp2, "\n\n");
      fprintf(fp2, "# hub outlet curve\n");
      for (i = 0; i < ar->p_houtlet->nump; i++)
      {
         fprintf(fp2, "%f  ", ar->p_houtlet->x[i]);
         fprintf(fp2, "%f  ", ar->p_houtlet->y[i]);
         fprintf(fp2," %f\n", ar->p_houtlet->z[i]);
      }
      fclose(fp2);
   }

   // scale shroud contour to machine size
   for (i = 0; i < ar->p_shroud->nump; i++)
   {
      ar->p_shroud->x[i] *= ar->ref;
      ar->p_shroud->y[i] *= ar->ref;
      ar->p_shroud->z[i] *= ar->ref;
   }
   for (i = 0; i < ar->me[ar->be_num-1]->ml->p->nump; i++)
   {
      ar->me[ar->be_num-1]->ml->p->x[i] *= ar->ref;
      ar->me[ar->be_num-1]->ml->p->y[i] *= ar->ref;
      // Nullpunkt immer an der Schaufelachse
      ar->me[ar->be_num-1]->ml->p->z[i] += ar->h_run;
      ar->me[ar->be_num-1]->ml->p->z[i] *= ar->ref;
   }
   for (i = 0; i < ar->p_sinlet->nump; i++)
   {
      ar->p_sinlet->x[i] *= ar->ref;
      ar->p_sinlet->y[i] *= ar->ref;
      ar->p_sinlet->z[i] *= ar->ref;
   }
   for (i = 0; i < ar->p_sbend->nump; i++)
   {
      ar->p_sbend->x[i] *= ar->ref;
      ar->p_sbend->y[i] *= ar->ref;
      ar->p_sbend->z[i] *= ar->ref;
   }
   for (i = 0; i < ar->p_score->nump; i++)
   {
      ar->p_score->x[i] *= ar->ref;
      ar->p_score->y[i] *= ar->ref;
      ar->p_score->z[i] *= ar->ref;
   }
   for (i = 0; i < ar->p_soutlet->nump; i++)
   {
      ar->p_soutlet->x[i] *= ar->ref;
      ar->p_soutlet->y[i] *= ar->ref;
      ar->p_soutlet->z[i] *= ar->ref;
   }

   // scale hub contour to machine size
   for (i = 0; i < ar->p_hub->nump; i++)
   {
      ar->p_hub->x[i] *= ar->ref;
      ar->p_hub->y[i] *= ar->ref;
      ar->p_hub->z[i] *= ar->ref;
   }
   for (i = 0; i < ar->me[0]->ml->p->nump; i++)
   {
      ar->me[0]->ml->p->x[i] *= ar->ref;
      ar->me[0]->ml->p->y[i] *= ar->ref;
      // Nullpunkt immer an der Schaufelachse
      ar->me[0]->ml->p->z[i] += ar->h_run;
      ar->me[0]->ml->p->z[i] *= ar->ref;
   }
   for (i = 0; i < ar->p_hinlet->nump; i++)
   {
      ar->p_hinlet->x[i] *= ar->ref;
      ar->p_hinlet->y[i] *= ar->ref;
      ar->p_hinlet->z[i] *= ar->ref;
   }
   for (i = 0; i < ar->p_hbend->nump; i++)
   {
      ar->p_hbend->x[i] *= ar->ref;
      ar->p_hbend->y[i] *= ar->ref;
      ar->p_hbend->z[i] *= ar->ref;
   }
   for (i = 0; i < ar->p_hcore->nump; i++)
   {
      ar->p_hcore->x[i] *= ar->ref;
      ar->p_hcore->y[i] *= ar->ref;
      ar->p_hcore->z[i] *= ar->ref;
   }
   for (i = 0; i < ar->p_houtlet->nump; i++)
   {
      ar->p_houtlet->x[i] *= ar->ref;
      ar->p_houtlet->y[i] *= ar->ref;
      ar->p_houtlet->z[i] *= ar->ref;
   }

#ifdef OLD_STYLE
#ifdef CONTOUR_WIREFRAME
   // create shroud contour wireframe
   CreateContourWireframe(ar->me[ar->be_num-1]->ml, ar->me[ar->be_num-1]->ml->p->nump);
   // create hub contour wireframe
   if (ar->mod->outl)
   {
      CreateContourWireframe(ar->me[0]->ml, ar->me[0]->ml->p->nump-NPOIN_SPLN_OUTLET+1);
   }
   else
   {
      CreateContourWireframe(ar->me[0]->ml, ar->me[0]->ml->p->nump);
   }
#endif                                         // CONTOUR_WIREFRAME
#endif                                         // OLD_STYLE

   return err;
}


#ifdef CONTOUR_WIREFRAME
void CreateContourWireframe(struct curve *c, int nump)
// caller: CreateAR_Contours()
{
   int i, j;
   static int ncall = 0;
   const int nsec = 36;
   float x, y, z;
   float angle, roma[2][2];
   const float rot = 2 * M_PI/nsec;
   FILE *fp_3d=NULL, *fp_2d=NULL;
   char fname[255];

   sprintf(fname, "ar_contour2d_%02d.txt", ncall);
   fn = DebugFilename(fname);
   if(fn)
   fp_2d = fopen(fn, "w");
   sprintf(fname, "ar_contour3d_%02d.txt", ncall++);
   fn = DebugFilename(fname);
   if(fn)
   fp_3d = fopen(fn, "w");

   for (i = 0; i <= nsec; i++)
   {
      angle      = i * rot;
      roma[0][0] =  cos(angle);
      roma[0][1] = -sin(angle);
      roma[1][0] =  sin(angle);
      roma[1][1] =  cos(angle);
      for (j = 0; j < nump; j++)
      {
         x = c->p->x[j] * roma[0][0] + c->p->y[j] * roma[0][1];
         y = c->p->x[j] * roma[1][0] + c->p->y[j] * roma[1][1];
         z = c->p->z[j];
         if (fp_3d)  fprintf(fp_3d, "%10.8f %10.8f %10.8f\n", x, y, z);
      }
      if (fp_3d)  fprintf(fp_3d, "\n");
   }

   for (i = 0; i < nump; i++)
      if (fp_2d)  fprintf(fp_2d, "%10.8f %10.8f %10.8f\n", c->p->x[i], c->p->y[i], c->p->z[i]);

   if (fp_2d)  fclose(fp_2d);
   if (fp_3d)  fclose(fp_3d);
}
#endif                                            // CONTOUR_WIREFRAME
