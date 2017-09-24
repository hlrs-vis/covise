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
#include "include/ar_createbe.h"
#include "include/ar_intersect.h"

#define INIT_PORTION    25
#define BSPLN_DEGREE    3

#define TOLERANCE       1.0e-6
#define NPOIN_HELP_POLY    7
#define POLY_SECTION_RATIO 0.4

int CreateAR_BladeElements(struct axial *ar)
//caller: ReadAxialRunner()
{
   int i;
   int err = 0;

   DumpAR(ar);

   // modify pivot location for blade elements
   for (i = 0; i < ar->be_num; i++)
   {
      ar->be[i]->pivot = (ar->piv-ar->be[i]->lec)/
         (1.0f-ar->be[i]->lec-ar->be[i]->tec);
   }

   for (i = 0; i < ar->be_num; i++)
   {
      ar->be[i]->rad     = 0.5f * ar->ref * (ar->diam[0] + (1.0f - ar->diam[0]) * ar->be[i]->para);
      ar->be[i]->le_part[0] = ar->le_part[0];
      ar->be[i]->le_part[1] = ar->le_part[1];
      ar->be[i]->le_part[2] = ar->le_part[2];
      ar->be[i]->te_part[0] = ar->te_part[0];
      ar->be[i]->te_part[1] = ar->te_part[1];
      SurfacesAR_BladeElement(ar->be[i], ar->bangle,
         ar->ref, ar->rot_clockwise,
         ar->clspline);
   }
   err = BladeContourIntersect(ar);

   WriteGNU_AR(ar);

   return err;
}


void DetermineCoefficients(float *x, float *y, float *a)
// caller: ModifyAR_BladeElements4Covise()
{
   float buf;

   a[2]  = (y[1] - y[2]) * (x[0] - x[1]) - (y[0] - y[1]) * (x[1] - x[2]);
   buf   = float((pow(x[1], 2) - pow(x[2], 2)) * (x[0] - x[1]));
   buf  -= float(((pow(x[0], 2) - pow(x[1], 2)) * (x[1] - x[2])));
   a[2] /= buf;
   a[1]  = float(((y[0] - y[1]) - (pow(x[0], 2) - pow(x[1], 2)) * a[2]) / (x[0] - x[1]));
   a[0]  = y[2] - float(pow(x[2], 2)) * a[2] - x[2] * a[1];
}


float EvaluateParameter(float x, float *a)
// caller: ModifyAR_BladeElements4Covise()
{
   float val;

   val = a[2] * float(pow(x, 2)) + a[1] * x + a[0];
   return val;
}


int ModifyAR_BladeElements4Covise(struct axial *ar)
// caller: CreateAR_BladeElements()
{
   int i;
   float a[3];                                    // coefficients for parameter distribution, 2nd order
   float x[3], y[3];                              // locations and values to function
   // location of sliders; DO NOT CHANGE !!!
   const int left   = 0;
   const int middle = (int)(ar->be_num / 2);
   const int right  = ar->be_num - 1;

   if (ar->be_single) return 0;

   // slider parameter locations
   x[0] = ar->be[left]->para;
   x[1] = ar->be[middle]->para;
   x[2] = ar->be[right]->para;

   // inlet angle
   y[0] = ar->be[left]->angle[0];
   y[1] = ar->be[middle]->angle[0];
   y[2] = ar->be[right]->angle[0];
   DetermineCoefficients(&x[0], &y[0], &a[0]);
   for (i = 0; i < ar->be_num && !ar->parlock[LOCK_INANGLE]; i++)
      ar->be[i]->angle[0]= EvaluateParameter(ar->be[i]->para, &a[0]);

   // outlet angle
   y[0] = ar->be[left]->angle[1];
   y[1] = ar->be[middle]->angle[1];
   y[2] = ar->be[right]->angle[1];
   DetermineCoefficients(&x[0], &y[0], &a[0]);
   for (i = 0; i < ar->be_num && !ar->parlock[LOCK_OUTANGLE]; i++)
      ar->be[i]->angle[1]= EvaluateParameter(ar->be[i]->para, &a[0]);

   // inlet angle mod.
   y[0] = ar->be[left]->mod_angle[0];
   y[1] = ar->be[middle]->mod_angle[0];
   y[2] = ar->be[right]->mod_angle[0];
   DetermineCoefficients(&x[0], &y[0], &a[0]);
   for (i = 0; i < ar->be_num && !ar->parlock[LOCK_INMOD]; i++)
      ar->be[i]->mod_angle[0]= EvaluateParameter(ar->be[i]->para, &a[0]);

   // outlet angle mod.
   y[0] = ar->be[left]->mod_angle[1];
   y[1] = ar->be[middle]->mod_angle[1];
   y[2] = ar->be[right]->mod_angle[1];
   DetermineCoefficients(&x[0], &y[0], &a[0]);
   for (i = 0; i < ar->be_num && !ar->parlock[LOCK_OUTMOD]; i++)
      ar->be[i]->mod_angle[1]= EvaluateParameter(ar->be[i]->para, &a[0]);

   // profile thickness
   y[0] = ar->be[left]->p_thick;
   y[1] = ar->be[middle]->p_thick;
   y[2] = ar->be[right]->p_thick;
   DetermineCoefficients(&x[0], &y[0], &a[0]);
   for (i = 0; i < ar->be_num && !ar->parlock[LOCK_PTHICK]; i++)
      ar->be[i]->p_thick = EvaluateParameter(ar->be[i]->para, &a[0]);

   // trailing edge thickness
   y[0] = ar->be[left]->te_thick;
   y[1] = ar->be[middle]->te_thick;
   y[2] = ar->be[right]->te_thick;
   DetermineCoefficients(&x[0], &y[0], &a[0]);
   for (i = 0; i < ar->be_num && !ar->parlock[LOCK_TETHICK]; i++)
      ar->be[i]->te_thick= EvaluateParameter(ar->be[i]->para, &a[0]);

   // maximum camber
   y[0] = ar->be[left]->camb;
   y[1] = ar->be[middle]->camb;
   y[2] = ar->be[right]->camb;
   DetermineCoefficients(&x[0], &y[0], &a[0]);
   for (i = 0; i < ar->be_num && !ar->parlock[LOCK_MAXCAMB]; i++)
      ar->be[i]->camb = EvaluateParameter(ar->be[i]->para, &a[0]);

   // camber position
   y[0] = ar->be[left]->camb_pos;
   y[1] = ar->be[middle]->camb_pos;
   y[2] = ar->be[right]->camb_pos;
   DetermineCoefficients(&x[0], &y[0], &a[0]);
   for (i = 0; i < ar->be_num && !ar->parlock[LOCK_CAMBPOS]; i++)
      ar->be[i]->camb_pos= EvaluateParameter(ar->be[i]->para, &a[0]);

   // profile shift
   y[0] = ar->be[left]->bp_shift;
   y[1] = ar->be[middle]->bp_shift;
   y[2] = ar->be[right]->bp_shift;
   DetermineCoefficients(&x[0], &y[0], &a[0]);
   for (i = 0; i < ar->be_num && !ar->parlock[LOCK_BPSHIFT]; i++)
      ar->be[i]->bp_shift= EvaluateParameter(ar->be[i]->para, &a[0]);

   return 1;
}


int SurfacesAR_BladeElement(struct be *be, float bangle, float /*ref*/,
int clock, int clspline)
// caller: CreateAR_BladeElements()
{
   int i, t_sec;
   float phi_r, cl_sec, scale_te, te, t, tmax, angle, factor;
   float p1[3], p2[3], p3[3], p4[3], q1[3], q2[3], q3[3],q4[3],roma[2][2];
   float qeq_a, qeq_b, qeq_c, qeq_r, s[3], m[3];
   float ratio, len, alpha = 0.0;
   struct Point *he_poly = NULL;
   struct Flist *he_knot = NULL;
   struct Point *cl_poly = NULL;
   struct Flist *cl_knot = NULL;
   FILE *fp=NULL, *fp2=NULL;
   char fname[255];
   char *fn;
   static int nbe = 0;

   sprintf(fname, "ar_beplane_%02d.txt", nbe++);
   fn = DebugFilename(fname);
   if(fn)
       fp = fopen(fn, "w");

   // delete previous data and allocate new
   if (cl_poly)
   {
      FreePointStruct(cl_poly);
      cl_poly = NULL;
   }
   if (cl_knot)
   {
      FreeFlistStruct(cl_knot);
      cl_knot = NULL;
   }
   FreePointStruct(be->cl);
   FreePointStruct(be->clg);
   FreePointStruct(be->ps);
   FreePointStruct(be->ss);
   FreePointStruct(be->cl_cart);
   FreePointStruct(be->ps_cart);
   FreePointStruct(be->ss_cart);

   // CALCULATE BLADE TRIANGLE FROM ANGLES AND ARC LENGTH
   // solve quadratic equation to determine p2 and p4
   s[0]  = s[1]  = s[2]  = 0.0;
   qeq_a = qeq_b = qeq_c = qeq_r = 0.0;

   qeq_a  = float((1.0 - be->camb_pos) * pow(tan(RAD(be->angle[0])), 2));
   qeq_a += float((2.0 * be->camb_pos - 1.0) * tan(RAD(be->angle[0])) * tan(RAD(be->angle[1])));
   qeq_a -= float(be->camb_pos * pow(tan(RAD(be->angle[1])), 2));

   qeq_b  = float((1.0 - 2.0 * be->camb_pos) * tan(RAD(be->angle[0])) * tan(RAD(be->angle[1])));
   qeq_b += float(2.0 * (be->camb_pos - 1.0) * pow(tan(RAD(be->angle[0])), 2));
   qeq_b -= 1.0f;

   qeq_c  = float((1.0f - be->camb_pos) * pow(tan(RAD(be->angle[0])), 2));
   qeq_c += (1.0f - be->camb_pos);

   qeq_r = float(pow(qeq_b, 2) + 4.0 * qeq_a * qeq_c);
   if (qeq_r < 0.0)
   {
      fatal("ERROR-calc. of quadratic eqn. for camber position: only complex solutions.");
   }
   else
   {
      s[0] = float((-qeq_b + sqrt(pow(qeq_b, 2) - 4.0 * qeq_a * qeq_c)) / (2.0 * qeq_a));
      s[1] = float(s[0] * tan(RAD(be->angle[1])));
      if ((s[0] < (0.0 - TOLERANCE)) || (s[0] > (1.0 + TOLERANCE)))
      {
         s[0] = float((-qeq_b - sqrt(pow(qeq_b, 2) - 4.0 * qeq_a * qeq_c)) / (2.0 * qeq_a));
         s[1] = float(s[0] * tan(RAD(be->angle[1])));
         if ((s[0] < (0.0 - TOLERANCE)) || (s[0] > (1.0 + TOLERANCE)))
         {
            fatal("ERROR-calc. of camber position: solutions are outside [0.0;1.0].");
         }
      }
   }

   // calculate blade triangle points, scale to arc length
   phi_r = be->bl_wrap * be->rad;

   p1[0] = p1[1] = p1[2] = 0.0;
   p2[0] = p2[1] = p2[2] = 0.0;
   p3[0] = p3[1] = p3[2] = 0.0;
   p4[0] = p4[1] = p4[2] = 0.0;

   p1[0] = phi_r;
   p1[1] = float(phi_r * (s[0] * tan(RAD(be->angle[1])) + (1.0 - s[0]) *
      tan(RAD(be->angle[0]))));

   p2[0] = s[0] * phi_r;
   p2[1] = s[1] * phi_r;

   p3[0] = p3[1] = p3[2] = 0.0f;

   p4[0] = (1.0f - be->camb_pos) * p1[0];
   p4[1] = (1.0f - be->camb_pos) * p1[1];

   // rotate blade triangle around TE point, base to x-axis
   q1[0] = q1[1] = q1[2] = 0.0;
   q2[0] = q2[1] = q2[2] = 0.0;
   q3[0] = q3[1] = q3[2] = 0.0;
   q4[0] = q4[1] = q4[2] = 0.0;

   angle      = float(M_PI - atan(p1[1]/p1[0]));
   roma[0][0] = float(cos(angle));
   roma[0][1] = float(-sin(angle));
   roma[1][0] = float(sin(angle));
   roma[1][1] = float(cos(angle));

   q1[0] = roma[0][0] * p1[0] + roma[0][1] * p1[1];
   q1[1] = roma[1][0] * p1[0] + roma[1][1] * p1[1];

   q2[0] = roma[0][0] * p2[0] + roma[0][1] * p2[1];
   q2[1] = roma[1][0] * p2[0] + roma[1][1] * p2[1];

   q3[0] = roma[0][0] * p3[0] + roma[0][1] * p3[1];
   q3[1] = roma[1][0] * p3[0] + roma[1][1] * p3[1];

   q4[0] = roma[0][0] * p4[0] + roma[0][1] * p4[1];
   q4[1] = roma[1][0] * p4[0] + roma[1][1] * p4[1];

   // CL AND SURFACE CALCULATION
   cl_poly = AllocPointStruct();
   // **************************************************
   // create curve from two single splines
   if(clspline)
   {
      // move q4 according to blade element camber and skew param.
      q4[0] += (be->le_part[2]-0.5f)*(q1[0]-q3[0]);
      q4[1] = be->camb * q2[1];
      // le point (q1) and first two le points on cl polygon
      AddVPoint(cl_poly, q1);
      ratio = be->camb / 12.0f;
      s[0]  = q1[0] + ratio * (q2[0] - q1[0]);
      s[1]  = q1[1] + ratio * (q2[1] - q1[1]);
      AddVPoint(cl_poly, s);
      /*ratio = be->camb / 4.0;
        s[0]  = q1[0] + ratio * (q2[0] - q1[0]);
        s[1]  = q1[1] + ratio * (q2[1] - q1[1]);
        s[2]  = 0.0;
        AddVPoint(cl_poly, s);*/
      // help polygon from third le point to q4 (max camber point)
      m[1] = q4[1];                               // always
      m[2] = 0.0;                                 // always
      m[0] = q1[0] + (q2[0] - q1[0]) / (q2[1] - q1[1]) * (m[1] - q1[1]);
      he_poly = CurvePolygon(s, m, q4, be->le_part[0], be->le_part[1]);
      he_knot = BSplineKnot(he_poly, BSPLN_DEGREE);
      for(i = 1; i < NPOIN_HELP_POLY; i++)
      {
         ratio = (float)i / (float)(NPOIN_HELP_POLY-1);
         BSplinePoint(BSPLN_DEGREE, he_poly, he_knot, ratio, s);
         AddVPoint(cl_poly, s);
      }
      FreePointStruct(he_poly);
      FreeFlistStruct(he_knot);
      // help polygon from q4 to third last te point
      he_poly = AllocPointStruct();
      he_knot = AllocFlistStruct(INIT_PORTION);
      m[0]  = q3[0] + (q2[0] - q3[0]) / (q2[1] - q3[1]) * (m[1] - q3[1]);
      ratio = be->camb / 8.0f;
      s[0]  = q3[0] + ratio * (q2[0] - q3[0]);
      s[1]  = q3[1] + ratio * (q2[1] - q3[1]);
      s[2]  = 0.0;
      he_poly = CurvePolygon(q4, m, s, be->te_part[0], be->te_part[1]);
      he_knot = BSplineKnot(he_poly, BSPLN_DEGREE);
      for (i = 1; i < NPOIN_HELP_POLY; i++)
      {
         ratio = (float)i / (float)(NPOIN_HELP_POLY-1);
         BSplinePoint(BSPLN_DEGREE, he_poly, he_knot, ratio, m);
         AddVPoint(cl_poly, m);
      }
      // last two te points on cl polygon and te point (q3)
      AddVPoint(cl_poly, s);
      /*ratio = be->camb / 8.0;
        s[0]  = q3[0] + ratio * (q2[0] - q3[0]);
        s[1]  = q3[1] + ratio * (q2[1] - q3[1]);
        s[2]  = 0.0;
        AddVPoint(cl_poly, s);*/
   }
   else
   {
      // only one spline in triangle
      cl_poly->nump = 0;
      AddVPoint(cl_poly, q1);
      ratio = be->camb / 12.0f;
      s[0]  = q1[0] + ratio * (q2[0] - q1[0]);
      s[1]  = q1[1] + ratio * (q2[1] - q1[1]);
      AddVPoint(cl_poly,s);
      ratio = be->camb / 8.0f;
      m[0]  = q3[0] + ratio * (q2[0] - q3[0]);
      m[1]  = q3[1] + ratio * (q2[1] - q3[1]);
      he_poly = CurvePolygon(s, q2, m, be->le_part[0], be->le_part[1]);
      he_knot = BSplineKnot(he_poly, BSPLN_DEGREE);
      for (i = 1; i < 2*NPOIN_HELP_POLY; i++)
      {
         ratio = (float)i / (float)(2*NPOIN_HELP_POLY-1);
         BSplinePoint(BSPLN_DEGREE, he_poly, he_knot, ratio, m);
         AddVPoint(cl_poly, m);
      }
   }
   AddVPoint(cl_poly, q3);
   FreePointStruct(he_poly);
   FreeFlistStruct(he_knot);

   // INDEX 0: cl polygon points, construction
   for (i = 0; fp && i < cl_poly->nump; i++)
   {
      if(i < cl_poly->nump-1)
         alpha = float(atan((cl_poly->y[i]-cl_poly->y[i+1])/
            (cl_poly->x[i]-cl_poly->x[i+1])));
      fprintf(fp, "%f %f   %f\n", cl_poly->x[i], cl_poly->y[i],
         alpha*180.0/M_PI);
   }
   if (fp)  fprintf(fp, "\n\n");

   // rotate back cl polygon points, calculate arc length
   roma[0][0] = float(cos(-angle));
   roma[0][1] = float(-sin(-angle));
   roma[1][0] = float(sin(-angle));
   roma[1][1] = float(cos(-angle));
   for(i = 0; i < cl_poly->nump; i++)
   {
      s[0]          = roma[0][0] * cl_poly->x[i] + roma[0][1] * cl_poly->y[i];
      s[1]          = roma[1][0] * cl_poly->x[i] + roma[1][1] * cl_poly->y[i];
      s[2]          = 0.0;
      cl_poly->x[i] = s[0];
      cl_poly->y[i] = s[1];
      cl_poly->z[i] = s[2];
      if (i)
      {
         len  = float(pow((cl_poly->x[i] - cl_poly->x[i-1]), 2));
         len += float(pow((cl_poly->y[i] - cl_poly->y[i-1]), 2));
         len += float(pow((cl_poly->z[i] - cl_poly->z[i-1]), 2));
         len  = float(sqrt(len));
         be->cl_len += len;
      }
   }

   // INDEX 1: cl polygon points, rotated back
   if(fp) fprintf(fp,"# cl polygon points, INDEX 1\n");
   for (i = 0; fp && i < cl_poly->nump; i++)
   {
      if(i < cl_poly->nump-1)
         alpha = float(atan((cl_poly->y[i]-cl_poly->y[i+1])/
            (cl_poly->x[i]-cl_poly->x[i+1])));
      fprintf(fp, "%f %f    %f\n", cl_poly->x[i], cl_poly->y[i],
         alpha*180.0/M_PI);
   }
   if (fp)  fprintf(fp, "\n\n");

   // CL AND CL GRADIENT
   be->cl  = AllocPointStruct();
   be->clg = AllocPointStruct();
   cl_knot = BSplineKnot(cl_poly, BSPLN_DEGREE);
   // cl points and gradient
   sprintf(fname, "ar_becl_%02d.txt", nbe-1);
   fn = DebugFilename(fname);
   if(fn)
   fp2 = fopen(fn, "w");
   for (i = 0; i < be->bp->num; i++)
   {
      cl_sec = float(pow(be->bp->c[i], be->bp_shift));
      BSplinePoint(BSPLN_DEGREE, cl_poly, cl_knot, cl_sec, &s[0]);
      AddVPoint(be->cl, s);
      BSplineNormal(BSPLN_DEGREE, cl_poly, cl_knot, cl_sec, &s[0]);
      AddVPoint(be->clg, s);
      if(fp2) fprintf(fp2,"%f %f %f\n",s[0], s[1], s[2]);
   }
   if(fp2) fclose(fp2);

   // CALCULATE PS/SS
   // scale_te is tot. profile thickness scale up
   be->ss  = AllocPointStruct();
   be->ps  = AllocPointStruct();
   t_sec   = be->bp->t_sec;
   tmax    = be->bp->t[t_sec] * be->cl_len + be->bp->c[t_sec] * be->te_thick;
   scale_te = be->p_thick / tmax;
   for (i = 0; i < be->bp->num; i++)
   {
      // pressure side (-) and suction side (+), machine size
      te   = 0.5f * be->bp->c[i] * be->te_thick;
      t    = 0.5f * be->cl_len   * be->bp->t[i];
      s[0] = be->cl->x[i] - be->clg->x[i] * (scale_te * t + te);
      s[1] = be->cl->y[i] - be->clg->y[i] * (scale_te * t + te);
      s[2] = be->cl->z[i] - be->clg->z[i] * (scale_te * t + te);
      AddVPoint(be->ps, s);
      s[0] = be->cl->x[i] + be->clg->x[i] * (scale_te * t + te);
      s[1] = be->cl->y[i] + be->clg->y[i] * (scale_te * t + te);
      s[2] = be->cl->z[i] + be->clg->z[i] * (scale_te * t + te);
      AddVPoint(be->ss, s);
   }

   // INDEX 2: cl, ps/ss surfaces
   for (i = 0; fp  && i < be->cl->nump; i++)
   {
      if(i < be->cl->nump-1)
         alpha = float(atan((be->cl->y[i]-be->cl->y[i+1])/
            (be->cl->x[i]-be->cl->x[i+1])));

      fprintf(fp, "%f %f  %f  ", be->cl->x[i], be->cl->y[i], alpha*180.0/M_PI);
      fprintf(fp, "%f %f  %f  ", be->ps->x[i], be->ps->y[i], be->ps->z[i]);
      fprintf(fp, "%f %f  %f\n", be->ss->x[i], be->ss->y[i], be->ss->z[i]);
   }
   if (fp)  fprintf(fp, "\n\n");

   // calculate pivot coords, translate blade elements
   BSplinePoint(BSPLN_DEGREE, cl_poly, cl_knot, be->pivot, &s[0]);
   for (i = 0; i < be->cl->nump; i++)
   {
      be->cl->x[i] -= s[0];
      be->cl->y[i] -= s[1];
      be->cl->z[i] -= s[2];
      be->ps->x[i] -= s[0];
      be->ps->y[i] -= s[1];
      be->ps->z[i] -= s[2];
      be->ss->x[i] -= s[0];
      be->ss->y[i] -= s[1];
      be->ss->z[i] -= s[2];
   }

   // INDEX 3: cl, ps/ss surfaces, translated
   for (i = 0; fp && i < be->cl->nump; i++)
   {
      fprintf(fp, "%f %f  %f  ", be->cl->x[i], be->cl->y[i], be->cl->z[i]);
      fprintf(fp, "%f %f  %f  ", be->ps->x[i], be->ps->y[i], be->ps->z[i]);
      fprintf(fp, "%f %f  %f\n", be->ss->x[i], be->ss->y[i], be->ss->z[i]);
   }
   if (fp)  fprintf(fp, "\n\n");

   // rotate blade element around pivot
   roma[0][0] = float(cos(RAD(bangle)));
   roma[0][1] = float(-sin(RAD(bangle)));
   roma[1][0] = float(sin(RAD(bangle)));
   roma[1][1] = float(cos(RAD(bangle)));
   for (i = 0; i < be->cl->nump; i++)
   {
      s[0] = roma[0][0] * be->cl->x[i] + roma[0][1] * be->cl->y[i];
      s[1] = roma[1][0] * be->cl->x[i] + roma[1][1] * be->cl->y[i];
      s[2] = 0.0;
      be->cl->x[i] = s[0];
      be->cl->y[i] = s[1];
      be->cl->z[i] = s[2];
      s[0] = roma[0][0] * be->ps->x[i] + roma[0][1] * be->ps->y[i];
      s[1] = roma[1][0] * be->ps->x[i] + roma[1][1] * be->ps->y[i];
      s[2] = 0.0;
      be->ps->x[i] = s[0];
      be->ps->y[i] = s[1];
      be->ps->z[i] = s[2];
      s[0] = roma[0][0] * be->ss->x[i] + roma[0][1] * be->ss->y[i];
      s[1] = roma[1][0] * be->ss->x[i] + roma[1][1] * be->ss->y[i];
      s[2] = 0.0;
      be->ss->x[i] = s[0];
      be->ss->y[i] = s[1];
      be->ss->z[i] = s[2];
   }

   // INDEX 4: cl, ps/ss surfaces, translated and rotated
   for (i = 0; fp && i < be->cl->nump; i++)
   {
      if(i < be->cl->nump-1)
         alpha = float(atan((be->cl->y[i]-be->cl->y[i+1])/
            (be->cl->x[i]-be->cl->x[i+1])));
      fprintf(fp, "%f %f  %f  ", be->cl->x[i], be->cl->y[i], alpha*180.0/M_PI);
      fprintf(fp, "%f %f  %f  ", be->ps->x[i], be->ps->y[i], be->ps->z[i]);
      fprintf(fp, "%f %f  %f\n", be->ss->x[i], be->ss->y[i], be->ss->z[i]);
   }
   if (fp)  fprintf(fp, "\n\n");

   // transform into cartesian coords
   be->cl_cart = AllocPointStruct();
   be->ps_cart = AllocPointStruct();
   be->ss_cart = AllocPointStruct();
   for (i = 0; i < be->cl->nump; i++)
   {
      if(clock) factor = -1.0f/be->rad;
      else factor = 1.0f/be->rad;
      s[0] = float(be->rad * cos(be->cl->x[i]*factor));
      s[1] = float(be->rad * sin(be->cl->x[i]*factor));
      s[2] = be->cl->y[i];
      AddVPoint(be->cl_cart, s);
      s[0] = float(be->rad * cos(be->ps->x[i]*factor));
      s[1] = float(be->rad * sin(be->ps->x[i]*factor));
      s[2] = be->ps->y[i];
      AddVPoint(be->ps_cart, s);
      s[0] = float(be->rad * cos(be->ss->x[i]*factor));
      s[1] = float(be->rad * sin(be->ss->x[i]*factor));
      s[2] = be->ss->y[i];
      AddVPoint(be->ss_cart, s);
   }
   if (fp)  fclose(fp);
   return 1;
}
