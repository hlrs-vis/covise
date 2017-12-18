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
#include "include/ar_intersect.h"

int BladeContourIntersect(struct axial *ar)
// caller: CreateAR_BladeElements()
{
   int err = 0;

   if ((ar->mhub = (struct margin *)calloc(1, sizeof(struct margin))) == NULL)
      fatal("memory for (struct margin *) mhub");
   if ((ar->mshroud = (struct margin *)calloc(1, sizeof(struct margin))) == NULL)
      fatal("memory for (struct margin *) mshroud");

   CalcExtension(ar->mhub, ar, 0);
   CalcIntersection(ar->mhub, ar->me[0]->ml, ar->be[1], ar->mod);

   CalcExtension(ar->mshroud, ar, 1);
   CalcIntersection(ar->mshroud, ar->me[ar->be_num-1]->ml, ar->be[ar->be_num-2], ar->mod);

   if(!ar->mhub->cl_int->x ||
      !ar->mshroud->cl_int->x) err = BLADE_CONTOUR_INTERSECT_ERR;
   return err;
}


int CalcExtension(struct margin *ma, struct axial *ar, int side)
// caller: BladeContourIntersect()
{
   int i;
   int be1 = 0;
   int be2 = 0;
   float p[3];
   float r1 = 0;
   float r2 = 0;

   if (ma->ps_ext)
   {
      FreePointStruct(ma->ps_ext);
      ma->ps_ext = NULL;
      FreePointStruct(ma->ps_int);
      ma->ps_int = NULL;
      FreePointStruct(ma->ss_ext);
      ma->ss_ext = NULL;
      FreePointStruct(ma->ss_int);
      ma->ss_int = NULL;
      FreePointStruct(ma->cl_ext);
      ma->cl_ext = NULL;
      FreePointStruct(ma->cl_int);
      ma->cl_int = NULL;
   }
   ma->ps_ext = AllocPointStruct();
   ma->ps_int = AllocPointStruct();
   ma->ss_ext = AllocPointStruct();
   ma->ss_int = AllocPointStruct();
   ma->cl_ext = AllocPointStruct();
   ma->cl_int = AllocPointStruct();

   // set blade elements for extension side
   if (side == 0)
   {
      be1 = 1;
      be2 = 0;
      r1  = ar->be[be1]->rad;
      r2  = ar->be[be2]->rad;
   }
   else if (side == 1)
   {
      be1 = ar->be_num-2;
      be2 = ar->be_num-1;
      r1  = ar->be[be1]->rad;
      r2  = ar->be[be2]->rad;
   }
   else
   {
      fatal("ERROR-calc. of blade extension: Illegal number for extension side.");
   }

   ma->dr = r2 - r1;
   for (i = 0; i < ar->be[be1]->cl->nump; i++)
   {
      p[0] = ar->be[be2]->ps_cart->x[i] - ar->be[be1]->ps_cart->x[i];
      p[1] = ar->be[be2]->ps_cart->y[i] - ar->be[be1]->ps_cart->y[i];
      p[2] = ar->be[be2]->ps_cart->z[i] - ar->be[be1]->ps_cart->z[i];
      AddVPoint(ma->ps_ext, p);
      p[0] = ar->be[be2]->ss_cart->x[i] - ar->be[be1]->ss_cart->x[i];
      p[1] = ar->be[be2]->ss_cart->y[i] - ar->be[be1]->ss_cart->y[i];
      p[2] = ar->be[be2]->ss_cart->z[i] - ar->be[be1]->ss_cart->z[i];
      AddVPoint(ma->ss_ext, p);
      p[0] = ar->be[be2]->cl_cart->x[i] - ar->be[be1]->cl_cart->x[i];
      p[1] = ar->be[be2]->cl_cart->y[i] - ar->be[be1]->cl_cart->y[i];
      p[2] = ar->be[be2]->cl_cart->z[i] - ar->be[be1]->cl_cart->z[i];
      AddVPoint(ma->cl_ext, p);
   }
   // NASTY !!! this does not work properly as intended ?!
   //NormPointStruct(ma->ps_ext);
   //NormPointStruct(ma->ss_ext);

   return 1;
}


#define  CRV_START_ALL  0.30
#define  CRV_END_ALL    0.80
#define CRV_START_INL   0.15
#define CRV_START_BEND  0.00
#define CRV_END_OUTL 1.00

int CalcIntersection(struct margin *ma, struct curve *c, struct be *be, struct model *mod)
//caller: BladeContourIntersect()
{
   int i, j;
   int nc_start, nc_end;
   float p[3], b[3], e[3], h1[2], h2[2];
   float u[2], v1[2], v2[2], m1[2], m2[2], p1[2], p2[2];
   float u_mag, p_dot, t;
   const float s_ext = 1.0;                       //NASTY: depending on spacing of last two be's!!!
   static int ncall = 0;
   FILE *fp_int=NULL, *fp_lin=NULL, *fp_ext=NULL;
   char fname[255];
   char *fn;

   sprintf(fname, "ar_lines_%02d.txt", ncall);
   fn = DebugFilename(fname);
   if(fn)
   fp_lin = fopen(fn, "w");
   sprintf(fname, "ar_extension_%02d.txt", ncall);
   fn = DebugFilename(fname);
   if(fn)
   fp_ext = fopen(fn, "w");
   sprintf(fname, "ar_intersect_%02d.txt", ncall++);
   fn = DebugFilename(fname);
   if(fn)
   fp_int = fopen(fn, "w");

   for (i = 0; i < be->bp->num; i++)
   {
      // pressure side extension
      p[0] = be->ps_cart->x[i];
      p[1] = be->ps_cart->y[i];
      p[2] = be->ps_cart->z[i];
      if (fp_ext) fprintf(fp_ext, "%10.8f %10.8f %10.8f ", p[0], p[1], p[2]);
      // suction side extension
      p[0] = be->ss_cart->x[i];
      p[1] = be->ss_cart->y[i];
      p[2] = be->ss_cart->z[i];
      if (fp_ext) fprintf(fp_ext, "%10.8f %10.8f %10.8f\n", p[0], p[1], p[2]);
   }
   if (fp_ext) fprintf(fp_ext, "\n");
   for (i = 0; i < be->bp->num; i++)
   {
      // pressure side extension
      p[0] = be->ps_cart->x[i] + s_ext * ma->ps_ext->x[i];
      p[1] = be->ps_cart->y[i] + s_ext * ma->ps_ext->y[i];
      p[2] = be->ps_cart->z[i] + s_ext * ma->ps_ext->z[i];
      if (fp_ext) fprintf(fp_ext, "%10.8f %10.8f %10.8f ", p[0], p[1], p[2]);
      // suction side extension
      p[0] = be->ss_cart->x[i] + s_ext * ma->ss_ext->x[i];
      p[1] = be->ss_cart->y[i] + s_ext * ma->ss_ext->y[i];
      p[2] = be->ss_cart->z[i] + s_ext * ma->ss_ext->z[i];
      if (fp_ext) fprintf(fp_ext, "%10.8f %10.8f %10.8f\n", p[0], p[1], p[2]);
   }

   // start and end indices of contour curve to be searched
   // intersection located approx. at middle of core region curve;
   // in case of truncated modelling, nc_start / nc_end have to be
   // adapted to modelled fraction of the meridian curve
   nc_start = (int)(CRV_START_ALL * (c->p->nump - 1));
   nc_end   = (int)(CRV_END_ALL * (c->p->nump - 1));
   if (!mod->inl)
      nc_start = (int)(CRV_START_INL * (c->p->nump -1));
   if (!mod->bend)
      nc_start = (int)(CRV_START_BEND * (c->p->nump -1));
   if (!mod->outl)
      nc_end = (int)(CRV_END_OUTL * (c->p->nump -1));

   // search intersection in meridian plane, r-z-plane
   // pressure side intersection
   for (i = 0; i < be->bp->num; i++)
   {
      // extension base point
      b[0]  = be->rad;
      b[1]  = be->ps_cart->z[i];
      // extension vector
      u[0]  = ma->dr;
      u[1]  = ma->ps_ext->z[i];
      u_mag = (float)sqrt(pow(u[0], 2) + pow(u[1], 2));
      // extension end point
      e[0]  = b[0] + s_ext * u[0];
      e[1]  = b[1] + s_ext * u[1];
      // search contour curve:
      // scalar dot product of perpendicular projection to blade extension of
      // two adjacent contour points must be negative, line segments intersect
      for (j = nc_start; j < nc_end; j++)
      {
         // two contour curve points
         h1[0] = (float)sqrt(pow(c->p->x[j], 2) + pow(c->p->y[j], 2));
         h1[1] = c->p->z[j];
         h2[0] = (float)sqrt(pow(c->p->x[j+1], 2) + pow(c->p->y[j+1], 2));
         h2[1] = c->p->z[j+1];
         // vector difference to extension base point
         v1[0] = h1[0] - b[0];
         v1[1] = h1[1] - b[1];
         v2[0] = h2[0] - b[0];
         v2[1] = h2[1] - b[1];
         // projection of vector difference along extension vector
         m1[0] = (u[0] * v1[0] + u[1] * v1[1])/u_mag * u[0]/u_mag;
         m1[1] = (u[0] * v1[0] + u[1] * v1[1])/u_mag * u[1]/u_mag;
         m2[0] = (u[0] * v2[0] + u[1] * v2[1])/u_mag * u[0]/u_mag;
         m2[1] = (u[0] * v2[0] + u[1] * v2[1])/u_mag * u[1]/u_mag;
         // perpendicular projection to extension vector
         p1[0] = m1[0] - v1[0];
         p1[1] = m1[1] - v1[1];
         p2[0] = m2[0] - v2[0];
         p2[1] = m2[1] - v2[1];
         // scalar dot product of projections
         p_dot = p1[0] * p2[0] + p1[1] * p2[1];
         if (p_dot < 0.0)
         {
            //fprintf(stderr, "i = %d \t j = %d\n", i, j);
            //fprintf(stderr, "p_dot = %10.8f\n", p_dot);
            //fprintf(stderr, "p1,1 = %10.8f  p1,2 = %10.8f\n", p1[0], p1[1]);
            //fprintf(stderr, "p2,1 = %10.8f  p2,2 = %10.8f\n", p2[0], p2[1]);
            if (fp_lin)
            {
               fprintf(fp_lin, "%10.8f %10.8f %10.8f %10.8f\n", b[0], b[1], h1[0], h1[1]);
               fprintf(fp_lin, "%10.8f %10.8f %10.8f %10.8f\n\n\n", e[0], e[1], h2[0], h2[1]);
            }
            t  = (h1[0] - b[0]) * (h2[1] - h1[1]) - (h1[1] - b[1]) * (h2[0] - h1[0]);
            t /= (u[0] * (h2[1] - h1[1]) - u[1] * (h2[0] - h1[0]));
            p[0] = be->ps_cart->x[i] + t * ma->ps_ext->x[i];
            p[1] = be->ps_cart->y[i] + t * ma->ps_ext->y[i];
            p[2] = be->ps_cart->z[i] + t * ma->ps_ext->z[i];
            if (fp_int) fprintf(fp_int, "%10.8f\t%10.8f\t%10.8f\n", p[0], p[1], p[2]);
            AddVPoint(ma->ps_int, p);
            break;
         }
      }
   }
   if (fp_int) fprintf(fp_int, "\n\n");

   // suction side intersection
   for (i = 0; i < be->bp->num; i++)
   {
      // extension base point
      b[0]  = be->rad;
      b[1]  = be->ss_cart->z[i];
      // extension vector
      u[0]  = ma->dr;
      u[1]  = ma->ss_ext->z[i];
      u_mag = (float)sqrt(pow(u[0], 2) + pow(u[1], 2));
      // extension end point
      e[0]  = b[0] + s_ext * u[0];
      e[1]  = b[1] + s_ext * u[1];
      // search contour curve:
      // scalar dot product of perpendicular projection to blade extension of
      // two adjacent contour points must be negative, line segments intersect
      for (j = nc_start; j < nc_end; j++)
      {
         // two contour curve points
         h1[0] = (float)sqrt(pow(c->p->x[j], 2) + pow(c->p->y[j], 2));
         h1[1] = c->p->z[j];
         h2[0] = (float)sqrt(pow(c->p->x[j+1], 2) + pow(c->p->y[j+1], 2));
         h2[1] = c->p->z[j+1];
         // vector difference to extension base point
         v1[0] = h1[0] - b[0];
         v1[1] = h1[1] - b[1];
         v2[0] = h2[0] - b[0];
         v2[1] = h2[1] - b[1];
         // projection of vector difference along extension vector
         m1[0] = (u[0] * v1[0] + u[1] * v1[1])/u_mag * u[0]/u_mag;
         m1[1] = (u[0] * v1[0] + u[1] * v1[1])/u_mag * u[1]/u_mag;
         m2[0] = (u[0] * v2[0] + u[1] * v2[1])/u_mag * u[0]/u_mag;
         m2[1] = (u[0] * v2[0] + u[1] * v2[1])/u_mag * u[1]/u_mag;
         // perpendicular projection to extension vector
         p1[0] = m1[0] - v1[0];
         p1[1] = m1[1] - v1[1];
         p2[0] = m2[0] - v2[0];
         p2[1] = m2[1] - v2[1];
         // scalar dot product of projections
         p_dot = p1[0] * p2[0] + p1[1] * p2[1];
         if (p_dot < 0.0)
         {
            //fprintf(stderr, "i = %d \t j = %d\n", i, j);
            //fprintf(stderr, "p_dot = %10.8f\n", p_dot);
            //fprintf(stderr, "p1,1 = %10.8f  p1,2 = %10.8f\n", p1[0], p1[1]);
            //fprintf(stderr, "p2,1 = %10.8f  p2,2 = %10.8f\n", p2[0], p2[1]);
            if (fp_lin)
            {
               fprintf(fp_lin, "%10.8f %10.8f %10.8f %10.8f\n", b[0], b[1], h1[0], h1[1]);
               fprintf(fp_lin, "%10.8f %10.8f %10.8f %10.8f\n\n\n", e[0], e[1], h2[0], h2[1]);
            }
            t  = (h1[0] - b[0]) * (h2[1] - h1[1]) - (h1[1] - b[1]) * (h2[0] - h1[0]);
            t /= (u[0] * (h2[1] - h1[1]) - u[1] * (h2[0] - h1[0]));
            p[0] = be->ss_cart->x[i] + t * ma->ss_ext->x[i];
            p[1] = be->ss_cart->y[i] + t * ma->ss_ext->y[i];
            p[2] = be->ss_cart->z[i] + t * ma->ss_ext->z[i];
            if (fp_int) fprintf(fp_int, "%10.8f\t%10.8f\t%10.8f\n", p[0], p[1], p[2]);
            AddVPoint(ma->ss_int, p);
            break;
         }
      }
   }

   // centre line intersection
   for (i = 0; i < be->bp->num; i++)
   {
      // extension base point
      b[0]  = be->rad;
      b[1]  = be->cl_cart->z[i];
      // extension vector
      u[0]  = ma->dr;
      u[1]  = ma->cl_ext->z[i];
      u_mag = (float)sqrt(pow(u[0], 2) + pow(u[1], 2));
      // extension end point
      e[0]  = b[0] + s_ext * u[0];
      e[1]  = b[1] + s_ext * u[1];
      // search contour curve:
      // scalar dot product of perpendicular projection to blade extension of
      // two adjacent contour points must be negative, line segments intersect
      for (j = nc_start; j < nc_end; j++)
      {
         // two contour curve points
         h1[0] = (float)sqrt(pow(c->p->x[j], 2) + pow(c->p->y[j], 2));
         h1[1] = c->p->z[j];
         h2[0] = (float)sqrt(pow(c->p->x[j+1], 2) + pow(c->p->y[j+1], 2));
         h2[1] = c->p->z[j+1];
         // vector difference to extension base point
         v1[0] = h1[0] - b[0];
         v1[1] = h1[1] - b[1];
         v2[0] = h2[0] - b[0];
         v2[1] = h2[1] - b[1];
         // projection of vector difference along extension vector
         m1[0] = (u[0] * v1[0] + u[1] * v1[1])/u_mag * u[0]/u_mag;
         m1[1] = (u[0] * v1[0] + u[1] * v1[1])/u_mag * u[1]/u_mag;
         m2[0] = (u[0] * v2[0] + u[1] * v2[1])/u_mag * u[0]/u_mag;
         m2[1] = (u[0] * v2[0] + u[1] * v2[1])/u_mag * u[1]/u_mag;
         // perpendicular projection to extension vector
         p1[0] = m1[0] - v1[0];
         p1[1] = m1[1] - v1[1];
         p2[0] = m2[0] - v2[0];
         p2[1] = m2[1] - v2[1];
         // scalar dot product of projections
         p_dot = p1[0] * p2[0] + p1[1] * p2[1];
         if (p_dot < 0.0)
         {
            //fprintf(stderr, "i = %d \t j = %d\n", i, j);
            //fprintf(stderr, "p_dot = %10.8f\n", p_dot);
            //fprintf(stderr, "p1,1 = %10.8f  p1,2 = %10.8f\n", p1[0], p1[1]);
            //fprintf(stderr, "p2,1 = %10.8f  p2,2 = %10.8f\n", p2[0], p2[1]);
            if (fp_lin)
            {
               fprintf(fp_lin, "%10.8f %10.8f %10.8f %10.8f\n", b[0], b[1], h1[0], h1[1]);
               fprintf(fp_lin, "%10.8f %10.8f %10.8f %10.8f\n\n\n", e[0], e[1], h2[0], h2[1]);
            }
            t  = (h1[0] - b[0]) * (h2[1] - h1[1]) - (h1[1] - b[1]) * (h2[0] - h1[0]);
            t /= (u[0] * (h2[1] - h1[1]) - u[1] * (h2[0] - h1[0]));
            p[0] = be->cl_cart->x[i] + t * ma->cl_ext->x[i];
            p[1] = be->cl_cart->y[i] + t * ma->cl_ext->y[i];
            p[2] = be->cl_cart->z[i] + t * ma->cl_ext->z[i];
            if (fp_int) fprintf(fp_int, "%10.8f\t%10.8f\t%10.8f\n", p[0], p[1], p[2]);
            AddVPoint(ma->cl_int, p);
            break;
         }
      }
   }
   if (fp_lin) fclose(fp_lin);
   if (fp_int) fclose(fp_int);
   if (fp_ext) fclose(fp_ext);
   return 1;
}
