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
#include "include/axial.h"
#include "include/ar_updatemer.h"
#include "include/ar_blademer.h"
#include "../BSpline/include/bspline.h"

int UpdateAR_Meridians(struct axial *ar)
// caller: ReadAxialRunner()
{
   int i, j, err = 0;
   float x[3], y[3];
   struct curve *bl_cl = NULL;
   struct curve *bl_ps = NULL;
   struct curve *bl_ss = NULL;
   struct Point *inter_cl = NULL;
   struct Point *inter_ps = NULL;
   struct Point *inter_ss = NULL;
   char *fn;
   char tmp[50];
   float alpha=0.0;
   FILE *fp_cl=NULL, *fp_ps=NULL, *fp_ss=NULL;
   static int check_index = 3;
   FILE *fp=NULL;

   fn = DebugFilename("ar_meridianup_cl.txt");
   if(fn)
   fp_cl = fopen(fn, "w");
   fn = DebugFilename("ar_meridianup_ps.txt");
   if(fn)
   fp_ps = fopen(fn, "w");
   fn = DebugFilename("ar_meridianup_ss.txt");
   if(fn)
   fp_ss = fopen(fn, "w");
   sprintf(tmp, "ar_bl_check_%d.txt", check_index);
   fn = DebugFilename(tmp);
   if(fn)
   fp = fopen(fn, "w");

   if (fp)
   {
      fprintf(fp, "# blade line %d at beginning of UpdateAR_Meridians()\n", check_index);
      for (i = 0; i < ar->be_num; i++)
      {
         fprintf(fp, "%f  ", ar->be[i]->cl_cart->x[check_index]);
         fprintf(fp, "%f  ", ar->be[i]->cl_cart->y[check_index]);
         fprintf(fp, "%f\n", ar->be[i]->cl_cart->z[check_index]);
      }
      fprintf(fp, "\n\n");
   }
   // memory for intersection data
   for (i = 0; i < ar->be_num; i++)
   {
      if (ar->me[i]->cl)
      {
         FreePointStruct(ar->me[i]->cl);
         FreePointStruct(ar->me[i]->ps);
         FreePointStruct(ar->me[i]->ss);
      }
      ar->me[i]->cl = AllocPointStruct();
      ar->me[i]->ps = AllocPointStruct();
      ar->me[i]->ss = AllocPointStruct();
   }

   // for each blade line (hub to shroud: cl, ps, ss)
   for (j = 0; j < ar->be[0]->cl_cart->nump; j++)
   {
      if (bl_cl)
      {
         FreeCurveStruct(bl_cl);
         FreeCurveStruct(bl_ps);
         FreeCurveStruct(bl_ss);
         FreePointStruct(inter_cl);
         FreePointStruct(inter_ps);
         FreePointStruct(inter_ss);
      }
      bl_cl    = AllocCurveStruct();
      bl_ps    = AllocCurveStruct();
      bl_ss    = AllocCurveStruct();
      inter_cl = AllocPointStruct();
      inter_ps = AllocPointStruct();
      inter_ss = AllocPointStruct();

      if (fp_cl)  fprintf(fp_cl, "# bladeline\n");
      if (fp_ps)  fprintf(fp_ps, "# bladeline\n");
      if (fp_ss)  fprintf(fp_ss, "# bladeline\n");
      if (fp && j == check_index)
         fprintf(fp, "# blade line at initialisation\n");

      for (i = 0; i < ar->be_num; i++)
      {
         // centre line, bl
         x[0] = ar->be[i]->cl_cart->x[j];
         x[1] = ar->be[i]->cl_cart->y[j];
         x[2] = ar->be[i]->cl_cart->z[j];
         if (fp_cl)  fprintf(fp_cl, "%f  %f  %f\n", x[0], x[1], x[2]);
         if (fp && j == check_index)
         {
            fprintf(fp, "%f  %f  %f\n", x[0], x[1], x[2]);
         }
         CalcCylindricalCoords(x);
         AddCurvePoint(bl_cl, x[0], x[1], x[2], 0.0, 0.0);
         // pressure side, bl
         x[0] = ar->be[i]->ps_cart->x[j];
         x[1] = ar->be[i]->ps_cart->y[j];
         x[2] = ar->be[i]->ps_cart->z[j];
         if (fp_ps)  fprintf(fp_ps, "%f  %f  %f\n", x[0], x[1], x[2]);
         CalcCylindricalCoords(x);
         AddCurvePoint(bl_ps, x[0], x[1], x[2], 0.0, 0.0);
         // suction side, bl
         x[0] = ar->be[i]->ss_cart->x[j];
         x[1] = ar->be[i]->ss_cart->y[j];
         x[2] = ar->be[i]->ss_cart->z[j];
         if (fp_ss)  fprintf(fp_ss, "%f  %f  %f\n", x[0], x[1], x[2]);
         CalcCylindricalCoords(x);
         AddCurvePoint(bl_ss, x[0], x[1], x[2], 0.0, 0.0);
      }
      err = BladeMeridianIntersection2(bl_cl, ar, inter_cl);
      err = BladeMeridianIntersection2(bl_ps, ar, inter_ps);
      err = BladeMeridianIntersection2(bl_ss, ar, inter_ss);
      if(err) return err;

      if (fp && j == check_index)
      {
         fprintf(fp, "\n\n");
         fprintf(fp, "# entire intersection (index %d)\n", j);
      }
      if (fp_cl)  fprintf(fp_cl, "\n\n");
      if (fp_cl)  fprintf(fp_cl, "# entire intersection (index %d)\n", j);
      x[0] = y[0] = ar->mhub->cl_int->x[j];
      x[1] = y[1] = ar->mhub->cl_int->y[j];
      x[2] = y[2] = ar->mhub->cl_int->z[j];
      if (fp_cl)  fprintf(fp_cl, "%f  %f  %f\n", y[0], y[1], y[2]);
      //AddVPoint(ar->me[0]->cl, y);
      if (fp && j == check_index)
      {
         fprintf(fp, "%f  %f  %f\n", y[0], y[1], y[2]);
      }
      for (i = 0; i < inter_cl->nump; i++)
      {
         x[0] = inter_cl->x[i];
         x[1] = inter_cl->y[i];
         x[2] = inter_cl->z[i];
         CalcCartesianCoords(&x[0]);
         if (fp_cl)  fprintf(fp_cl, "%f  %f  %f\n", x[0], x[1], x[2]);
         //AddVPoint(ar->me[i+1]->cl, x);
         if (fp && j == check_index)
         {
            fprintf(fp, "%f  %f  %f\n", x[0], x[1], x[2]);
         }
      }
      x[0] = y[0] = ar->mshroud->cl_int->x[j];
      x[1] = y[1] = ar->mshroud->cl_int->y[j];
      x[2] = y[2] = ar->mshroud->cl_int->z[j];
      if (fp_cl)  fprintf(fp_cl, "%f  %f  %f\n", y[0], y[1], y[2]);
      if (fp_cl)  fprintf(fp_cl, "\n\n");
      //AddVPoint(ar->me[ar->be_num-1]->cl, y);
      if (fp && j == check_index)
      {
         fprintf(fp, "%f  %f  %f\n", y[0], y[1], y[2]);
         fprintf(fp, "\n\n");
      }

      if (fp_ps)  fprintf(fp_ps, "\n\n");
      if (fp_ps)  fprintf(fp_ps, "# entire intersection (index %d)\n", j);
      x[0] = y[0] = ar->mhub->ps_int->x[j];
      x[1] = y[1] = ar->mhub->ps_int->y[j];
      x[2] = y[2] = ar->mhub->ps_int->z[j];
      if (fp_ps)  fprintf(fp_ps, "%f  %f  %f\n", y[0], y[1], y[2]);
      for (i = 0; i < inter_ps->nump; i++)
      {
         x[0] = inter_ps->x[i];
         x[1] = inter_ps->y[i];
         x[2] = inter_ps->z[i];
         CalcCartesianCoords(&x[0]);
         if (fp_ps)  fprintf(fp_ps, "%f  %f  %f\n", x[0], x[1], x[2]);
      }
      x[0] = y[0] = ar->mshroud->ps_int->x[j];
      x[1] = y[1] = ar->mshroud->ps_int->y[j];
      x[2] = y[2] = ar->mshroud->ps_int->z[j];
      if (fp_ps)  fprintf(fp_ps, "%f  %f  %f\n", y[0], y[1], y[2]);
      if (fp_ps)  fprintf(fp_ps, "\n\n");

      if (fp_ss)  fprintf(fp_ss, "\n\n");
      if (fp_ss)  fprintf(fp_ss, "# entire intersection (index %d)\n", j);
      x[0] = y[0] = ar->mhub->ss_int->x[j];
      x[1] = y[1] = ar->mhub->ss_int->y[j];
      x[2] = y[2] = ar->mhub->ss_int->z[j];
      if (fp_ss)  fprintf(fp_ss, "%f  %f  %f\n", y[0], y[1], y[2]);
      for (i = 0; i < inter_ss->nump; i++)
      {
         x[0] = inter_ss->x[i];
         x[1] = inter_ss->y[i];
         x[2] = inter_ss->z[i];
         CalcCartesianCoords(&x[0]);
         if (fp_ss)  fprintf(fp_ss, "%f  %f  %f    %f  %f  %f\n", x[0], x[1], x[2], inter_ss->x[i],inter_ss->y[i],inter_ss->z[i]);
      }
      x[0] = y[0] = ar->mshroud->ss_int->x[j];
      x[1] = y[1] = ar->mshroud->ss_int->y[j];
      x[2] = y[2] = ar->mshroud->ss_int->z[j];
      if (fp_ss)  fprintf(fp_ss, "%f  %f  %f\n", y[0], y[1], y[2]);
      if (fp_ss)  fprintf(fp_ss, "\n\n");

      // assign intersection points to meridional description
      // centre line
      x[0] = ar->mhub->cl_int->x[j];
      x[1] = ar->mhub->cl_int->y[j];
      x[2] = ar->mhub->cl_int->z[j];
      CalcCylindricalCoords(x);
      AddVPoint(ar->me[0]->cl, x);
      for (i = 0; i < inter_cl->nump; i++)
      {
         x[0] = inter_cl->x[i];
         x[1] = inter_cl->y[i];
         x[2] = inter_cl->z[i];
         AddVPoint(ar->me[i+1]->cl, x);
      }
      x[0] = ar->mshroud->cl_int->x[j];
      x[1] = ar->mshroud->cl_int->y[j];
      x[2] = ar->mshroud->cl_int->z[j];
      CalcCylindricalCoords(x);
      AddVPoint(ar->me[ar->be_num-1]->cl, x);
      // pressure side
      x[0] = ar->mhub->ps_int->x[j];
      x[1] = ar->mhub->ps_int->y[j];
      x[2] = ar->mhub->ps_int->z[j];
      CalcCylindricalCoords(x);
      AddVPoint(ar->me[0]->ps, x);
      for (i = 0; i < inter_ps->nump; i++)
      {
         x[0] = inter_ps->x[i];
         x[1] = inter_ps->y[i];
         x[2] = inter_ps->z[i];
         AddVPoint(ar->me[i+1]->ps, x);
      }
      x[0] = ar->mshroud->ps_int->x[j];
      x[1] = ar->mshroud->ps_int->y[j];
      x[2] = ar->mshroud->ps_int->z[j];
      CalcCylindricalCoords(x);
      AddVPoint(ar->me[ar->be_num-1]->ps, x);
      // suction side
      x[0] = ar->mhub->ss_int->x[j];
      x[1] = ar->mhub->ss_int->y[j];
      x[2] = ar->mhub->ss_int->z[j];
      CalcCylindricalCoords(x);
      AddVPoint(ar->me[0]->ss, x);
      for (i = 0; i < inter_ss->nump; i++)
      {
         x[0] = inter_ss->x[i];
         x[1] = inter_ss->y[i];
         x[2] = inter_ss->z[i];
         AddVPoint(ar->me[i+1]->ss, x);
      }
      x[0] = ar->mshroud->ss_int->x[j];
      x[1] = ar->mshroud->ss_int->y[j];
      x[2] = ar->mshroud->ss_int->z[j];
      CalcCylindricalCoords(x);
      AddVPoint(ar->me[ar->be_num-1]->ss, x);
   }
   if (fp_cl)  fclose(fp_cl);
   if (fp_ps)  fclose(fp_ps);
   if (fp_ss)  fclose(fp_ss);
   if (fp)  fprintf(fp, "# blade line %d at end of UpdateAR_Meridians()\n", check_index);
   for (i = 0; fp && i < ar->be_num; i++)
   {
      fprintf(fp, "%f  ", ar->be[i]->cl_cart->x[check_index]);
      fprintf(fp, "%f  ", ar->be[i]->cl_cart->y[check_index]);
      fprintf(fp, "%f\n", ar->be[i]->cl_cart->z[check_index]);
   }
   if (fp)
   {
      fprintf(fp, "\n\n");
      fclose(fp);
   }
   fp=NULL;
   fn = DebugFilename("ar_meridian_surfaces.txt");
   if(fn)
   fp = fopen(fn, "w");

   if (fp)  fprintf(fp, "# centre line (cyl. coords)\n");
   for (i = 0; i < ar->be_num; i++)
   {
      for (j = 0; j < ar->me[i]->cl->nump; j++)
      {
         x[0] = ar->me[i]->cl->x[j];
         x[1] = ar->me[i]->cl->y[j];
         x[2] = ar->me[i]->cl->z[j];
         if (fp)  fprintf(fp, "%f %f %f\n", x[0], x[1], x[2]);
      }
      if (fp)  fprintf(fp, "\n");
   }
   if (fp)  fprintf(fp, "\n\n");

   if (fp)  fprintf(fp, "# pressure side (cyl. coords)\n");
   for (i = 0; i < ar->be_num; i++)
   {
      for (j = 0; j < ar->me[i]->ps->nump; j++)
      {
         x[0] = ar->me[i]->ps->x[j];
         x[1] = ar->me[i]->ps->y[j];
         x[2] = ar->me[i]->ps->z[j];
         if (fp)  fprintf(fp, "%f %f %f\n", x[0], x[1], x[2]);
      }
      if (fp)  fprintf(fp, "\n");
   }
   if (fp)  fprintf(fp, "\n\n");

   if (fp)  fprintf(fp, "# suction side (cyl. coords)\n");
   for (i = 0; i < ar->be_num; i++)
   {
      for (j = 0; j < ar->me[i]->ss->nump; j++)
      {
         x[0] = ar->me[i]->ss->x[j];
         x[1] = ar->me[i]->ss->y[j];
         x[2] = ar->me[i]->ss->z[j];
         if (fp)  fprintf(fp, "%f %f %f\n", x[0], x[1], x[2]);
      }
      if (fp)  fprintf(fp, "\n");
   }
   if (fp)  fclose(fp);

   fp=NULL;
   fn = DebugFilename("ar_meridian_surfaces_swap.txt");
   if(fn)
   fp = fopen(fn, "w");

   if (fp)
   {
      fprintf(fp, "# centre line (cyl. coords)\n");
      for (i = 0; i < ar->be_num; i++)
      {
         for (j = 0; j < ar->me[i]->cl->nump; j++)
         {
            x[0] = ar->me[i]->cl->x[j];
            x[1] = ar->me[i]->cl->y[j];
            x[2] = ar->me[i]->cl->z[j];
            if(j < ar->me[i]->cl->nump-1)
               alpha = atan((x[2]-ar->me[i]->cl->z[j+1])/
                  (x[0]*x[1]-ar->me[i]->cl->x[j+1]*
                  ar->me[i]->cl->y[j+1]));
            fprintf(fp, "%f %f %f   %f\n", x[0], x[1], x[2],
               alpha*180.0/M_PI);
         }
         if (fp)  fprintf(fp, "\n");
      }
      fprintf(fp, "\n\n");

      fprintf(fp, "# pressure side (cyl. coords)\n");
      for (i = 0; i < ar->be_num; i++)
      {
         for (j = 0; j < ar->me[i]->ps->nump; j++)
         {
            x[0] = ar->me[i]->ps->x[j];
            x[1] = ar->me[i]->ps->y[j];
            x[2] = ar->me[i]->ps->z[j];
            fprintf(fp, "%f %f %f\n", x[0], x[1], x[2]);
         }
         fprintf(fp, "\n");
      }
      fprintf(fp, "\n\n");

      fprintf(fp, "# suction side (cyl. coords)\n");
      for (i = 0; i < ar->be_num; i++)
      {
         for (j = 0; j < ar->me[i]->ss->nump; j++)
         {
            x[0] = ar->me[i]->ss->x[j];
            x[1] = ar->me[i]->ss->y[j];
            x[2] = ar->me[i]->ss->z[j];
            fprintf(fp, "%f %f %f\n", x[0], x[1], x[2]);
         }
         fprintf(fp, "\n");
      }
      fclose(fp);
   }

   return err;
}


// FL: not needed anymore! GridGenerator works with r phi z coords.
int SwapAR_MeridianCoords(struct meridian *me)
// caller: UpdateAR_Meridians()
// swap blade and ml coordinates for grid generation
{
   int j;
   float tmp;

   // swap meridional line coordinates:
   // (r, 0.0, z) -> (r, z, 0.0)
   for (j = 0; j < me->ml->p->nump; j++)
   {
      tmp             = me->ml->p->y[j];
      me->ml->p->y[j] = me->ml->p->z[j];
      me->ml->p->z[j] = tmp;
   }
   // swap blade coordinates:
   // (r, phi, z) -> (phi, z, r)
   for (j = 0; j < me->cl->nump; j++)
   {
      tmp          = me->cl->x[j];
      me->cl->x[j] = me->cl->y[j];
      me->cl->y[j] = me->cl->z[j];
      me->cl->z[j] = tmp;
      tmp          = me->ps->x[j];
      me->ps->x[j] = me->ps->y[j];
      me->ps->y[j] = me->ps->z[j];
      me->ps->z[j] = tmp;
      tmp          = me->ss->x[j];
      me->ss->x[j] = me->ss->y[j];
      me->ss->y[j] = me->ss->z[j];
      me->ss->z[j] = tmp;
   }
   return 0;
}
