#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef WIN32
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#endif
#include <math.h>
#include <assert.h>
#include "../General/include/geo.h"
#include "../General/include/log.h"
#include "../General/include/cov.h"
#include "../General/include/points.h"
#include "../General/include/curve.h"
#include "../General/include/profile.h"
#include "../General/include/common.h"
#include "../General/include/fatal.h"
#include "../General/include/v.h"
#include "include/axial.h"
#include "include/ar2cov.h"
#include "include/ar_arclenbe.h"
#include "include/ar_blademer.h"
#include "include/ar_condarea.h"
#include "include/ar_contours.h"
#include "include/ar_createbe.h"
#include "include/ar_initbe.h"
#include "include/ar_intersect.h"
#include "include/ar_meridiancont.h"
#include "include/ar_updatemer.h"
#include "include/ar_angles.h"

#define NUMBER_OF_SECTIONS 120
#define TOLERANCE       1.0e-6
#define CHAR_LEN               200
#define PLOT_SCALE_ENLARGE     0.05

static int last_err = 0;
const char *err_msg[] =
{
   NULL,"Design data missing!",
   "Could not create blades. Hub is too short!",
   "Angle at trailing edge > leading edge or le_angle > 90 degrees!",
   "Inlet diametre must be bigger than runner inlet diam.!",
   "Inlet height mut be bigger than zero!",
   "No blade-meridian-intersection point found!",
   "Could not write catesian blade data!"
};

static struct Point *GetConformalView(struct Point *src, float l0);
static int AddLine2PlotVector(struct Point *line, float *xpl, float *ypl,
float *xy, int *pcount);
static void FitPSSSCurves(struct Point *cl, struct Point *ps,struct Point *ss);
static float myatan(float x, float y);

struct covise_info *Axial2Covise(struct axial *ar)
{
   struct covise_info *ci = NULL;
   int i, j, jmax_ps, jmax_ss, err = 0;
   int offs, nose, te;
   float x, y, z;
   FILE *ferr;
   char fname[255];
   char *fn;
   static int fcount = 0;

   // create AxialRunner geometry first
   ModifyAR_BladeElements4Covise(ar);
   if ((err = CreateAR_Contours(ar)))
   {
      last_err = err;
      return NULL;
   }
   dprintf(2, "CreateAR_Contours()...         DONE\n");
   CreateAR_MeridianContours(ar);
   dprintf(2, "CreateAR_MeridianContours()... DONE\n");
   CreateAR_ConduitAreas(ar);
   dprintf(2, "CreateAR_ConduitAreas()...     DONE\n");
   ArclenAR_BladeElements(ar);
   dprintf(2, "ArclenAR_BladeElements()...    DONE\n");
   ar->des->spec_revs = ar->des->revs*
	   sqrt(ar->des->dis)/pow(ar->des->head,0.75f);
   if(ar->euler)
   {
      if((err = CalcAR_BladeAngles(ar)))
      {
         last_err = err;
         return NULL;
      }
   }
#ifdef PLOT_BLADE_EDGES
   PlotAR_BladeEdges(ar);
#endif                                         // PLOT_BLADE_EDGES

   // memory for polygon data, delete previous data
   if (!(err = CreateAR_BladeElements(ar)))
   {
      if((err = UpdateAR_Meridians(ar)))
      {
         last_err = err;
         return NULL;
      }
      dprintf(2, "CreateAR_BladeElements()...     DONE\n");
      if ((ci = AllocCoviseInfo(ar->be_num)) != NULL)
      {
         // points to copy, distinguish cases: te_thick
         if (ar->be[0]->te_thick > TOLERANCE)
         {
            jmax_ps = ar->be[0]->ps_cart->nump - 1;
            jmax_ss = jmax_ps;
            offs    = 2 * ar->be[0]->ps_cart->nump - 1;
            nose    = ar->be[0]->ps_cart->nump - 1;
            te      = 1;
         }
         else
         {
            jmax_ps = ar->be[0]->ps_cart->nump - 1;
            jmax_ss = jmax_ps;
            offs    = 2 * ar->be[0]->ps_cart->nump - 1;
            nose    = ar->be[0]->ps_cart->nump - 1;
            te      = 0;
         }
         // assign points to global array from blade elements:
         // inner blade margin (hub)
         assert(ar->mhub);
         assert(ar->mhub->ps_int);
         assert(ar->mhub->ps_int->x);
         assert(ar->mhub->ss_int);
         for (j = jmax_ps; j >= 0; j--)
         {
            x = ar->mhub->ps_int->x[j];
            y = ar->mhub->ps_int->y[j];
            z = ar->mhub->ps_int->z[j];
            AddPoint(ci->p, x, y, z);
         }
         for (j = 1; j <= jmax_ss; j++)
         {
            x = ar->mhub->ss_int->x[j];
            y = ar->mhub->ss_int->y[j];
            z = ar->mhub->ss_int->z[j];
            AddPoint(ci->p, x, y, z);
         }
         // inner blade elements
         for (i = 1; i < ar->be_num-1; i++)
         {
            for (j = jmax_ps; j >= 0; j--)
            {
               x = ar->be[i]->ps_cart->x[j];
               y = ar->be[i]->ps_cart->y[j];
               z = ar->be[i]->ps_cart->z[j];
               AddPoint(ci->p, x, y, z);
            }
            for (j = 1; j <= jmax_ss; j++)
            {
               x = ar->be[i]->ss_cart->x[j];
               y = ar->be[i]->ss_cart->y[j];
               z = ar->be[i]->ss_cart->z[j];
               AddPoint(ci->p, x, y, z);
            }
            CreateAR_BEPolygons(ci, i, offs, te);
         }
         // outer blade margin (shroud)
         for (j = jmax_ps; j >= 0; j--)
         {
            x = ar->mshroud->ps_int->x[j];
            y = ar->mshroud->ps_int->y[j];
            z = ar->mshroud->ps_int->z[j];
            AddPoint(ci->p, x, y, z);
         }
         for (j = 1; j <= jmax_ss; j++)
         {
            x = ar->mshroud->ss_int->x[j];
            y = ar->mshroud->ss_int->y[j];
            z = ar->mshroud->ss_int->z[j];
            AddPoint(ci->p, x, y, z);
         }
         CreateAR_BEPolygons(ci, ar->be_num-1, offs, te);
         CreateAR_TipPolygons(ci, nose, te);
      }
      // generate all blades of the runner
      dprintf(2, "rotating blades\n");
      RotateBlade4Covise(ci, ar->nob);
      // runner hub and shroud
      dprintf(2, "creating contour polygons\n");
      CreateAR_CoviseContours(ci, ar);
   }
   else
   {
      dprintf(0, "ERROR from CreateAR_BladeElements() in routine Axial2Covise()\n");
      last_err = err;
      return ci = NULL;
   }

   sprintf(fname, "ar_polygons_%02d.txt", fcount++);
   fn = DebugFilename(fname);
   if (fn && *fn && (ferr = fopen(fn, "w")) != NULL)
   {
      fprintf(ferr, "\nblade polygon vertices:\n");
      fprintf(ferr, "ci->vx->num = %d\t, ci->vx->max = %d\n", ci->vx->num, ci->vx->max);
      j = 1;
      fprintf(ferr, "%3d: ", j++);
      for (i = 0; i < ci->vx->num; i++)
      {
         fprintf(ferr, "vx[%3d] = %3d   ", i, ci->vx->list[i]);
         if (!((i+1)%3)) fprintf(ferr, "\n%3d: ", j++);
      }
      fprintf(ferr, "\nindices of blade polygon start vertices:\n");
      fprintf(ferr, "ci->pol->num = %d\t, ci->pol->max = %d\n", ci->pol->num, ci->pol->max);
      for (i = 0; i < ci->pol->num; i++)
      {
         fprintf(ferr, "pol[%3d] = %3d  ", i, ci->pol->list[i]);
         if (!((i+1)%4)) fprintf(ferr, "\n");
      }
      fprintf(ferr, "\nhub polygon vertices:\n");
      fprintf(ferr, "ci->cvx->num = %d\t, ci->cvx->max = %d\n", ci->cvx->num, ci->cvx->max);
      j = 1;
      fprintf(ferr, "%4d: ", j++);
      for (i = 0; i < ci->cvx->num; i++)
      {
         fprintf(ferr, "cvx[%4d] = %4d   ", i, ci->cvx->list[i]);
         if (!((i+1)%3)) fprintf(ferr, "\n%3d: ", j++);
      }
      fprintf(ferr, "\nindices of hub polygon start vertices:\n");
      fprintf(ferr, "ci->cpol->num = %d\t, ci->cpol->max = %d\n", ci->cpol->num, ci->cpol->max);
      for (i = 0; i < ci->cpol->num; i++)
      {
         fprintf(ferr, "cpol[%4d] = %4d  ", i, ci->cpol->list[i]);
         if (!((i+1)%4)) fprintf(ferr, "\n");
      }
      fprintf(ferr, "\ncoordinates of all vertices:\n");
      fprintf(ferr, "ci->p->nump = %d\t, ci->p->maxp = %d\n", ci->p->nump, ci->p->maxp);
      for (i = 0; i < ci->p->nump; i++)
      {
         fprintf(ferr, "p[%4d].x = %7.3f\t", i, ci->p->x[i]);
         fprintf(ferr, "p[%4d].y = %7.3f\t", i, ci->p->y[i]);
         fprintf(ferr, "p[%4d].z = %7.3f\n", i, ci->p->z[i]);
      }
      fclose(ferr);
   }
   return ci;
}


char *GetLastErr(void)
{
   return (char *)err_msg[last_err];
}


void CreateAR_BEPolygons(struct covise_info *ci, int be, int offs, int te)
{
   int i, ivx[3];
   static int ipol;

   if (be == 1) ipol = 0;
   // surface polygons
   for (i = 0; i < offs-1; i++)
   {
      // 1st polygon
      ivx[0] = (be - 1) * offs + i;
      ivx[1] = ivx[0] + offs;
      ivx[2] = ivx[1] + 1;
      Add2Ilist(ci->pol, ipol);
      Add2Ilist(ci->vx, ivx[0]);
      Add2Ilist(ci->vx, ivx[1]);
      Add2Ilist(ci->vx, ivx[2]);
      ipol += 3;
      // 2nd polygon
      ivx[0] = be * offs + 1 + i;
      ivx[1] = ivx[0] - offs;
      ivx[2] = ivx[1] - 1;
      Add2Ilist(ci->pol, ipol);
      Add2Ilist(ci->vx, ivx[0]);
      Add2Ilist(ci->vx, ivx[1]);
      Add2Ilist(ci->vx, ivx[2]);
      ipol += 3;
   }
   // case: te_thick > 0
   if (te)
   {
      // trailing edge polygons
      // 1st polygon
      ivx[0] = be * offs - 1;
      ivx[1] = ivx[0] + offs;
      ivx[2] = ivx[0] + 1;
      Add2Ilist(ci->pol, ipol);
      Add2Ilist(ci->vx, ivx[0]);
      Add2Ilist(ci->vx, ivx[1]);
      Add2Ilist(ci->vx, ivx[2]);
      ipol += 3;
      // 2nd polygon
      ivx[0] = be * offs;
      ivx[1] = ivx[0] - offs;
      ivx[2] = ivx[0] - 1;
      Add2Ilist(ci->pol, ipol);
      Add2Ilist(ci->vx, ivx[0]);
      Add2Ilist(ci->vx, ivx[1]);
      Add2Ilist(ci->vx, ivx[2]);
      ipol += 3;
   }
}


void CreateAR_TipPolygons(struct covise_info *ci, int npoin, int te)
{
   int i, ivx[3];
   int ipol = ci->vx->num;
   const int np = ci->p->nump - 2 * npoin - 1;

   // nose polygon
   ivx[0] = np + npoin + 1;
   ivx[1] = ivx[0] - 1;
   ivx[2] = ivx[1] - 1;
   Add2Ilist(ci->pol, ipol);
   Add2Ilist(ci->vx, ivx[0]);
   Add2Ilist(ci->vx, ivx[1]);
   Add2Ilist(ci->vx, ivx[2]);
   ipol += 3;
   for (i = 1; i < npoin-1; i++)
   {
      // 1st polygon
      ivx[0] = np + npoin + 1 + i;
      ivx[1] = ivx[0] - 1;
      ivx[2] = np + npoin - i;
      Add2Ilist(ci->pol, ipol);
      Add2Ilist(ci->vx, ivx[0]);
      Add2Ilist(ci->vx, ivx[1]);
      Add2Ilist(ci->vx, ivx[2]);
      ipol += 3;
      // 2nd polygon
      ivx[0] = np + npoin - i;
      ivx[1] = ivx[0] - 1;
      ivx[2] = np + npoin + 1 + i;
      Add2Ilist(ci->pol, ipol);
      Add2Ilist(ci->vx, ivx[0]);
      Add2Ilist(ci->vx, ivx[1]);
      Add2Ilist(ci->vx, ivx[2]);
      ipol += 3;
   }
   // te polygon(s), case: (te)
   if (te)
   {
      // 1st polygon
      ivx[0] = np + 2 * npoin;
      ivx[1] = ivx[0] - 1;
      ivx[2] = np + 1;
      Add2Ilist(ci->pol, ipol);
      Add2Ilist(ci->vx, ivx[0]);
      Add2Ilist(ci->vx, ivx[1]);
      Add2Ilist(ci->vx, ivx[2]);
      ipol += 3;
      // 2nd polygon
      ivx[0] = np + 1;
      ivx[1] = np;
      ivx[2] = np + 2 * npoin;
      Add2Ilist(ci->pol, ipol);
      Add2Ilist(ci->vx, ivx[0]);
      Add2Ilist(ci->vx, ivx[1]);
      Add2Ilist(ci->vx, ivx[2]);
      ipol += 3;
   }
   else                                           // te polygon, case (!te)
   {
      ivx[0] = np + 1;
      ivx[1] = np;
      ivx[2] = np + 2 * npoin - 1;
      Add2Ilist(ci->pol, ipol);
      Add2Ilist(ci->vx, ivx[0]);
      Add2Ilist(ci->vx, ivx[1]);
      Add2Ilist(ci->vx, ivx[2]);
      ipol += 3;
   }
}


void RotateBlade4Covise(struct covise_info *ci, int nob)
{
   int i, j, ipol, ivx;
   int np, npol, nvx;
   float rot, roma[2][2];
   float x, y, z;

   np         = ci->p->nump;
   npol       = ci->pol->num;
   nvx        = ci->vx->num;
   rot        = 2 * M_PI / nob;
   roma[0][0] =  cos(rot);
   roma[0][1] = -sin(rot);
   roma[1][0] =  sin(rot);
   roma[1][1] =  cos(rot);

   for (i = 0; i < nob-1; i++)
   {
      // calculate rotated blade point coordinates
      for (j = i*np; j < (i+1)*np; j++)
      {
         x = ci->p->x[j] * roma[0][0] + ci->p->y[j] * roma[0][1];
         y = ci->p->x[j] * roma[1][0] + ci->p->y[j] * roma[1][1];
         z = ci->p->z[j];
         AddPoint(ci->p, x, y, z);
      }
      // assign rotated polygon vertices
      for (j = i*nvx; j < (i+1)*nvx; j++)
      {
         ivx = ci->vx->list[j] + np;
         Add2Ilist(ci->vx, ivx);
      }
      // assign rotated polygon start vertices
      ipol = ci->pol->list[ci->pol->num-1];
      for (j = 0; j < npol; j++)
      {
         ipol  += 3;
         Add2Ilist(ci->pol, ipol);
      }
   }
}


void CreateAR_CoviseContours(struct covise_info *ci, struct axial *ar)
{
   int i, j, ind, hub;
   int c_nump, nphub;
   float x, y, z;
   float angle, roma[2][2];
   const int npblade  = ci->p->nump;
   const float rot = 2 * (float) M_PI / NUMBER_OF_SECTIONS;

   // 1st point on hub contour curve
   c_nump = ar->me[0]->ml->p->nump - 1;
   if (ar->mod->outl)
   {
      c_nump -= NPOIN_SPLN_OUTLET;
   }
   // add hub cap middle point coords to global array
   x = 0.0;
   y = 0.0;
   z = ar->me[0]->ml->p->z[c_nump];
   AddPoint(ci->p, x, y, z);
   // append hub contour point coords, reverse order
   for (i = c_nump; i >= 0; i--)
      AddPoint(ci->p, ar->me[0]->ml->p->x[i], ar->me[0]->ml->p->y[i], ar->me[0]->ml->p->z[i]);
   // rotate hub contour and append points
   for (i = 1; i < NUMBER_OF_SECTIONS; i++)
   {
      angle      = i * rot;
      roma[0][0] =  cos(angle);
      roma[0][1] = -sin(angle);
      roma[1][0] =  sin(angle);
      roma[1][1] =  cos(angle);
      for (j = 0; j < c_nump+2; j++)
      {
         ind = npblade + j;
         x   = ci->p->x[ind] * roma[0][0] + ci->p->y[ind] * roma[0][1];
         y   = ci->p->x[ind] * roma[1][0] + ci->p->y[ind] * roma[1][1];
         z   = ci->p->z[ind];
         AddPoint(ci->p, x, y, z);
      }
   }
   // create hub polygons
   hub = 1;
   for (i = 1; i <= NUMBER_OF_SECTIONS; i++)
      CreateContourPolygons(ci->lpol, ci->lvx, i, (c_nump+2), npblade, hub);

   // append shroud contour point coordinates to global array
   nphub = ci->p->nump - npblade;

   c_nump = ar->me[ar->be_num-1]->ml->p->nump - 1;
   for (i = c_nump; i >= 0; i--)
      AddPoint(ci->p, ar->me[ar->be_num-1]->ml->p->x[i], ar->me[ar->be_num-1]->ml->p->y[i], ar->me[ar->be_num-1]->ml->p->z[i]);
   // rotate shroud contour and append points
   for (i = 1; i < NUMBER_OF_SECTIONS; i++)
   {
      angle      = i * rot;
      roma[0][0] =  cos(angle);
      roma[0][1] = -sin(angle);
      roma[1][0] =  sin(angle);
      roma[1][1] =  cos(angle);
      for (j = 0; j < c_nump+1; j++)
      {
         ind = npblade + nphub + j;
         x   = ci->p->x[ind] * roma[0][0] + ci->p->y[ind] * roma[0][1];
         y   = ci->p->x[ind] * roma[1][0] + ci->p->y[ind] * roma[1][1];
         z   = ci->p->z[ind];
         AddPoint(ci->p, x, y, z);
      }
   }
   // create shroud polygons
   hub = 0;
   for (i = 1; i <= NUMBER_OF_SECTIONS; i++)
      CreateContourPolygons(ci->cpol, ci->cvx, i, (c_nump+1), (npblade+nphub), hub);
}


void CreateContourPolygons(struct Ilist *ci_pol, struct Ilist *ci_vx, int sec, int offs, int np, int hub)
{
   int i;
   int vx[3];
   static int ipol;

   if (sec == 1) ipol = 0;
   if (sec < NUMBER_OF_SECTIONS)
   {
      if (hub)                                    // single hub cap polygon
      {
         vx[0]= np + sec * offs;
         vx[1]= vx[0] + 1;
         vx[2]= vx[1] - offs;
         Add2Ilist(ci_vx, vx[0]);
         Add2Ilist(ci_vx, vx[1]);
         Add2Ilist(ci_vx, vx[2]);
         Add2Ilist(ci_pol, ipol);
         ipol += 3;
      }
      for (i = 1; i < offs-1; i++)
      {
         // 1st polygon
         vx[0]= np + (sec - 1) * offs + i;
         vx[1]= vx[0] + offs;
         vx[2]= vx[1] + 1;
         Add2Ilist(ci_vx, vx[0]);
         Add2Ilist(ci_vx, vx[1]);
         Add2Ilist(ci_vx, vx[2]);
         Add2Ilist(ci_pol, ipol);
         ipol += 3;
         // 2nd polygon
         vx[0]= np + sec * offs + 1 + i;
         vx[1]= vx[0] - offs;
         vx[2]= vx[1] - 1;
         Add2Ilist(ci_vx, vx[0]);
         Add2Ilist(ci_vx, vx[1]);
         Add2Ilist(ci_vx, vx[2]);
         Add2Ilist(ci_pol, ipol);
         ipol += 3;
      }
   }
   else                                           // (sec == NUMBER_OF_SECTIONS)
   {
      if (hub)                                    // single hub cap polygon
      {
         vx[0]= np;
         vx[1]= np + 1;
         vx[2]= np + (sec - 1) * offs + 1;
         Add2Ilist(ci_vx, vx[0]);
         Add2Ilist(ci_vx, vx[1]);
         Add2Ilist(ci_vx, vx[2]);
         Add2Ilist(ci_pol, ipol);
         ipol += 3;
      }
      for (i = 1; i < offs-1; i++)
      {
         // 1st polygon
         vx[0]= np + (sec - 1) * offs + i;
         vx[1]= np + i;
         vx[2]= vx[1] + 1;
         Add2Ilist(ci_vx, vx[0]);
         Add2Ilist(ci_vx, vx[1]);
         Add2Ilist(ci_vx, vx[2]);
         Add2Ilist(ci_pol, ipol);
         ipol += 3;
         // 2nd polygon
         vx[0]= np + 1 + i;
         vx[1]= np + (sec - 1) * offs + 1 + i;
         vx[2]= vx[1] - 1;
         Add2Ilist(ci_vx, vx[0]);
         Add2Ilist(ci_vx, vx[1]);
         Add2Ilist(ci_vx, vx[2]);
         Add2Ilist(ci_pol, ipol);
         ipol += 3;
      }
   }
}


int PutBladeData(struct axial *ar)
{
   int i, j, ibe_max;

   char fn[200];
   FILE *fp;

   // **************************************************
   // blades in cartesian coords.
   sprintf(fn,"blade.data");
   if( (fp = fopen(fn,"w+")) == NULL)
   {
      last_err = PUT_BLADEDATA_ERR;
      return 1;
   }
   fprintf(fp,"# blade data sorted by blade elements (%d)\n",
      ar->be_num);
   fprintf(fp,"# rated Q, H, n, z: %.3f, %.3f, %.3f, %d\n#\n",
      ar->des->dis, ar->des->head, ar->des->revs, ar->nob);
   fprintf(fp,"# x   y   z [m]\n");
   // ps
   fprintf(fp,"# pressure sides\n");
   fprintf(fp,"\n\n\n# ps %d/%d, %d points\n",1,ar->be_num,
      ar->mhub->ps_int->nump);
   for(j = 0; j < ar->mhub->ps_int->nump; j++)
      fprintf(fp,"%14.5f  %14.5f  %14.5f\n",
         ar->mhub->ps_int->x[j],ar->mhub->ps_int->y[j],
         ar->mhub->ps_int->z[j]);
   for(i = 1; i < ar->be_num-1; i++)
   {
      fprintf(fp,"\n\n# ps %d/%d, %d points\n",i+1,ar->be_num,
         ar->be[i]->ps_cart->nump);
      for(j = 0; j < ar->be[i]->ps_cart->nump; j++)
         fprintf(fp,"%14.5f  %14.5f  %14.5f\n",
            ar->be[i]->ps_cart->x[j], ar->be[i]->ps_cart->y[j],
            ar->be[i]->ps_cart->z[j]);
   }
   fprintf(fp,"\n\n# ps %d/%d, %d points\n",ar->be_num,ar->be_num,
      ar->mshroud->ps_int->nump);
   for(j = 0; j < ar->mshroud->ps_int->nump; j++)
      fprintf(fp,"%14.5f  %14.5f  %14.5f\n",
         ar->mshroud->ps_int->x[j],ar->mshroud->ps_int->y[j],
         ar->mshroud->ps_int->z[j]);
   // ss
   fprintf(fp,"# suction sides\n");
   fprintf(fp,"\n\n\n# ss %d/%d, %d points\n",1,ar->be_num,
      ar->mhub->ss_int->nump);
   for(j = 0; j < ar->mhub->ss_int->nump; j++)
      fprintf(fp,"%14.5f  %14.5f  %14.5f\n",
         ar->mhub->ss_int->x[j],ar->mhub->ss_int->y[j],
         ar->mhub->ss_int->z[j]);
   for(i = 1; i < ar->be_num-1; i++)
   {
      fprintf(fp,"\n\n# ss %d/%d, %d points\n",i+1,ar->be_num,
         ar->be[i]->ss_cart->nump);
      for(j = 0; j < ar->be[i]->ss_cart->nump; j++)
         fprintf(fp,"%14.5f  %14.5f  %14.5f\n",
            ar->be[i]->ss_cart->x[j], ar->be[i]->ss_cart->y[j],
            ar->be[i]->ss_cart->z[j]);
   }
   fprintf(fp,"\n\n# ss %d/%d, %d points\n",ar->be_num,ar->be_num,
      ar->mshroud->ss_int->nump);
   for(j = 0; j < ar->mshroud->ss_int->nump; j++)
      fprintf(fp,"%14.5f  %14.5f  %14.5f\n",
         ar->mshroud->ss_int->x[j],ar->mshroud->ss_int->y[j],
         ar->mshroud->ss_int->z[j]);
   // center lines
   fprintf(fp,"# center lines\n");
   fprintf(fp,"\n\n\n# cl %d/%d, %d points\n",1,ar->be_num,
      ar->mhub->cl_int->nump);
   for(j = 0; j < ar->mhub->cl_int->nump; j++)
      fprintf(fp,"%14.5f  %14.5f  %14.5f\n",
         ar->mhub->cl_int->x[j],ar->mhub->cl_int->y[j],
         ar->mhub->cl_int->z[j]);
   for(i = 1; i < ar->be_num-1; i++)
   {
      fprintf(fp,"\n\n# cl %d/%d, %d points\n",i+1,ar->be_num,
         ar->be[i]->cl_cart->nump);
      for(j = 0; j < ar->be[i]->cl_cart->nump; j++)
         fprintf(fp,"%14.5f  %14.5f  %14.5f\n",
            ar->be[i]->cl_cart->x[j], ar->be[i]->cl_cart->y[j],
            ar->be[i]->cl_cart->z[j]);
   }
   fprintf(fp,"\n\n# cl %d/%d, %d points\n",ar->be_num,ar->be_num,
      ar->mshroud->cl_int->nump);
   for(j = 0; j < ar->mshroud->cl_int->nump; j++)
      fprintf(fp,"%14.5f  %14.5f  %14.5f\n",
         ar->mshroud->cl_int->x[j],ar->mshroud->cl_int->y[j],
         ar->mshroud->cl_int->z[j]);

   fclose(fp);
   // **************************************************
   // blades for proE.
   sprintf(fn,"blade.ibl");
   if( (fp = fopen(fn,"w+")) == NULL)
   {
      last_err = PUT_BLADEDATA_ERR;
      return 1;
   }
   fprintf(fp,"Closed Index Arclength\n");
   fprintf(fp,"begin section ! %3d\n",0);
   fprintf(fp,"begin curve\n");
   for(j = ar->mhub->ps_int->nump-1; j > 0; j--)
      fprintf(fp,"%14.7f  %14.7f  %14.7f\n",
         ar->mhub->ps_int->x[j],ar->mhub->ps_int->y[j],
         ar->mhub->ps_int->z[j]);
   for(j = 0; j < ar->mhub->ss_int->nump; j++)
      fprintf(fp,"%14.7f  %14.7f  %14.7f\n",
         ar->mhub->ss_int->x[j],ar->mhub->ss_int->y[j],
         ar->mhub->ss_int->z[j]);
   for(i = 1; i < ar->be_num-1; i++)
   {
      fprintf(fp,"begin section ! %3d\n",i);
      fprintf(fp,"begin curve\n");
      for(j = ar->be[i]->ps_cart->nump-1; j > 0; j--)
         fprintf(fp,"%14.7f  %14.7f  %14.7f\n",
            ar->be[i]->ps_cart->x[j],
            ar->be[i]->ps_cart->y[j],
            ar->be[i]->ps_cart->z[j]);
      for(j = 0; j < ar->be[i]->ss_cart->nump; j++)
         fprintf(fp,"%14.7f  %14.7f  %14.7f\n",
            ar->be[i]->ss_cart->x[j],
            ar->be[i]->ss_cart->y[j],
            ar->be[i]->ss_cart->z[j]);
   }
   fprintf(fp,"begin section ! %3d\n",ar->be_num-1);
   fprintf(fp,"begin curve\n");
   for(j = ar->mshroud->ps_int->nump-1; j > 0; j--)
      fprintf(fp,"%14.7f  %14.7f  %14.7f\n",
         ar->mshroud->ps_int->x[j],ar->mshroud->ps_int->y[j],
         ar->mshroud->ps_int->z[j]);
   for(j = 0; j < ar->mshroud->ss_int->nump; j++)
      fprintf(fp,"%14.7f  %14.7f  %14.7f\n",
         ar->mshroud->ss_int->x[j],ar->mshroud->ss_int->y[j],
         ar->mshroud->ss_int->z[j]);
   fclose(fp);
   // center line
   sprintf(fn,"centerline.ibl");
   if( (fp = fopen(fn,"w+")) == NULL)
   {
      last_err = PUT_BLADEDATA_ERR;
      return 1;
   }
   fprintf(fp,"Closed Index Arclength\n");
   fprintf(fp,"begin section ! %3d\n",0);
   fprintf(fp,"begin curve\n");
   for(j = 0; j < ar->mhub->cl_int->nump; j++)
      fprintf(fp,"%14.7f  %14.7f  %14.7f\n",
         ar->mhub->cl_int->x[j],ar->mhub->cl_int->y[j],
         ar->mhub->cl_int->z[j]);
   for(i = 1; i < ar->be_num-1; i++)
   {
      fprintf(fp,"begin section ! %3d\n",i);
      fprintf(fp,"begin curve\n");
      for(j = 0; j < ar->be[i]->cl_cart->nump; j++)
         fprintf(fp,"%14.7f  %14.7f  %14.7f\n",
            ar->be[i]->cl_cart->x[j], ar->be[i]->cl_cart->y[j],
            ar->be[i]->cl_cart->z[j]);
   }
   fprintf(fp,"begin section ! %3d\n",ar->be_num-1);
   fprintf(fp,"begin curve\n");
   for(j = 0; j < ar->mshroud->cl_int->nump; j++)
      fprintf(fp,"%14.7f  %14.7f  %14.7f\n",
         ar->mshroud->cl_int->x[j],ar->mshroud->cl_int->y[j],
         ar->mshroud->cl_int->z[j]);
   fclose(fp);


	// hub contour
	sprintf(fn,"hub.ibl");
	if( (fp = fopen(fn,"w+")) == NULL) {
		last_err = PUT_BLADEDATA_ERR;
		return 1;
	}
	fprintf(fp,"Closed Index Arclength\n");
	fprintf(fp,"begin section\n");
	fprintf(fp,"begin curve\n");
	for(i = 0; i < ar->me[0]->ml->p->nump; i++) {
		fprintf(fp,"%14.5f	%14.5f	%14.5f\n",
				ar->me[0]->ml->p->x[i], ar->me[0]->ml->p->y[i],
				ar->me[0]->ml->p->z[i]);
	}
	fclose(fp);

	// shroud contour
	ibe_max = ar->be_num-1;
	sprintf(fn,"shroud.ibl");
	if( (fp = fopen(fn,"w+")) == NULL) {
		last_err = PUT_BLADEDATA_ERR;
		return 1;
	}
	fprintf(fp,"Closed Index Arclength\n");
	fprintf(fp,"begin section\n");
	fprintf(fp,"begin curve\n");
	for(i = j; i < ar->me[ibe_max]->ml->p->nump; i++) {
		fprintf(fp,"%14.5f	%14.5f	%14.5f\n",
				ar->me[ibe_max]->ml->p->x[i],
				ar->me[ibe_max]->ml->p->y[i],
				ar->me[ibe_max]->ml->p->z[i]);
	}
	fclose(fp);

   return 0;
}


void GetMeridianContourNumbers(int *num_points, float *xy, struct axial *ar,
int /*ext_flag*/)
{
   int ml_firstp, ml_lastp, be_last;

   ml_firstp = 0;
   ml_lastp = ar->me[0]->ml->p->nump-1;
   be_last  = ar->be_num-1;

   *num_points = 4*(ar->me[0]->ml->p->nump-1)+4*(ar->be_num-1);
   xy[0]       = 0.0;
   xy[1]       = ar->me[be_last]->ml->p->z[ml_lastp]
      * (1.0 - (SIGN( ar->me[be_last]->ml->p->z[ml_lastp]))*
      PLOT_SCALE_ENLARGE);
   xy[2]       = ar->me[be_last]->ml->p->x[ml_firstp]
      * (1.0 + (SIGN( ar->me[be_last]->ml->p->x[ml_firstp]))*
      PLOT_SCALE_ENLARGE);
   xy[3]       = ar->me[0]->ml->p->z[ml_firstp]
      * (1.0 + (SIGN(ar->me[0]->ml->p->z[ml_firstp]))*
      PLOT_SCALE_ENLARGE);
}


void GetMeridianContourPlotData(struct axial *ar, float *xpl, float *ypl,
int /*ext_flag*/)
{
   int i, iend, j;
   int ibe_max;

   // meridian contour
   ibe_max = ar->be_num-1;
   iend    = ar->me[0]->ml->p->nump-1;

   j = 0;
   for(i = 0; i < iend; i++)
   {
      xpl[j] = ar->me[0]->ml->p->x[i];
      ypl[j] = ar->me[0]->ml->p->z[i];
      j++;
      xpl[j] = ar->me[0]->ml->p->x[i+1];
      ypl[j] = ar->me[0]->ml->p->z[i+1];
      j++;
   }
   for(i = 0; i < iend; i++)
   {
      xpl[j] = ar->me[ibe_max]->ml->p->x[i];
      ypl[j] = ar->me[ibe_max]->ml->p->z[i];
      j++;
      xpl[j] = ar->me[ibe_max]->ml->p->x[i+1];
      ypl[j] = ar->me[ibe_max]->ml->p->z[i+1];
      j++;
   }

   // blade edges
   for(i = 0; i < ibe_max; i++)
   {
      xpl[j] = ar->me[i]->cl->x[0];
      ypl[j] = ar->me[i]->cl->z[0];
      j++;
      xpl[j] = ar->me[i+1]->cl->x[0];
      ypl[j] = ar->me[i+1]->cl->z[0];
      j++;
   }
   iend = ar->me[0]->cl->nump-1;
   for(i = 0; i < ibe_max; i++)
   {
      xpl[j] = ar->me[i]->cl->x[iend];
      ypl[j] = ar->me[i]->cl->z[iend];
      j++;
      xpl[j] = ar->me[i+1]->cl->x[iend];
      ypl[j] = ar->me[i+1]->cl->z[iend];
      j++;
   }
}


void GetXMGRCommands(char *plbuf, float *xy, const char *title, const char *xlabel, const char *ylabel, int q_flag)
{
   int i, d, m;
   char buf[CHAR_LEN];

   float xy_min, xy_max, b, ytic, factor;

   // make boundaries even numbers
   for(i = 0; i < 4; i++)
   {
      if((b = xy[i]))
      {
         d = m = 0;
         while( ABS(b) > 10)
         {
            b /= 10;
            d++;
         }
         while( ABS(b) < 1)
         {
            b *= 10;
            m++;
         }
         if( ((i >= 2) && (((int)(b*10000))%((int)(b)*10000))!=0) ||
            (i < 2 && (b < 0)))
            b += 0.5*SIGN(b);                     // max-val. -> step up
         b *= 2;                                  // double to get 0.5,5,50 ... steps
         b  = (float)((int)(b))/2.0;
         if(d)
            b *= pow((float)10,(float)d);
         else if(m)
            b /= pow((float)10,(float)m);
         xy[i] = b;
      }
   }

   // quadratic plot scale if q_flag
   if(q_flag)
   {
      xy_min = MIN(xy[0], xy[1]);
      xy_max = MAX(xy[2], xy[3]);
      sprintf(buf, "WORLD %f,%f,%f,%f\n", xy_min, xy_min, xy_max, xy_max);
   }
   else
   {
      sprintf(buf, "WORLD %f,%f,%f,%f\n", xy[0], xy[1], xy[2], xy[3]);
   }
   sprintf(plbuf,"AUTOSCALE\n");
   strcat(plbuf, buf);
   strcat(plbuf, "SETS SYMBOL 27\n");
   strcat(plbuf, "SETS LINESTYLE 0\n");
   sprintf(buf,"title \"%s\"\n", title);
   strcat(plbuf, buf);
   sprintf(buf, "xaxis  label \"%s\"\n", xlabel);
   strcat(plbuf, buf);
   sprintf(buf, "yaxis  label \"%s\"\n", ylabel);
   strcat(plbuf, buf);
   factor = 10.0;
   ytic = (float)(((int)(((xy[3]-xy[1])*factor)))/(9.0*factor));
   while(ytic == 0.0)
   {
      factor *= 10.0;
      ytic = (float)(((int)(((xy[3]-xy[1])*factor)))/(9.0*factor));
   }
   sprintf(buf, "yaxis  tick major %f\nyaxis  tick minor %f\n",
      ytic,ytic/2.0);
   strcat(plbuf, buf);

}


void GetEulerAnglesPlotData(struct axial *ar, float *xpl,
float *ypl, float *xy)
{
   int i, j, pcount;

   pcount = 0;
   for(j = 0; j < 2; j++)
   {
      for(i = 1; i < ar->be_num; i++)
      {
         xpl[pcount] = ar->be[i-1]->para;
         ypl[pcount] = ar->be[i-1]->angle[j];
         pcount++;
         xpl[pcount] = ar->be[i]->para;
         ypl[pcount] = ar->be[i]->angle[j];
         pcount++;
         dprintf(5,"euler_angles: %f, %f\n",
            ar->be[i-1]->angle[j],ar->be[i]->angle[j]);
      }
   }
   // plot range
   xy[0]   =  0.0;
   xy[2]   =  1.0;
   xy[1]   = 0.0f;
   xy[3]   = 90.0f;
   if(ar->be[0]->angle[0] > xy[3])
   {
      xy[3]  = ar->be[0]->angle[0]+10.0f;
      xy[3] *= 0.1f;
      xy[3]  = (float)((int)xy[3])/0.1f;
   }
}


void GetConformalViewPlotData(struct axial *ar, float *xpl, float *ypl,
float *xy, int c, int v_count)
{
   static int pcount;
   int i;
   float l0;
   struct Point *cl,*ps, *ss;

   dprintf(3,"GetConformalViewPlotData(): ...\n");
   // start length coord. of cl.
   l0 = ar->me[c]->cl->z[ar->me[c]->cl->nump-1];
   cl = GetConformalView(ar->me[c]->cl, l0);
   ps = GetConformalView(ar->me[c]->ps, l0 + sqrt(pow(ar->me[c]->ps->x[ar->me[c]->ps->nump-1]
      -ar->me[c]->cl->x[ar->me[c]->ps->nump-1],2)
      + pow(ar->me[c]->ps->z[ar->me[c]->ps->nump-1]
      -ar->me[c]->cl->z[ar->me[c]->ps->nump-1],2)));
   ss = GetConformalView(ar->me[c]->ss, l0 - sqrt(pow(ar->me[c]->ss->x[ar->me[c]->ss->nump-1]
      -ar->me[c]->cl->x[ar->me[c]->ss->nump-1],2)
      + pow(ar->me[c]->ss->z[ar->me[c]->ss->nump-1]
      -ar->me[c]->cl->z[ar->me[c]->ss->nump-1],2)));
   FitPSSSCurves(cl,ps,ss);
   // create plot vectors
   if(!v_count)
   {
      pcount = 0;
      for(i = 0; i < 4; i++) xy[i] = 0.0;
   }
   AddLine2PlotVector(cl, xpl, ypl, xy, &pcount);
   AddLine2PlotVector(ps, xpl, ypl, xy, &pcount);
   AddLine2PlotVector(ss, xpl, ypl, xy, &pcount);

   dprintf(5,"GetConformalViewPlotData: pcount = %3d\n",pcount);

   if(xy[0] > ps->x[0]) xy[0] = ps->x[0];         // xmin
   if(xy[1] > ps->x[1]) xy[1] = ps->x[1];         // ymin

   if((xy[2]-xy[0]) > (xy[3]-xy[1]) ) xy[3] = xy[1] + (xy[2]-xy[0]);
   else xy[2] = xy[0] + (xy[3]-xy[1]);

   FreePointStruct(cl);
   FreePointStruct(ps);
   FreePointStruct(ss);

   dprintf(3," done!\n",pcount);
}


static struct Point *GetConformalView(struct Point *src, float l0)
{
   int i;
   float l, s, dl, ds, len;

   struct Point *line;

   line = AllocPointStruct();

   len = 0.0;
   l = l0;
   s = src->x[src->nump-1] * src->y[src->nump-1];
   AddPoint(line,s,l,len);
   dprintf(3,"\n GetConformalView():\n");
   for(i = src->nump-2; i >= 0; i--)
   {
      dl  = -l;
      l  += sqrt(pow(src->x[i+1]-src->x[i],2) + pow(src->z[i+1]-src->z[i],2));
      dl += l;
      ds  = -s;
      s  += 0.5*(src->x[i+1] + src->x[i]) * (src->y[i] - src->y[i+1]);
      ds += s;
      len+= sqrt(dl*dl + ds*ds);
      AddPoint(line,s,l,len);
      dprintf(5,"\t %3d  %16.8f  %16.8f  %16.8f (%16.8f)\n",
         i,s,l,len,180.0/M_PI*atan((l-line->y[line->nump-2])/
         (s-line->x[line->nump-2])) );
   }
   return (line);
}


static int AddLine2PlotVector(struct Point *line, float *xpl, float *ypl,
float *xy, int *pcount)
{
   int i;

   for(i = 1; i < line->nump; i++)
   {
      xpl[(*pcount)] = line->x[i-1];
      ypl[(*pcount)] = line->y[i-1];
      (*pcount)++;
      xpl[(*pcount)] = line->x[i];
      ypl[(*pcount)] = line->y[i];
      (*pcount)++;
      if(line->x[i] > xy[2]) xy[2] = line->x[i];
      if(line->y[i] > xy[3]) xy[3] = line->y[i];
   }
   return (*pcount);
}


static void FitPSSSCurves(struct Point *cl, struct Point *ps, struct Point *ss)
{
   int i, ilast;

   float dx, dy, len, para;

   if( ((ilast = ps->nump-1) != cl->nump-1) || ilast != ss->nump-1)
   {
      fprintf(stderr,"\n!!! lost blade profile point somehow!!!\n");
      exit(1);
   }
   // modify ps points to fit to cl!
   len   = ps->z[ilast];
   dx    = cl->x[ilast]-ps->x[ilast];
   dy    = cl->y[ilast]-ps->y[ilast];
   for(i = 1; i <= ilast; i++)
   {
      para = 1.0 - (len - ps->z[i])/len;
      ps->x[i] += para*dx;
      ps->y[i] += para*dy;
   }
   // ss points
   len   = ss->z[ilast];
   dx    = cl->x[ilast]-ss->x[ilast];
   dy    = cl->y[ilast]-ss->y[ilast];
   for(i = 1; i <= ilast; i++)
   {
      para = 1.0 - (len - ss->z[i])/len;
      ss->x[i] += para*dx;
      ss->y[i] += para*dy;
   }
}


void GetCamberPlotData(struct axial *ar, float *xpl, float *ypl, float *xy,
int c, int v_count)
{
   static int pcount;
   int i;

   float l0, alpha, p_l, len, alpha_deg, rad2deg;
   struct Point *cl;

   rad2deg = 180.0f/(float)M_PI;

   // ****************************************
   // get conformal view
   l0 = 0.0;                                      // start length coord. of cl.
   cl = GetConformalView(ar->me[c]->cl, l0);

   // ****************************************
   // get camber values
   alpha = atan((cl->y[0]-cl->y[1])/(cl->x[0]-cl->x[1]));
   len = cl->z[cl->nump-2];
   if(!v_count)
   {
      pcount = 0;
      xy[0] = 0.0;
      xy[2] = 1.0;
   }
   xpl[pcount] = cl->z[0]/len;
   ypl[pcount] = alpha*rad2deg;
   pcount++;
   for(i = 2; i < cl->nump; i++)
   {
      alpha     = atan((cl->y[i-1]-cl->y[i])/(cl->x[i-1]-cl->x[i]));
      p_l       = cl->z[i-1]/len;
      if((alpha_deg = alpha*rad2deg) < 0.0) alpha_deg += 180.0;
      xpl[pcount] = p_l;
      ypl[pcount] = alpha_deg;
      pcount++;
      if(i < cl->nump-1)
      {
         xpl[pcount] = p_l;
         ypl[pcount] = alpha_deg;
         pcount++;
      }
   }
   dprintf(3," GetCamberPlotData(): done!\n\n");
   xy[1] = 0.0;
   xy[3] = 90.0;

   FreePointStruct(cl);
}


void GetNormalizedCamber(struct axial *ar, float *xpl, float *ypl, float *xy,
int c, int v_count)
{
   static int pcount;
   int i, istart;

   float l0, alpha, p_l, len, alpha_deg, rad2deg, delta, alphamin, alphamax;
   struct Point *cl;

   rad2deg = 180.0f/(float)M_PI;

   // ****************************************
   // get conformal view
   l0 = 0.0;                                      // start length coord. of cl.
   cl = GetConformalView(ar->me[c]->cl, l0);

   // ****************************************
   // get camber values
   alpha = atan((cl->y[0]-cl->y[1])/(cl->x[0]-cl->x[1]));
   len = cl->z[cl->nump-2];
   if(!v_count)
   {
      pcount = 0;
      xy[0] = 0.0;
      xy[2] = 1.0;
   }
   istart = pcount;
   xpl[pcount] = cl->z[0]/len;
   ypl[pcount] = alphamin = alpha*rad2deg;
   pcount++;
   for(i = 2; i < cl->nump; i++)
   {
      alpha     = atan((cl->y[i-1]-cl->y[i])/(cl->x[i-1]-cl->x[i]));
      p_l       = cl->z[i-1]/len;
      if((alpha_deg = alpha*rad2deg) < 0.0) alpha_deg += 180.0;
      xpl[pcount] = p_l;
      ypl[pcount] = alpha_deg;
      pcount++;
      if(i < cl->nump-1)
      {
         xpl[pcount] = p_l;
         ypl[pcount] = alphamax = alpha_deg;
         pcount++;
      }
   }
   delta = 1.0/(alphamax-alphamin);
   for(i = istart; i < pcount; i++)
   {
      ypl[i] -= alphamin;
      ypl[i] *= delta;
   }
   dprintf(3," GetCamberPlotData(): done!\n\n");
   xy[1] = 0.0;
   xy[3] = 1.0;

   FreePointStruct(cl);
}


void GetMaxThicknessData(struct axial *ar, float *xpl, float *ypl, float *xy)
{
   int i, j, pcount;
   float t, t_max;

   pcount = 0;
   // ss

   dprintf(5,"GetMaxThicknessData() ...\n",pcount);

   for(i = 0; i < ar->be_num; i++)
   {
      t_max = 0.0;
      for(j = 0; j < ar->be[i]->cl->nump; j++)
      {
         t = sqrt(pow(ar->be[i]->cl_cart->x[j]-ar->be[i]->ss_cart->x[j],2) +
            pow(ar->be[i]->cl_cart->y[j]-ar->be[i]->ss_cart->y[j],2) +
            pow(ar->be[i]->cl_cart->z[j]-ar->be[i]->ss_cart->z[j],2));
         if(t_max < t) t_max = t;
      }
      xpl[pcount] = ar->be[i]->para;
      ypl[pcount] = -t_max;
      pcount++;
      if(i && i < ar->be_num-1)
      {
         xpl[pcount] = ar->be[i]->para;
         ypl[pcount] = -t_max;
         pcount++;
      }
   }
   dprintf(5,"GetMaxThicknessData: pcount = %d (ss done!)\n",pcount);

   // ps
   for(i = 0; i < ar->be_num; i++)
   {
      t_max = 0.0;
      for(j = 0; j < ar->be[i]->cl->nump; j++)
      {
         t = sqrt(pow(ar->be[i]->cl_cart->x[j]-ar->be[i]->ps_cart->x[j],2) +
            pow(ar->be[i]->cl_cart->y[j]-ar->be[i]->ps_cart->y[j],2) +
            pow(ar->be[i]->cl_cart->z[j]-ar->be[i]->ps_cart->z[j],2));
         if(t_max < t) t_max = t;
      }
      xpl[pcount] = ar->be[i]->para;
      ypl[pcount] = t_max;
      pcount++;
      if(i && i < ar->be_num-1)
      {
         xpl[pcount] = ar->be[i]->para;
         ypl[pcount] = t_max;
         pcount++;
      }
   }
   dprintf(5,"GetMaxThicknessData: pcount = %d (ps done!)\n",pcount);
   // hub-line
   xpl[pcount] = xpl[0];
   ypl[pcount] = xy[1] =  1.2*ypl[0];             // max. thickness supposed to be at hub (0)
   pcount++;
   xpl[pcount] = xpl[0];
   ypl[pcount] = xy[3] = -1.2*ypl[0];             // max. thickness supposed to be at hub (0)
   pcount++;

   xy[0] =  0.0;
   xy[2] =  1.0;

}


void GetMaxThicknessDistrib(struct axial *ar, float *xpl, float *ypl,float *xy)
{
   int i, j, pcount, i_totalmax;
   float t, t_max, t_totalmax, para_max,l, cllen,  *len;

   pcount = 0;

   if( (len = (float*)calloc(ar->be[0]->cl->nump,sizeof(float))) == NULL)
   {
      fatal("no mem. for *len!"); exit(-1);
   }

   t_totalmax = 0.0;
   for(i = 0; i < ar->be_num; i++)
   {
      t_max = 0.0;
      cllen = l = 0.0;
      // cl-length
      for(j = 0; j < ar->be[i]->cl->nump-1; j++)
      {
         len[j] = cllen;
         l = sqrt(pow(ar->be[i]->cl_cart->x[j+1]-ar->be[i]->cl_cart->x[j],2)+
            pow(ar->be[i]->cl_cart->y[j+1]-ar->be[i]->cl_cart->y[j],2)+
            pow(ar->be[i]->cl_cart->z[j+1]-ar->be[i]->cl_cart->z[j],2));
         cllen += l;
      }
      len[ar->be[i]->cl->nump-1] = cllen;
      // max. thickness position
      for(j = 0; j < ar->be[i]->ss->nump; j++)
      {
         t = sqrt(pow(ar->be[i]->ps_cart->x[j]-ar->be[i]->ss_cart->x[j],2) +
            pow(ar->be[i]->ps_cart->y[j]-ar->be[i]->ss_cart->y[j],2) +
            pow(ar->be[i]->ps_cart->z[j]-ar->be[i]->ss_cart->z[j],2));
         if(t_max < t)
         {
            t_max = t;
            para_max = len[j]/cllen;
         }
      }
      xpl[pcount] = ar->be[i]->para;
      ypl[pcount] = para_max;
      pcount++;
      if(i && i < ar->be_num-1)
      {
         xpl[pcount] = ar->be[i]->para;
         ypl[pcount] = para_max;
         pcount++;
      }
      if(t_totalmax < t_max)
      {
         i_totalmax = pcount;
         t_totalmax = t_max;
      }
      dprintf(5,"i = %3d: t_max = %f, para_max = %f\n",i,t_max,para_max);
   }
   free(len);

   // hub-line
   xy[0] =  0.0;
   xy[2] =  1.0;
   xy[1] =  0.95*ypl[i_totalmax];
   xy[3] =  1.05*ypl[i_totalmax];
}


void GetOverlapPlotData(struct axial *ar, float *xpl, float *ypl, float *xy)
{
   int i, pcount;
   float theta0, rad2deg, ratio;

   pcount  = 0;
   xy[3]   = 0.0;
   xy[1]   = 200.0;
   rad2deg = 180.0f/(float)M_PI;
   theta0  = 360.0f/ar->nob;
   ratio   = 100 * ( ((ar->me[0]->cl->y[0] - ar->me[0]->cl->y[ar->me[0]->cl->nump-1]) * rad2deg)
      / theta0 - 1.0);
   if(xy[1] > ratio) xy[1] = ratio;
   if(xy[3] < ratio) xy[3] = ratio;
   xpl[pcount] = ar->me[0]->para;
   ypl[pcount] = ratio;
   pcount++;
   for(i = 1; i < ar->be_num; i++)
   {
      ratio   = 100.0 * ( ((ar->me[i]->cl->y[0] - ar->me[i]->cl->y[ar->me[i]->cl->nump-1]) * rad2deg)
         / theta0 - 1.0);
      xpl[pcount] = ar->me[i]->para;
      ypl[pcount] = ratio;
      pcount++;
      if(i < ar->be_num-1)
      {
         xpl[pcount] = ar->me[i]->para;
         ypl[pcount] = ratio;
         pcount++;
      }
      if(xy[1] > ratio) xy[1] = ratio;
      if(xy[3] < ratio) xy[3] = ratio;
   }
   xy[0]   =  0.0;
   xy[2]   =  1.0;
}


void GetBladeAnglesPlotData(struct axial *ar, float *xpl, float *ypl,float *xy)
{
   int i, num, pcount;
   float dl, ds, rad2deg, alpha;

   struct Flist *le_ang;
   struct Flist *te_ang;

   struct Point *cl;

   le_ang = AllocFlistStruct(ar->be_num+1);
   te_ang = AllocFlistStruct(ar->be_num+1);

   rad2deg = 180.0f/(float)M_PI;
   pcount  = 0;

   // calc blade angles at inlet & outlet
   for(i = 0; i < ar->be_num; i++)
   {
      if((cl  = ar->me[i]->cl))
      {
         num = cl->nump-1;
         // le angle
         dl  = sqrt(pow(cl->x[1]-cl->x[0],2) + pow(cl->z[1]-cl->z[0],2));
         ds  = 0.5*(cl->x[1] + cl->x[0]) * (cl->y[0] - cl->y[1]);
         if( (alpha = atan(dl/ds)*rad2deg) < 0.0) alpha += 180.0;
         Add2Flist(le_ang, alpha);
         // te_angle
         dl  = sqrt(pow(cl->x[num]-cl->x[num-1],2) +
            pow(cl->z[num]-cl->z[num-1],2));
         ds  = 0.5*(cl->x[num] + cl->x[num-1]) * (cl->y[num-1] - cl->y[num]);
         if( (alpha = atan(dl/ds)*rad2deg) < 0.0) alpha += 180.0;
         Add2Flist(te_ang, alpha);
      }
   }
   if((ar->be_num != le_ang->num) ||(ar->be_num != te_ang->num))
   {
      fatal("point number mismatch for blade angles!");
      return;
   }
   // fill plot vectors
   xpl[pcount] = ar->me[0]->para;
   xy[1] = xy[3] = ypl[pcount] = le_ang->list[0];
   pcount++;
   for(i = 1; i < ar->be_num; i++)
   {
      xpl[pcount] = ar->me[i]->para;
      if(xy[1] > (ypl[pcount] = le_ang->list[i])) xy[1] = ypl[pcount];
      else if (xy[3] < ypl[pcount]) xy[3] = ypl[pcount];
      pcount++;
      if(i < ar->be_num-1)
      {
         xpl[pcount] = ar->me[i]->para;
         ypl[pcount] = le_ang->list[i];
         pcount++;
      }
   }
   xpl[pcount] = ar->me[0]->para;
   ypl[pcount] = te_ang->list[0];
   pcount++;
   for(i = 1; i < ar->be_num; i++)
   {
      xpl[pcount] = ar->me[i]->para;
      if(xy[1] > (ypl[pcount] = te_ang->list[i])) xy[1] = ypl[pcount];
      else if (xy[3] < ypl[pcount]) xy[3] = ypl[pcount];
      pcount++;
      if(i < ar->be_num-1)
      {
         xpl[pcount] = ar->me[i]->para;
         ypl[pcount] = te_ang->list[i];
         pcount++;
      }
   }
   xy[1]   = 0.0;
   xy[3]   = 90.0;
   xy[0]   =  0.0;
   xy[2]   =  1.0;
   FreeFlistStruct(le_ang);
   FreeFlistStruct(te_ang);
}

void GetChordAnglesPlotData(struct axial *ar,float *xpl,float *ypl,float *xy)
{
	int i, pcount, ilast;

	float x[2], r, dphi, dz;

	struct Point *cl;

	pcount = 0;

	for(i = 0; i < ar->be_num; i++) {
		if((cl  = ar->me[i]->cl)) {
			ilast = cl->nump-1;
			r     = 0.5*(cl->x[0]+cl->x[ilast]);
			dphi  = fabs(cl->y[0]-cl->y[ilast]);
			dz    = cl->z[0]-cl->z[ilast];
			x[0] = ar->me[i]->para;
			x[1] = 180.0/M_PI*myatan(r*dphi,dz);

			if(i) {
				xpl[pcount] = x[0];
				ypl[pcount] = x[1];
				pcount++;
			}
			xpl[pcount] = x[0];
			ypl[pcount] = x[1];
			pcount++;

		}
		else {
			dprintf(0," Centre line for plane no. %d missing!\n",i);
		}
	}
	
	xy[1]   = 0.0;
	xy[3]   = 90.0;
	xy[0]   =  0.0;
	xy[2]   =  1.0;
}

static float myatan(float x, float y)
{
	if(fabs(x) <= 1.e-8f && y > 0.0f) return (float) M_PI/2.0f;
	else if(fabs(x) <= 1.e-8f && y < 0.0f) return -(float)M_PI/2.0f;
	else return atan(y/x);
}

void GetParamPlotData(struct axial *ar, float *xpl, float *ypl,float *xy,
int ival)
{
   int i;
   float max = -1.e9f, min = 1.e9f;

   switch(ival)
   {
      case 4:
         for(i = 0; i < ar->be_num-1; i++)
         {
            xpl[2*i]   = ar->be[i]->para;
            ypl[2*i]   = ar->be[i]->mod_angle[0];
            xpl[2*i+1] = ar->be[i+1]->para;
            ypl[2*i+1] = ar->be[i+1]->mod_angle[0];
            min = MIN(min,ypl[2*i+1]);
            max = MAX(max,ypl[2*i+1]);
         }
         break;
      case 5:
         for(i = 0; i < ar->be_num-1; i++)
         {
            xpl[2*i]   = ar->be[i]->para;
            ypl[2*i]   = ar->be[i]->mod_angle[1];
            xpl[2*i+1] = ar->be[i+1]->para;
            ypl[2*i+1] = ar->be[i+1]->mod_angle[1];
            min = MIN(min,ypl[2*i+1]);
            max = MAX(max,ypl[2*i+1]);
         }
         break;
      case 6:
         for(i = 0; i < ar->be_num-1; i++)
         {
            xpl[2*i]   = ar->be[i]->para;
            ypl[2*i]   = ar->be[i]->p_thick;
            xpl[2*i+1] = ar->be[i+1]->para;
            ypl[2*i+1] = ar->be[i+1]->p_thick;
            min = MIN(min,ypl[2*i+1]);
            max = MAX(max,ypl[2*i+1]);
         }
         break;
      case 7:
         for(i = 0; i < ar->be_num-1; i++)
         {
            xpl[2*i]   = ar->be[i]->para;
            ypl[2*i]   = ar->be[i]->te_thick;
            xpl[2*i+1] = ar->be[i+1]->para;
            ypl[2*i+1] = ar->be[i+1]->te_thick;
            min = MIN(min,ypl[2*i+1]);
            max = MAX(max,ypl[2*i+1]);
         }
         break;
      case 8:
         for(i = 0; i < ar->be_num-1; i++)
         {
            xpl[2*i]   = ar->be[i]->para;
            ypl[2*i]   = ar->be[i]->camb;
            xpl[2*i+1] = ar->be[i+1]->para;
            ypl[2*i+1] = ar->be[i+1]->camb;
            min = MIN(min,ypl[2*i+1]);
            max = MAX(max,ypl[2*i+1]);
         }
         break;
      case 9:
         for(i = 0; i < ar->be_num-1; i++)
         {
            xpl[2*i]   = ar->be[i]->para;
            ypl[2*i]   = ar->be[i]->camb_pos;
            xpl[2*i+1] = ar->be[i+1]->para;
            ypl[2*i+1] = ar->be[i+1]->camb_pos;
            min = MIN(min,ypl[2*i+1]);
            max = MAX(max,ypl[2*i+1]);
         }
         break;
      case 10:
         for(i = 0; i < ar->be_num-1; i++)
         {
            xpl[2*i]   = ar->be[i]->para;
            ypl[2*i]   = ar->be[i]->bp_shift;
            xpl[2*i+1] = ar->be[i+1]->para;
            ypl[2*i+1] = ar->be[i+1]->bp_shift;
            min = MIN(min,ypl[2*i+1]);
            max = MAX(max,ypl[2*i+1]);
         }
         break;
      default:
         xpl[0] = 0.0; ypl[0]=0.0;
         xpl[1] = 1.0; ypl[1]=1.0;
         min = 0.0; max = 1.0;
         dprintf(1," No valid option %d!\n",ival);
         break;
   }
   if((max-min)<= 0.0) max = min+1.0;
   xy[0] = 0.0; xy[2] = 1.0;
   xy[1] = min; xy[3] = max;
}
