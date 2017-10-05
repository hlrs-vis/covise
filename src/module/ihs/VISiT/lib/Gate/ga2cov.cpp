#include <stdio.h>
#include <stdlib.h>
#ifndef _WIN32
#include <strings.h>
#endif
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <math.h>
#include <Gate/include/gate.h>
#include <Gate/include/ga2cov.h>
#include <General/include/geo.h>
#include <General/include/cov.h>
#include <General/include/points.h>
#include <General/include/curve.h>
#include <General/include/profile.h>
#include <General/include/common.h>
#include <General/include/fatal.h>
#include <General/include/v.h>

#define NUMBER_OF_SECTIONS 36

struct covise_info *Gate2Covise(struct gate *ga)
{
   struct covise_info *ci = NULL;
   int j, jmax;
   int offs, te;
   float x, y, z;
#ifdef DEBUG_POLYGONS
   FILE *ferr;
   char fname[255];
   static int fcount = 0;
#endif                                         // DEBUG_POLYGONS

   // memory for polygon data, delete previous data
   if (CreateGA_BladeElements(ga))
   {
      if ((ci = AllocCoviseInfo(0)) != NULL)
      {
         // gate: just two "blade elements"
         // points to copy, gate: no element te_thick!
         jmax = ga->bp->num;
         offs = 2 * ga->bp->num -1;
         te   = 0;

         // runner contours and hub cap height
         FreePointStruct(ga->phub);
         FreePointStruct(ga->phub_n);
         FreePointStruct(ga->pshroud);
         ga->phub    = AllocPointStruct();
         ga->phub_n    = AllocPointStruct();
         ga->pshroud = AllocPointStruct();

         if (ga->geofromfile==1)
         {
            // read hub and shroud point list from cfg-File
            ReadPointStruct(ga->phub, "[hub contour]", ga->cfgfile);
            ReadPointStruct(ga->pshroud, "[shroud contour]", ga->cfgfile);
         }

         // assign points to global array from blade elements:
         // shroud blade points
         for (j = ga->bp->num-1; j >= 0; j--)
         {
            x = ga->ps->x[j];
            y = ga->ps->y[j];
            if (ga->geofromfile==0)
            {
               z = ga->in_z;
            }
            else
            {
               z = ga->pshroud->z[0];
            }
            AddPoint(ci->p, x, y, z);
         }
         for (j = 1; j < jmax; j++)
         {
            x = ga->ss->x[j];
            y = ga->ss->y[j];
            if (ga->geofromfile==0)
            {
               z = ga->in_z;
            }
            else
            {
               z = ga->pshroud->z[0];
            }
            AddPoint(ci->p, x, y, z);
         }
         // hub blade points
         for (j = ga->bp->num-1; j >= 0; j--)
         {
            x = ga->ps->x[j];
            y = ga->ps->y[j];
            if (ga->geofromfile==0)
            {
               z = ga->in_z + ga->in_height;
            }
            else
            {
               z = ga->phub->z[0];
            }
            AddPoint(ci->p, x, y, z);
         }
         for (j = 1; j < jmax; j++)
         {
            x = ga->ss->x[j];
            y = ga->ss->y[j];
            if (ga->geofromfile==0)
            {
               z = ga->in_z + ga->in_height;
            }
            else
            {
               z = ga->phub->z[0];
            }
            AddPoint(ci->p, x, y, z);
         }

         CreateGA_BEPolygons(ci, 2, offs);

      }
      // generate all blades of the runner
      RotateGABlade4Covise(ci, ga->nob);

      // hub and shroud contour
      CreateGA_CoviseContours(ci, ga);

   }

#ifdef DEBUG_POLYGONS
   int i;
   sprintf(fname, "ga_polygons_%02d.txt", fcount++);
   ferr = fopen(fname, "w");
   if (ferr)
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
#endif                                         // DEBUG_POLYGONS
   return ci;
}


int CreateGA_BEPolygons(struct covise_info *ci, int be, int offs)
{
   (void) be;
   int i, ivx[3];
   static int ipol;

   ipol = 0;

   // surface polygons
   for (i = 0; i < offs-1; i++)
   {
      // 1st polygon
      ivx[0] = i;
      ivx[1] = ivx[0]+offs;
      ivx[2] = ivx[1]+1;
      Add2Ilist(ci->pol, ipol);
      Add2Ilist(ci->vx, ivx[0]);
      Add2Ilist(ci->vx, ivx[1]);
      Add2Ilist(ci->vx, ivx[2]);
      ipol += 3;
      // 2nd polygon
      ivx[0] = offs+1+i;
      ivx[1] = ivx[0]-offs;
      ivx[2] = ivx[1]-1;
      Add2Ilist(ci->pol, ipol);
      Add2Ilist(ci->vx, ivx[0]);
      Add2Ilist(ci->vx, ivx[1]);
      Add2Ilist(ci->vx, ivx[2]);
      ipol += 3;
   }

   return(ipol);

}


void RotateGABlade4Covise(struct covise_info *ci, int nob)
// so for same as RotateBlade4Covise in ar2cov.c
{
   int i, j, ipol, ivx;
   int np, npol, nvx;
   float rot, roma[2][2];
   float x, y, z;

   np         = ci->p->nump;
   npol       = ci->pol->num;
   nvx        = ci->vx->num;
   rot        = float(2 * M_PI / nob);
   roma[0][0] = float(cos(rot));
   roma[0][1] = float(-sin(rot));
   roma[1][0] = float(sin(rot));
   roma[1][1] = float(cos(rot));

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


void CreateGA_CoviseContours(struct covise_info *ci, struct gate *ga)
{
   int i, j, ind, hub;
   int nphub;
   float x, y, z;
   float angle, roma[2][2];
   const int npblade  = ci->p->nump;
   const float rot = 2 * (float) M_PI / NUMBER_OF_SECTIONS;

   if (ga->geofromfile==0)
   {
      if (ga->radial==0)
      {
         // create hub point list from parameters
                                                  // hub inlet point
         AddPoint(ga->phub, 0.0, ga->in_rad, ga->in_z + ga->in_height);
         for (i = 0; i < ga->num_hub_arc; i++)    // ellipse corner
         {
            y = float(ga->hub_ab[0] * cos (M_PI/2. + i * M_PI/2. / (ga->num_hub_arc - 1)));
            z = float(ga->hub_ab[1] * sin (M_PI/2. + i * M_PI/2. / (ga->num_hub_arc - 1)));
            AddPoint(ga->phub, 0.0, y + ga->out_rad1 + ga->hub_ab[0] , z + ga->in_z + ga->in_height - ga->hub_ab[1]);
         }
         AddPoint(ga->phub, 0.0, ga->out_rad1, ga->out_z);

         // create shroud point list from parameters
                                                  // hub inlet point
         AddPoint(ga->pshroud, 0.0, ga->in_rad, ga->in_z);
         for (i = 0; i < NPOIN_SHROUD_AB; i++)    // ellipse corner
         {
            y = float(ga->shroud_ab[0] * cos (M_PI/2. + i * M_PI/2. / (NPOIN_SHROUD_AB - 1)));
            z = float(ga->shroud_ab[1] * sin (M_PI/2. + i * M_PI/2. / (NPOIN_SHROUD_AB - 1)));
            AddPoint(ga->pshroud, 0.0, y + ga->out_rad2 + ga->shroud_ab[0] , z + ga->in_z - ga->shroud_ab[1]);
         }
         AddPoint(ga->pshroud, 0.0, ga->out_rad2, ga->out_z);
      }
      else
      {
         // create hub point list from parameters
                                                  // hub inlet point
         AddPoint(ga->phub, 0.0, ga->in_rad, ga->in_z + ga->in_height);
         AddPoint(ga->phub, 0.0, ga->out_rad2, ga->in_z + ga->in_height);

         // create shroud point list from parameters
                                                  // hub inlet point
         AddPoint(ga->pshroud, 0.0, ga->in_rad, ga->in_z);
         AddPoint(ga->pshroud, 0.0, ga->out_rad2, ga->in_z);
      }
   }

   // here we have hub and shroud point lists ga->phub and ga->pshroud

   // create hub polygons
   // remark: hub can and does have corners.
   // For normal generation (further down) it is useful to retain these corners,
   // which means to split up the polygons in meridional direction
   // we have to duplicate the inner points
   // for covise polygons, we work with the lists ga->phub_n and ga->pshroud_n
   // grid generator gets ga->phub and ga->pshroud

                                                  // 1st point
   AddPoint(ga->phub_n, ga->phub->x[0], ga->phub->y[0], ga->phub->z[0]);
   for (i=1; i<ga->phub->nump-1; i++)
   {
                                                  // inner points
      AddPoint(ga->phub_n, ga->phub->x[i], ga->phub->y[i], ga->phub->z[i]);
                                                  // inner points
      AddPoint(ga->phub_n, ga->phub->x[i], ga->phub->y[i], ga->phub->z[i]);
   }
                                                  // last point
   AddPoint(ga->phub_n, ga->phub->x[i], ga->phub->y[i], ga->phub->z[i]);

   // append hub contour point coordinates
   for (i = ga->phub_n->nump-1; i >= 0; i--)
      AddPoint(ci->p, ga->phub_n->x[i], ga->phub_n->y[i], ga->phub_n->z[i]);

   // rotate hub contour and append points
   for (i = 1; i < NUMBER_OF_SECTIONS; i++)
   {
      angle      = i * rot;
      roma[0][0] = float(cos(angle));
      roma[0][1] = float(-sin(angle));
      roma[1][0] = float(sin(angle));
      roma[1][1] = float(cos(angle));
      for (j = 0; j < ga->phub_n->nump; j++)
      {
         ind = npblade + j;
         x   = ci->p->x[ind] * roma[0][0] + ci->p->y[ind] * roma[0][1];
         y   = ci->p->x[ind] * roma[1][0] + ci->p->y[ind] * roma[1][1];
         z   = ci->p->z[ind];
         AddPoint(ci->p, x, y, z);
      }
   }

   hub = 0;
   for (i = 1; i <= NUMBER_OF_SECTIONS; i++)
      CreateGAContourPolygons(ci->lpol, ci->lvx, i, ga->phub_n->nump, npblade, HUB);

   // append shroud contour point coordinates to global array
   nphub = ci->p->nump - npblade;
   for (i = ga->pshroud->nump-1; i >= 0; i--)
      AddPoint(ci->p, ga->pshroud->x[i], ga->pshroud->y[i], ga->pshroud->z[i]);

   // rotate hub contour and append points
   for (i = 1; i < NUMBER_OF_SECTIONS; i++)
   {
      angle      = i * rot;
      roma[0][0] = float(cos(angle));
      roma[0][1] = float(-sin(angle));
      roma[1][0] = float(sin(angle));
      roma[1][1] = float(cos(angle));
      for (j = 0; j < ga->pshroud->nump; j++)
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
      CreateGAContourPolygons(ci->cpol, ci->cvx, i, ga->pshroud->nump, (npblade+nphub), SHROUD );

   /*
       // the old version
      hub = 0;
      for (i = 1; i <= NUMBER_OF_SECTIONS; i++)
         CreateGAContourPolygons(ci->lpol, ci->lvx, i, ga->phub->nump, npblade);

         // append shroud contour point coordinates to global array
      nphub = ci->p->nump - npblade;
      for (i = ga->pshroud->nump-1; i >= 0; i--)
         AddPoint(ci->p, ga->pshroud->x[i], ga->pshroud->y[i], ga->pshroud->z[i]);

   // rotate hub contour and append points
   for (i = 1; i < NUMBER_OF_SECTIONS; i++) {
   angle      = i * rot;
   roma[0][0] =  cos(angle);
   roma[0][1] = -sin(angle);
   roma[1][0] =  sin(angle);
   roma[1][1] =  cos(angle);
   for (j = 0; j < ga->pshroud->nump; j++) {
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
   CreateGAContourPolygons(ci->cpol, ci->cvx, i, ga->pshroud->nump, (npblade+nphub) );
   */
}


void CreateGAContourPolygons(struct Ilist *ci_pol, struct Ilist *ci_vx, int sec, int offs, int np, int part)
{
   int i;
   int vx[3];
   static int ipol;

   int increment = 0;

   if (part==HUB)
      increment = 2;
   if (part==SHROUD)
      increment = 1;

   if (sec == 1) ipol = 0;
   if (sec < NUMBER_OF_SECTIONS)
   {
      for (i = 0; i < offs-1; i+=increment)
      {
         // 1st polygon
         vx[0]= np  + (sec - 1) * offs + i;
         vx[1]= vx[0] + offs;
         vx[2]= vx[1] + 1;
         Add2Ilist(ci_vx, vx[0]);
         Add2Ilist(ci_vx, vx[1]);
         Add2Ilist(ci_vx, vx[2]);
         Add2Ilist(ci_pol, ipol);
         ipol += 3;
         // 2nd polygon
         vx[0]= np  + sec * offs + 1 + i;
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
      for (i = 0; i < offs-1; i+=increment)
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
