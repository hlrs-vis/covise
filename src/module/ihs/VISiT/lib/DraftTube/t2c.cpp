#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../General/include/geo.h"
#include "../General/include/cov.h"
#include "../General/include/points.h"
#include "../General/include/common.h"
#include "../General/include/fatal.h"
#include "../General/include/log.h"
#include "../General/include/v.h"
#include "include/tube.h"
#include "include/t2c.h"
#include "include/transform.h"

#define  I_DEB(a) dprintf(5, "%s = %d\n", #a, a)
#define  F_DEB(a) dprintf(5, "%s = %f\n", #a, a)

static void CheckAndCreateBCEntrySurface(struct covise_info *ci, struct cs *cs);
static struct ci_cs *CheckAndCreateCPol(struct cs *cs);
static void CheckAndCreatePol(struct covise_info *ci, struct cs *cs, struct cs *lcs);
static void AddPolygon(struct covise_info *ci, struct cs *cs, struct cs *lcs, int ind1, int ind2);
static void AddPolygons(struct covise_info *ci, struct cs *cs, struct cs *lcs, int ind1, int ind2);

struct covise_info *Tube2Covise(struct tube *tu)
{
   struct covise_info *ci = NULL;
   struct cs *cs;
   int i;

   if (CalcTubeGeometry(tu))
   {
      if ((ci = AllocCoviseInfo(tu->cs_num)) != NULL)
      {
         for (i = 0; i < tu->cs_num; i++)
         {
            cs = tu->cs[i];

            // first we add the local points to te global list ...
            T_AddPoints2CI(ci->p, cs);

            // the complete geometry
            if (i)
            {
               CheckAndCreatePol(ci, cs, tu->cs[i-1]);
            }

            // one cross section
            ci->ci_cs[i] = CheckAndCreateCPol(cs);

            // the entry cross section
            if (!i)
            {
               ci->bcinnumPoints = cs->p->nump;
               CheckAndCreateBCEntrySurface(ci, cs);
            }
         }
      }
#ifdef DEBUG
      DumpIlist(ci->cpol);
      DumpIlist(ci->cvx);
      dprintf(5, "------------BCEntrySurface : bcinnumPoints = %d\n", ci->bcinnumPoints);
      DumpIlist(ci->bcinpol);
      DumpIlist(ci->bcinvx);
#endif
   }
#ifdef DEBUG
   Tube2CoviseDump(ci);
#endif
   return ci;
}


struct Point * CS_BorderPoints(struct cs *cs)
{
   struct Point *p = AllocPointStruct();

   T_AddPoints2CI(p, cs);
   return p;
}


void T_AddPoints2CI(struct Point *points, struct cs *cs)
{
   int i;
   float r[3];
   float p[3];

   r[0] = cs->c_m_x;
   r[1] = cs->c_m_y;
   r[2] = cs->c_m_z;
   for (i = 0; i < cs->p->nump; i++)
   {

      p[0] = cs->p->x[i];
      p[1] = cs->p->y[i];
      p[2] = cs->p->z[i];
      Transform(cs->T, r, p, p+1, p+2);
      AddPoint(points, p[0], p[1], p[2]);
   }
}


static void CheckAndCreatePol(struct covise_info *ci, struct cs *cs, struct cs *lcs)
{
   int i;
   int ind2;

   for (i = 0; i < 8; i += 2)
   {
      // Corner
      AddPolygons(ci, cs, lcs, i, i+1);

      // flat part
      ind2 = (i == 6 ? 0 : i+2);
      AddPolygon(ci, cs, lcs, i+1, ind2);
   }
}


static void AddPolygons(struct covise_info *ci, struct cs *cs, struct cs *lcs, int ind1, int ind2)
{
   int i, j;
   int offs, loffs;
   int i1, i2, i3, i4;

   offs = ci->p->nump - cs->p->nump;
   loffs = offs - lcs->p->nump;
   i1 = cs->cov_ind[ind1] + offs;
   i2 = cs->cov_ind[ind2] + offs;
   i3 = lcs->cov_ind[ind1] + loffs;
   i4 = lcs->cov_ind[ind2] + loffs;

   if (i1 == i2)
   {
      if (i3 == i4)
      {
         //nice, nothing todo ...
      }
      else
      {
         for (i = i3; i < i4; i++)
         {
            Add2Ilist(ci->vx, i);
            Add2Ilist(ci->pol, ci->vx->num-1);
            Add2Ilist(ci->vx, i1);
            Add2Ilist(ci->vx, i+1);
         }
      }
   }
   else
   {
      if (i3 == i4)
      {
         for (i = i1; i < i2; i++)
         {
            Add2Ilist(ci->vx, i4);
            Add2Ilist(ci->pol, ci->vx->num-1);
            Add2Ilist(ci->vx, i+1);
            Add2Ilist(ci->vx, i);
         }
      }
      else
      {
         for (i = i1, j = i3; i < i2; i++,j++)
         {
            Add2Ilist(ci->vx, j+1);
            Add2Ilist(ci->pol, ci->vx->num-1);
            Add2Ilist(ci->vx, j);
            Add2Ilist(ci->vx, i);
            Add2Ilist(ci->vx, i+1);
         }
      }
   }
}


static void CheckAndCreateBCEntrySurface(struct covise_info *ci, struct cs *cs)
{
   int i;

   Add2Ilist(ci->bcinpol, ci->bcinvx->num);
   for (i = cs->p->nump; i > 0; i--)
   {
      Add2Ilist(ci->bcinvx, ci->p->nump-i);
   }
}


static struct ci_cs *CheckAndCreateCPol(struct cs *cs)
{
   int i;
   struct ci_cs *ci_cs;

   if ((ci_cs = AllocCiCsStruct()) != NULL)
   {
      T_AddPoints2CI(ci_cs->p, cs);
      for (i = 0; i < cs->p->nump; i++)
         Add2Ilist(ci_cs->cvx, i);
   }
   return ci_cs;
}


static void AddPolygon(struct covise_info *ci, struct cs *cs, struct cs *lcs, int ind1, int ind2)
{
   int offs, loffs;
   int i1, i2, i3, i4;

   offs = ci->p->nump - cs->p->nump;
   loffs = offs - lcs->p->nump;
   i1 = cs->cov_ind[ind1] + offs;
   i2 = cs->cov_ind[ind2] + offs;
   i3 = lcs->cov_ind[ind2] + loffs;
   i4 = lcs->cov_ind[ind1] + loffs;

   if (i1 == i2)
   {
      if (i3 == i4)
      {
         //nice, nothing todo ...
      }
      else
      {
         Add2Ilist(ci->vx, i3);
         Add2Ilist(ci->pol, ci->vx->num-1);
         Add2Ilist(ci->vx, i1);
         Add2Ilist(ci->vx, i4);
      }
   }
   else
   {
      if (i3 == i4)
      {
         Add2Ilist(ci->vx, i4);
         Add2Ilist(ci->pol, ci->vx->num-1);
         Add2Ilist(ci->vx, i1);
         Add2Ilist(ci->vx, i2);
      }
      else
      {
         Add2Ilist(ci->vx, i4);
         Add2Ilist(ci->pol, ci->vx->num-1);
         Add2Ilist(ci->vx, i1);
         Add2Ilist(ci->vx, i2);
         Add2Ilist(ci->vx, i3);
      }
   }
}


#ifdef DEBUG
void CiCsDump(FILE *fp, struct ci_cs *c)
{
   fprintf(fp, "Dump of all points\n");
   DumpPoints(c->p, fp);
   fprintf(fp, "Dump of all CS-vertex\n");
   DumpIlist(c->cvx);
}
#endif

#ifdef DEBUG
void Tube2CoviseDump(struct covise_info *ci)
{
   FILE *fp=NULL;
   int i;
   char *fn;
   char buf[50];

   if (!ci)
   {
      dprintf(1, "Tube2CoviseDump(): struct covise_info *ci = NULL !!\n");
      return;
   }
   dprintf(1, "Dump of Tube2CoviseData:\n");
   dprintf(5, "\tNumber of points: %d\n", ci->p->nump);

   fn = DebugFilename("cs_global.txt");
   if(fn)
   fp = fopen(fn, "w");
   DumpPoints(ci->p, fp);
   fclose(fp);
   dprintf(3, "\tDump of all vertex (Geometry):\n");
   DumpIlist(ci->vx);
   dprintf(3, "\tDump of all polygon (Geometry):\n");
   DumpIlist(ci->pol);
   for (i = 0; i < ci->num_cs; i++)
   {
      dprintf(4, "\tcross-section %d:\n", i);
      fp=NULL;
      fn = DebugFilename(buf);
      if(fn)
      fp = fopen(fn, "w");
      sprintf(buf, "cross-section_%04d", i);
      if(fp)
      {
      CiCsDump(fp, ci->ci_cs[i]);
      fclose(fp);
      }
   }
}
#endif
