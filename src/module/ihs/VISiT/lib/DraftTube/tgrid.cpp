#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "include/tmesh.h"
#include "include/tgrid.h"
#include "../General/include/elements.h"
#include "include/tube.h"
#include "include/t2c.h"
#include "../General/include/vector.h"
#include "../General/include/nonlinpart.h"
#include "../General/include/fatal.h"
#include "../General/include/log.h"
#include "../General/include/common.h"
#include "../General/include/v.h"

static void TGridInit(struct tube *tu, struct tgrid *tg);
static struct tgrid *AllocTGrid(void);
static void AllocT_GS(struct tgrid *tg);
static void MeshPointsISection(struct tgrid *tg, struct gs *gs, int ind);
static void MeshPointsOSection(struct tube *tu, struct tgrid *tg, int sec, int ind);
static void Trimm2Ellipse(float v[3], float a, float b, float x, float y);

static float quadx[8] = { 1, 1, 0, -1, -1, -1, 0, 1};
static float quady[8] = { 0, 1, 1, 1, 0, -1, -1, -1};

#define  EVEN(x)  (IS_0((int)(x)/(int)(2)-(float)(x)/2.0))
#define  SI(x) dprintf(5, "%20s = %d\n", #x, (x))
#define  SF(x) dprintf(5, "%20s = %f\n", #x, (x))
#define  SV(x) dprintf(5, "%20s = (%f,%f,%f)\n", #x, x[0], x[1], x[2])

struct tgrid *CreateTGrid(struct tube *tu)
{
   int i, j, ind;
   float dx, dy;
   struct tgrid *tg;
   int elems;

   if ((tg = AllocTGrid()) != NULL)
   {
      TGridInit(tu, tg);

      // this is only some preparing ...
      for ( i = 0; i < tu->cs_num; i++)
      {
         AllocT_GS(tg);
         tg->gs[i]->num_elems = tu->cs[i]->c_nume;
         for (j = 0; j < 8; j++)
            tg->gs[i]->part[j] = tu->cs[i]->c_part[j];
         tg->gs[i]->linfact = tu->cs[i]->c_linfact;
         for ( j = 0; j < 8; j++)
         {

            dx = tu->cs[i]->c_width/2;
            dy = tu->cs[i]->c_height/2;
            V_0(tg->gs[i]->rr[j]);
            tg->gs[i]->rr[j][0] = dx*quadx[j];
            tg->gs[i]->rr[j][1] = dy*quady[j];
            V_Copy(tg->gs[i]->dr[j], tg->gs[i]->rr[j]);
            V_MultScal(tg->gs[i]->dr[j], 1.0f / (float)(GetoColElems(tg, j)));

            if (j == 1 || j == 3 || j == 5 || j == 7)
            {
               int ind = (j-1)/2;

               V_Copy(tg->gs[i]->ro[j], tg->gs[i]->rr[j]);
               Trimm2Ellipse(tg->gs[i]->ro[j], tu->cs[i]->c_a[ind],
                  tu->cs[i]->c_b[ind], dx, dy);
            }
            else
            {
               tg->gs[i]->ro[j][0] = dx*quadx[j];
               tg->gs[i]->ro[j][1] = dy*quady[j];
               tg->gs[i]->ro[j][2] = 0.0;
            }
            V_Copy(tg->gs[i]->rm[j], tg->gs[i]->ro[j]);
            V_MultScal(tg->gs[i]->rm[j], tg->gs[i]->part[j]);
         }
         for ( j = 0; j < 8; j++)
         {
            ind = (j+1 < 8 ? j+1 : j+1-8);
            elems = GetoColElems(tg, j);
            // dr
            V_Sub(tg->gs[i]->rr[j], tg->gs[i]->rr[ind], tg->gs[i]->dr[j]);
            V_MultScal(tg->gs[i]->dr[j], 1.0f / (float)(elems));
            // m
            V_Sub(tg->gs[i]->rm[j], tg->gs[i]->rm[ind], tg->gs[i]->m[j]);
            // dm
            V_Copy(tg->gs[i]->dm[j], tg->gs[i]->m[j]);
            V_MultScal(tg->gs[i]->dm[j], 1.0f / (float)(elems));
         }
      }

      // Now starts the Point-Creating for every CrossSection ...
      for ( i = 0; i < tu->cs_num; i++)
      {
         // first, we create the nodes for the inner sections
         AddPoint(tg->gs[i]->p, 0.0, 0.0, 0.0);
         for ( j = 0; j < 4; j++)
         {
            MeshPointsISection(tg, tg->gs[i], j);
         }
         // next the nodes for the outer sections
         for ( j = 0; j < 8; j++)
         {
            MeshPointsOSection(tu, tg, i, j);
         }
      }

      // and last (but not least) we create the global grid points
      //   and the the elements ...
      MeshElemsTube(tu, tg);
   }
   return tg;
}


int WriteTGrid(struct tgrid *tg, const char *fn)
{
   int i;
   int res;
   char buf[256];
   FILE *fp;

   res = 0;
   sprintf(buf, "%s.geo", fn);
   if ((fp = fopen(buf, "w")) != NULL)
   {
      fputs("## Geomtriedaten (automatisch erzeugt)\n\n\n\n\n\n\n\n\n\n", fp);
      fprintf(fp, "%d %d 0 0 %d %d %d %d\n", tg->p->nump, tg->e->nume,
         tg->p->nump, tg->e->nume,
         tg->p->nump, tg->e->nume);
      for (i = 0; i < tg->p->nump; i++)
      {
         fprintf(fp, "%5d %12.6f %12.6f %12.6f 0\n", i+1, tg->p->x[i], tg->p->y[i], tg->p->z[i]);
      }
      for (i = 0; i < tg->e->nume; i++)
      {
         fprintf(fp, "%6d %6d %6d %6d %6d %6d %6d %6d %6d 0\n",i+1,
            tg->e->e[i][0]+1, tg->e->e[i][1]+1, tg->e->e[i][2]+1, tg->e->e[i][3]+1,
            tg->e->e[i][4]+1, tg->e->e[i][5]+1, tg->e->e[i][6]+1, tg->e->e[i][7]+1);
      }
      fclose(fp);
      res++;
   }
   return res;
}


int WriteTBoundaryConditions(struct tgrid *tg, const char *fn)
{
   char buf[256];
   int i;
   FILE *fp;

   sprintf(buf, "%s.rb", fn);
   if ((fp = fopen(buf, "w")) != NULL)
   {
      fputs("##############################################\n", fp);
      fputs("# Randbedingungen fuer Saugrohr              #\n", fp);
      fputs("#                                            #\n", fp);
      fputs("#                                            #\n", fp);
      fputs("#                                            #\n", fp);
      fputs("#                                            #\n", fp);
      fputs("#                                            #\n", fp);
      fputs("#                                            #\n", fp);
      fputs("#                                            #\n", fp);
      fputs("##############################################\n", fp);
      fprintf(fp, "%d %d %d %d %d %d %d %d\n", tg->gs[0]->p->nump*6,
         tg->wall->numv, 0, 0, 0, 0, tg->in->numv+tg->out->numv, 0);
      for (i = 1; i <= tg->gs[0]->p->nump; i++)
      {
         fprintf(fp, "%d 1 0.0\n", i);
         fprintf(fp, "%d 2 0.0\n", i);
         fprintf(fp, "%d 3 -1.0\n", i);
         fprintf(fp, "%d 4 %f\n", i, tg->epsilon);
         fprintf(fp, "%d 5 %f\n", i, tg->k);
         fprintf(fp, "%d 6 %f\n", i, tg->T);
      }
      for (i = 0; i < tg->wall->numv; i++)
      {
         // erste NULL. (0:1) stehende, bewegte Wand
         // zweite NULL. (0:1) Temp.-Flaechen
         // dritte NULL: stehen lassen
         fprintf(fp, "%6d %6d %6d %6d 0 0 0 %6d\n",
            tg->wall->v[i][0]+1, tg->wall->v[i][1]+1, tg->wall->v[i][2]+1,
            tg->wall->v[i][3]+1, tg->wall->v[i][4]+1);
      }
      // Bilanzflaechen (Auswertung)
      for (i = 0; i < tg->in->numv; i++)
      {
         fprintf(fp, "%6d %6d %6d %6d %6d %6d\n",
            tg->in->v[i][0]+1, tg->in->v[i][1]+1, tg->in->v[i][2]+1,
            tg->in->v[i][3]+1, tg->in->v[i][4]+1, tg->bc_inval);
      }
      for (i = 0; i < tg->out->numv; i++)
      {
         fprintf(fp, "%6d %6d %6d %6d %6d %6d\n",
            tg->out->v[i][0]+1, tg->out->v[i][1]+1, tg->out->v[i][2]+1,
            tg->out->v[i][3]+1, tg->out->v[i][4]+1, tg->bc_outval);
      }
      fclose(fp);
   }
   return 0;
}


static void MeshPointsOSection(struct tube *tu, struct tgrid *tg, int sec, int ind)
{
   int i, j;
   int last;
   int ind1;
   int cols, rows;
   float tmp[3];
   float ltmp[3];
   float len, corner;
   float rr[3];                                   // vector to the endpoint on the rectangle
   float drr[3];
   float r0[3];                                   // vector to the startpoint on the inner line
   float dr0[3];
   float r1[3];                                   // vector to the endpoint on the outer line
   float dr[3];
   float P[3];
   float *part;

   last = tg->gs[sec]->p->nump;
   ind1 = (ind+1 > 7 ? ind+1-8 : ind+1);
   cols = GetoColElems(tg, ind);
   rows = GetoRowElems(tg);
   if ((part = (float *)calloc(rows+1, sizeof(float))) != NULL)
   {
      nonlinearpartition(part, rows+1, 1.0, tg->gs[sec]->linfact);
   }

   V_Copy(rr, tg->gs[sec]->rr[ind1]);
   V_Copy(drr, tg->gs[sec]->dr[ind]);

   V_Copy(r0, tg->gs[sec]->rm[ind1]);
   V_Copy(dr0, tg->gs[sec]->dm[ind]);
   if (ind == 0 || ind == 3 || ind == 4 || ind == 7)
   {
      corner = tu->cs[sec]->c_b[ind/2];
      len = tu->cs[sec]->c_height/2;
   }
   else
   {
      corner = tu->cs[sec]->c_a[ind/2];
      len = tu->cs[sec]->c_width/2;
   }
   V_0(tmp);
   for (i = 0; i < cols; i++)
   {
      V_Add(tmp, drr, tmp);
      V_Add(rr, drr, rr);
      V_Copy(r1, rr);
      V_Add(r0, dr0, r0);

      if (ind == 0 || ind == 2 || ind == 4 || ind == 6)
      {
         if (V_Len(tmp) < corner)
         {
            Trimm2Ellipse(r1, tu->cs[sec]->c_a[ind/2],
               tu->cs[sec]->c_b[ind/2],
               tu->cs[sec]->c_width/2,
               tu->cs[sec]->c_height/2);
         }
      }
      else
      {
         if (V_Len(tmp) > (len - corner))
         {
            Trimm2Ellipse(r1, tu->cs[sec]->c_a[ind/2],
               tu->cs[sec]->c_b[ind/2],
               tu->cs[sec]->c_width/2,
               tu->cs[sec]->c_height/2);
         }
      }
      V_Sub(r1, r0, dr);
      for (j = 0; j < rows; j++)
      {
         V_Copy(ltmp, dr);
         V_MultScal(ltmp, part[j+1]);
         V_Add(r0, ltmp, P);
         AddPoint(tg->gs[sec]->p, P[0], P[1], P[2]);
      }
   }
   tg->numoP[ind] = tg->gs[sec]->p->nump - last;
   tg->numoSP[ind] = last;
   tg->numosP[ind] = tg->numoP[ind] + (ind ? tg->numosP[ind-1] : tg->numisP[3]);
   if (part)   free(part);
}


static void MeshPointsISection(struct tgrid *tg, struct gs *gs, int quad)
{
   int i, j;
   int ind1, ind2;
   int cols, rows;
   int last;
   float r0[3];
   float r1[3];
   float P[3];
   float dr[3];
   float dr0[3];
   float dr1[3];

   ind1 = ((quad*2)+1 > 7 ? (quad*2)+1-8 : (quad*2)+1);
   ind2 = ((quad*2)+2 > 7 ? (quad*2)+2-8 : (quad*2)+2);
   last = gs->p->nump;
   cols = GetiColPoints(tg, quad);
   rows = GetiRowPoints(tg, quad);

   V_0(r0);
   V_Copy(dr0, gs->rm[quad*2]);
   V_MultScal(dr0, 1.0f / (float)(cols-1));

   V_Copy(r1, gs->rm[ind2]);
   V_Copy(dr1, gs->dm[ind1]);

   for (i = 1; i < cols; i++)
   {
      V_Add(r0, dr0, r0);
      V_Add(r1, dr1, r1);
      V_Sub(r1, r0, dr);
      V_MultScal(dr, 1.0f / (float)(rows-1));
      V_Copy(P, r0);
      AddPoint(gs->p, P[0], P[1], P[2]);
      for (j = 1; j < rows; j++)
      {
         V_Add(P, dr, P);
         AddPoint(gs->p, P[0], P[1], P[2]);
      }
   }
   tg->numiSP[quad] = last;
   tg->numiP[quad]  = gs->p->nump - last;
   tg->numisP[quad] = tg->numiP[quad] + (quad ? tg->numisP[quad-1] : 0);
}


static void TGridInit(struct tube *tu, struct tgrid *tg)
{
   int i;

   for (i = 0; i < 4; i++)
      tg->num_i[i] = tu->c_el[i];
   tg->num_o     = tu->c_el_o;
   tg->epsilon   = 0.003f;
   tg->k         = 0.001f;
   tg->T         = 0.0;
   tg->bc_inval  = 100;
   tg->bc_outval = 110;
}


void FreeStructTGrid(struct tgrid *tg)
{
   int i;

   FreePointStruct(tg->p);
   FreeElementStruct(tg->e);
   for (i = 0; i < tg->gs_num; i++)
      FreeStructT_GS(tg->gs[i]);
   free(tg->gs);
   free(tg);
}


void FreeStructT_GS(struct gs *gs)
{
   FreePointStruct(gs->p);
   free(gs);
}


static struct tgrid *AllocTGrid(void)
{
   struct tgrid *tg;

   if ((tg = (struct tgrid *)calloc(1, sizeof(struct tgrid))) != NULL)
   {
      tg->gs_max = 10;
      tg->gs = (struct gs **)calloc(tg->gs_max, sizeof(struct gs *));
      tg->p = AllocPointStruct();
      tg->e = AllocElementStruct();
   }
   return tg;
}


static void AllocT_GS(struct tgrid *tg)
{
   int nnum;

   nnum = tg->gs_num + 1;

   if (nnum > tg->gs_max)
   {
      tg->gs_max += 10;
      if ((tg->gs = (struct gs **)realloc(tg->gs,
         tg->gs_max*sizeof(struct gs *))) == NULL)
         fatal("Space");
   }
   if ((tg->gs[nnum-1] = (struct gs *)calloc(1, sizeof(struct gs))) == NULL)
      fatal("Space");;
   tg->gs[nnum-1]->p = AllocPointStruct();
   tg->gs_num++;
}


static void Trimm2Ellipse(float v[3], float a, float b, float rx, float ry)
{
   float xp, yp;
   float sx, sy;
   float lr[3];
   float save[3];
   float r[3];
   float m, d;
   float A, B, C;

   V_Copy(save, v);
   if (!IS_0(a) && !IS_0(b))
   {
      V_0(lr);
      V_0(r);
      sx = (v[0] < 0.0f ? -1.0f : 1.0f);
      sy = (v[1] < 0.0f ? -1.0f : 1.0f);
      r[0] = (rx - a);
      r[1] = (ry - b);

      xp = v[0] * sx;
      yp = v[1] * sy;

      // if m is infinite, we haven't to do anything (this could only be,
      //      if the point is on one of the axis
      if (!IS_0(xp))
      {
         m = yp/xp;
         d = m*r[0] - r[1];
         r[0] *= sx;
         r[1] *= sy;

         A = m*m + (b*b)/(a*a);
         B = 2*m*d;
         C = d*d - b*b;
         lr[0] = (-B + float(sqrt(B*B - 4*A*C))/(2*A));
         lr[1] = (m * lr[0] + d) * sy;
         lr [0] *= sx;
         V_Add(lr, r, v);
      }
   }
}


// Macros would be better, but for debugging ;-))
int GetiRowPoints(struct tgrid *tg, int ind)
{
   return (GetiRowElems(tg, ind)+1);
}


int GetiColPoints(struct tgrid *tg, int ind)
{
   return (GetiColElems(tg, ind)+1);
}


int GetoRowPoints(struct tgrid *tg)
{
   return (GetoRowElems(tg)+1);
}


int GetoColPoints(struct tgrid *tg, int ind)
{
   return (GetoColElems(tg, ind)+1);
}


int GetiRowElems(struct tgrid *tg, int ind)
{
   return (tg->num_i[(ind+1 > 3 ? ind+1-4 : ind+1)]);
}


int GetiColElems(struct tgrid *tg, int ind)
{
   return (tg->num_i[ind]);
}


int GetoRowElems(struct tgrid *tg)
{
   return (tg->num_o);
}


int GetoColElems(struct tgrid *tg, int ind)
{
   if (ind == 1 || ind == 6)
      return tg->num_i[0];
   else if (ind == 0 || ind == 3)
      return tg->num_i[1];
   else if (ind == 2 || ind == 5)
      return tg->num_i[2];
   return tg->num_i[3];
}


void DumpTGrid(struct tgrid *tg)
{
   int i, j;
   FILE *fp;
   char tmp[256];
   char *fn;

   SI(tg->num_o);
   SI(tg->gs_max);
   SI(tg->gs_num);
   for (i = 0; i < tg->gs_num; i++)
   {
      fprintf(stderr, "Grid-Section = %d\n", i);
      sprintf(tmp, "tg_%d.txt", i);
      fp=NULL;
      if ((fn = DebugFilename(tmp)))
         fp = fopen(fn, "w");
      for (j = 0; j < 8; j++)
      {
         SI(j);
         SV(tg->gs[i]->ro[j]);
         SV(tg->gs[i]->rm[j]);
         SV(tg->gs[i]->m[j]);
         SV(tg->gs[i]->dm[j]);
         if (fp)
         {
            fprintf(fp, "%f %f\n", tg->gs[i]->rm[j][0], tg->gs[i]->rm[j][1]);
         }
      }
      if (fp)
      {
         fprintf(fp, "%f %f\n", tg->gs[i]->rm[0][0], tg->gs[i]->rm[0][1]);
         fclose(fp);
      }
      sprintf(tmp, "grid_%d.txt", i);
      
      fp=NULL;
      fn = DebugFilename(tmp);
      if (fn && (fp = fopen(fn, "w")) != NULL)
      {
         for (j = 0; j < tg->gs[i]->p->nump; j++)
         {
            fprintf(fp, "%f %f\n", tg->gs[i]->p->x[j], tg->gs[i]->p->y[j]);
         }
         fclose(fp);
      }
      sprintf(tmp, "rect_%d.txt", i);
      fp=NULL;
      fn = DebugFilename(tmp);
      if (fn && (fp = fopen(fn, "w")) != NULL)
      {
         for (j = 0; j < 8; j++)
         {
            fprintf(fp, "%f %f\n", tg->gs[i]->rr[j][0], tg->gs[i]->rr[j][1]);
         }
         fprintf(fp, "%f %f\n", tg->gs[i]->rr[0][0], tg->gs[i]->rr[0][1]);
         fclose(fp);
      }
   }
   WriteTGrid(tg, "tgrid");
   WriteTBoundaryConditions(tg, "tgrid");
}


#ifdef   MAIN_DT_GRID
int main(int argc, char **argv)
{
   int status = 0;
   struct tube *tu;
   struct tgrid *gr;

   if (argc != 4)
   {
      fprintf(stderr, "usage(): dt_grid <conifg.cfg> <grid.txt> <bc.file>\n");
      exit(5);
   }
   if ((tu = ReadTube(argv[1])) != NULL)
   {
      CalcTubeGeometry(tu);
      if ((gr = CreateTGrid(tu)) != NULL)
      {
         if (!WriteTGrid(gr, argv[2]))
            status = 3;
         else
         {
            if (WriteTBoundaryConditions(gr, argv[3]))
               status = 4;
         }
      }
      else
         status = 2;
   }
   else
      status = 1;

   return status;
}
#endif
