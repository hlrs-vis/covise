#include <stdio.h>
#include <stdlib.h>
#include "include/tgrid.h"
#include "include/tmesh.h"
#include "include/tube.h"
#include "include/t2c.h"
#include "include/transform.h"
#include "../General/include/elements.h"
#include "../General/include/vector.h"
#include "../General/include/fatal.h"
#include "../General/include/common.h"
#include "../General/include/v.h"

static void BuildElement(struct Element *e, int c[4], int nps);
static void Elements4oSection(struct tgrid *tg, int quad);
static void Elements4iSection(struct tgrid *tg, int quad);
static void TransferCSGridPoints(struct cs *cs, struct gs *gs, struct Point *p);

#ifdef   DEBUG
#define  SI(x) fprintf(stderr, "%20s = %d\n", #x, (x))
#define  SF(x) fprintf(stderr, "%20s = %f\n", #x, (x))
#define  SV(x) fprintf(stderr, "%20s = (%f,%f,%f)\n", #x, x[0], x[1], x[2])
#endif                                            // DEBUG

void MeshElemsTube(struct tube *tu, struct tgrid *tg)
{
   int i, j, k;
   struct cs *cs2;
   struct gs *gs1;
   struct gs *gs2;
   float P[3];
   float r[3];
   float r1[3];
   float r2[3];
   struct Point *p1;
   struct Point *p2;
   int elparts;
   int el[4];
   int move;
   int nps;
   int end;
   int offs;
   int num_elemsps;
   int wallelemsps;
   int bc[5];

   // boundary conditions
   tg->in   = AllocVectorStruct(5);
   tg->out  = AllocVectorStruct(5);
   tg->wall = AllocVectorStruct(5);

   // the "0"-section is a special case: we dont mesh anything,
   // but we transfer the grid-points from the cross-section 0
   // in the global grid-point-list
   // in the loop we mesh always from the grid before to the actual ...

   p1 = NULL;
   p2 = AllocPointStruct();
   TransferCSGridPoints(tu->cs[0], tg->gs[0], p2);

   for (elparts = 0, k = 1; k < tu->cs_num; k++)
   {
      cs2 = tu->cs[k];
      gs1 = tg->gs[k-1];
      gs2 = tg->gs[k];

      if (p1)  FreePointStruct(p1);
      p1 = p2;
      p2 = AllocPointStruct();

      TransferCSGridPoints(cs2, gs2, p2);

      end = gs1->num_elems + (k == tu->cs_num-1 ? 1 : 0);
      for (j = 0; j < end; j++)
      {
         elparts++;
         for (i = 0; i < tg->gs[0]->p->nump; i++)
         {
            GetPoint(p1, r1, i);
            GetPoint(p2, r2, i);
            V_Sub(r2, r1, r);
            V_MultScal(r, (float)(j) / (float)(gs1->num_elems));
            V_Add(r1, r, P);
            AddVPoint(tg->p, P);
         }
      }
   }
   elparts--;

   for (i = 0; i < 4; i++)
      Elements4iSection(tg, i);
   for (i = 0; i < 4; i++)
      Elements4oSection(tg, i);
   num_elemsps = tg->e->nume;

   nps = tg->gs[0]->p->nump;
   for (move = 0, i = 0; i < elparts-1; i++)
   {
      move += nps;
      for (j = 0; j < num_elemsps; j++)
      {
         el[0] = tg->e->e[j][0] + move;
         el[1] = tg->e->e[j][1] + move;
         el[2] = tg->e->e[j][2] + move;
         el[3] = tg->e->e[j][3] + move;
         BuildElement(tg->e, el, nps);
      }
   }

   // boundary conditions at the IN section
   for (i = 0; i < num_elemsps; i++)
   {
      bc[0] = tg->e->e[i][0];
      bc[1] = tg->e->e[i][1];
      bc[2] = tg->e->e[i][2];
      bc[3] = tg->e->e[i][3];
      bc[4] = i;
      AddVector(tg->in, bc);
   }
   // boundary conditions at the WALL section
   wallelemsps = tg->wall->numv;
   for (move = 0, i = 0; i < elparts-1; i++)
   {
      move += nps;
      for (j = 0; j < wallelemsps; j++)
      {
         bc[0] = tg->wall->v[j][0] + move;
         bc[1] = tg->wall->v[j][1] + move;
         bc[2] = tg->wall->v[j][2] + move;
         bc[3] = tg->wall->v[j][3] + move;
         bc[4] = tg->wall->v[j][4] + (i+1)*num_elemsps;
         AddVector(tg->wall, bc);
      }
   }

   // boundary conditions at the OUT section
   for (i = 0; i < num_elemsps; i++)
   {
      offs = (elparts-1)*num_elemsps+i;
      bc[0] = tg->e->e[offs][4];
      bc[1] = tg->e->e[offs][5];
      bc[2] = tg->e->e[offs][6];
      bc[3] = tg->e->e[offs][7];
      bc[4] = offs;
      AddVector(tg->out, bc);
   }

   if (p1)  FreePointStruct(p1);
   if (p2)  FreePointStruct(p2);
}


static void BuildElement(struct Element *e, int c[4], int nps)
{
   int E[8];

   E[0] = c[0]; E[1] = c[1]; E[2] = c[2]; E[3] = c[3];
   E[4] = c[0] + nps; E[5] = c[1] + nps; E[6] = c[2] + nps; E[7] = c[3] + nps;
   AddElement(e, E);
}


static void Elements4oSection(struct tgrid *tg, int quad)
{
   int i, j;
   int nquad;
   int cole;
   int rowe;
   int nps;
   int el[4];
   int s1, ss1, s01;
   int s2, ss2, s02;
   int offs;
   int bc[5];

   cole = GetoColElems(tg, quad*2);               // Number of columns of the (right) outer quadrant
   rowe = GetoRowElems(tg);                       // Number of rows in every outer quadrant
   nps = tg->gs[0]->p->nump;                      // Number of points in the complete CS
   //offs = tg->p->nump - 2 * nps;
   offs = 0;

   // first we make the right part of this quadrant
   s1  = offs + tg->numisP[quad];
   ss1 = offs + tg->numosP[quad*2+1] - rowe + 1;
   s01 = offs + tg->numoSP[quad*2];

   for (i = cole; i ; i--)
   {
      el[0] = s1 - (cole - i);
      el[1] = el[0] - 1;
      el[2] = s01 + (cole - i) * rowe;
      el[3] = (i == cole ? ss1 : s01 + (cole - i - 1)*rowe);
      BuildElement(tg->e, el, nps);

      for (j = 0; j < rowe-1; j++)
      {
         el[0] = (i == cole ? ss1 + j : s01 + (cole - i - 1)*rowe + j);
         el[1] = s01 + (cole - i)*rowe + j;
         el[2] = el[1] + 1;
         el[3] = el[0] + 1;
         BuildElement(tg->e, el, nps);
      }
      // Boundary conditions ...
      bc[0] = tg->e->e[tg->e->nume-1][3];
      bc[1] = tg->e->e[tg->e->nume-1][2];
      bc[2] = tg->e->e[tg->e->nume-1][6];
      bc[3] = tg->e->e[tg->e->nume-1][7];
      bc[4] = tg->e->nume-1;
      AddVector(tg->wall, bc);
   }

   nquad = (quad+1 > 3 ? quad+1-4 : quad+1);
   cole = GetoColElems(tg, quad*2+1);             // Number of columns of the (right) outer quadrant
   rowe = GetoRowElems(tg);                       // Number of rows in every outer quadrant
   s02 = offs + tg->numisP[nquad] - GetiRowElems(tg, nquad);
   s2  = offs + tg->numiSP[quad] + GetiRowElems(tg, quad);
   ss2 = offs + tg->numosP[(quad+1 > 3 ? quad+1-4 : quad+1)*2] - rowe + 1;

   for (i = 0; i < cole ; i++)
   {
      if (!i)
      {
         el[0] = s02;
         el[1] = s2;
      }
      else
      {
         el[0] = s2 + (i-1)*GetiRowPoints(tg, quad);
         el[1] = el[0] + GetiRowPoints(tg, quad);
      }
      el[2] = tg->numoSP[quad*2+1]+i*rowe;
      el[3] = (i == 0 ? ss2 : el[2] - rowe);
      BuildElement(tg->e, el, nps);

      for (j = 0; j < rowe-1; j++)
      {
         el[0] = (i == 0 ? ss2 + j : tg->numoSP[quad*2+1] + (i-1)*rowe + j);
         el[1] = tg->numoSP[quad*2+1] + i*rowe + j;
         el[2] = el[1] + 1;
         el[3] = el[0] + 1;
         BuildElement(tg->e, el, nps);
      }
      // Boundary conditions ...
      bc[0] = tg->e->e[tg->e->nume-1][3];
      bc[1] = tg->e->e[tg->e->nume-1][2];
      bc[2] = tg->e->e[tg->e->nume-1][6];
      bc[3] = tg->e->e[tg->e->nume-1][7];
      bc[4] = tg->e->nume-1;
      AddVector(tg->wall, bc);
   }
}


static void Elements4iSection(struct tgrid *tg, int quad)
{
   int i, j;
   int ind1;
   int npq, npi, nps, cole, rowp, rowlp;
   int el[4];
   int s, ss;
   int offs;

   ind1 = ((quad+1) > 3 ? (quad+1)-4 : (quad+1));

   cole  = GetiColElems(tg, quad);                // Number of columns in this section
   rowp  = GetiRowPoints(tg, quad);               // Number of rowpoints in this inner section
   rowlp = GetiRowPoints(tg, ind1);               // Number of rowpoints in the left section
   nps   = tg->gs[0]->p->nump;                    // Number of points in the complete CS
   npq   = tg->numiP[quad];                       // Number of points in this section
   npi   = tg->numisP[3];                         // Number of points in all 4 inner sections
   //offs  = tg->p->nump - 2 * nps;
   offs  = 0;

   // First element is special :-(
   el[0] = offs;
   s = el[1] = el[0] + 1 + (quad ? tg->numisP[quad-1] : 0);
   el[2] = el[1] + 1;
   el[3] = el[1] + npq;
   ss = el[3] = (el[3] - el[0] > npi ? el[3] - npi : el[3]);
   BuildElement(tg->e, el, nps);

   // Now the first left row ...
   for (j = 1; j < rowp-1; j++)
   {
      el[0] = ss + (j-1)*rowlp;
      el[1] = s + j;
      el[2] = el[1] + 1;
      el[3] = el[0] + rowlp;
      BuildElement(tg->e, el, nps);
   }

   // and now the rest of this (inner) Section
   for (i = 1; i < cole; i++)
   {
      for (j = 0; j < rowp-1; j++)
      {
         el[0] = s + j;
         el[1] = el[0] + rowp;
         el[2] = el[1] + 1;
         el[3] = el[0] + 1;
         BuildElement(tg->e, el, nps);
      }
      s += rowp;
   }
}


static void TransferCSGridPoints(struct cs *cs, struct gs *gs, struct Point *p)
{
   int i;
   float r[3];
   float P[3];

   r[0] = cs->c_m_x;
   r[1] = cs->c_m_y;
   r[2] = cs->c_m_z;
   for (i = 0; i < gs->p->nump; i++)
   {
      P[0] = gs->p->x[i];
      P[1] = gs->p->y[i];
      P[2] = gs->p->z[i];
      Transform(cs->T, r, P, P+1, P+2);
      AddVPoint(p, P);
   }
}
