#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "include/log.h"
#include "include/fatal.h"
#include "include/cfg.h"
#include "include/curve.h"
#include "include/points.h"

struct curve * AllocCurveStruct(void)
{
   struct curve *c;

   if ((c = (struct curve*)calloc(1, sizeof(struct curve))) == NULL)
      fatal("memory for (struct curve *)");
   c->p = AllocPointStruct();

   return(c);
}


int AddCurvePoint(struct curve *c, float x, float y, float z, float len, float par)
{
   int members;

   if (!c || !c->p)
   {
      dprintf(0, ":-( in AddCurvePoint()\n");
      exit(1);
   }
   if (c->p->nump+1 >= c->p->maxp)
   {
      members = c->p->maxp + c->p->portion;
      if ((c->len = (float *)realloc(c->len, members * sizeof(float))) == NULL)
         fatal("memory for (float *)len in struct curve");
      if ((c->par = (float *)realloc(c->par, members * sizeof(float))) == NULL)
         fatal("memory for (float *)par in struct curve");
   }
   AddPoint(c->p, x, y, z);
   if (len)
      c->len[c->p->nump-1] = len;
   else
      c->arclen = 1;
   if (par)
      c->par[c->p->nump-1] = par;
   else
      c->arclen = 1;

   return (c->p->nump-1);
}


int CalcCurveArclen(struct curve *c)
{
   int i;
   float len;

   len = 0.0;
   c->len[0] = c->par[0] = 0.0;
   for (i = 1; i < c->p->nump; i++)
   {
      len += sqrt(pow((c->p->x[i] - c->p->x[i-1]), 2) +
         pow((c->p->y[i] - c->p->y[i-1]), 2) +
         pow((c->p->z[i] - c->p->z[i-1]), 2));
      c->len[i] = len;
   }
   for (i = 1; i < c->p->nump; i++)
      c->par[i] = c->len[i] / len;
   c->arclen = 0;

   return(0);
}


int CalcCurveArclen2(struct curve *c)
{
   // x contains circumferential angle, y the merid. length
   // and z the radius.
   int i;
   float len, arc[2];

   len = 0.0;
   c->len[0] = c->par[0] = 0.0;
   arc[0] = c->p->x[0]*c->p->z[0];
   for (i = 1; i < c->p->nump; i++)
   {
      arc[1] = c->p->x[i]*c->p->z[i];
      len += sqrt(pow((arc[1] - arc[0]), 2) +
         pow((c->p->y[i] - c->p->y[i-1]), 2));
      c->len[i] = len;
      arc[0] = arc[1];
   }
   for (i = 1; i < c->p->nump; i++)
   {
      c->par[i] = c->len[i] / len;
   }
   c->arclen = 0;

   return(0);
}


struct curve *GetCurveMemory(struct curve *c)
{
   if(c)
   {
      FreeCurveStruct(c);
      c = NULL;
   }
   return(AllocCurveStruct());
}


void FreeCurveStruct(struct curve *c)
{
   if (c->p->nump && c->len)  free(c->len);
   if (c->p->nump && c->par)  free(c->par);
   if (c->p->nump && c->p)    FreePointStruct(c->p);
   free(c);
}


void DumpCurve(struct curve *c, FILE *fp)
{
   int j;

   for (j = 0; j < c->p->nump; j++)
   {
      if (fp)
      {
         fprintf(fp, "%8.4f %8.4f %8.4f %8.4f %8.4f\n",
            c->p->x[j], c->p->y[j], c->p->z[j], c->len[j], c->par[j]);
      }
      else
         dprintf(5, "j=%3d: x=%8.4f y=%8.4f z=%8.4f l=%8.4f p=%8.4f\n",
            j, c->p->x[j], c->p->y[j], c->p->z[j], c->len[j], c->par[j]);
   }
}
