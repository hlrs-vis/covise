#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../General/include/flist.h"
#include "../General/include/ilist.h"
#include "../General/include/points.h"
#include "../General/include/nodes.h"
#include "../General/include/elements.h"
#include "../General/include/fatal.h"

#include "include/rr_grid.h"

#ifndef ABS
#define ABS(a)    ( (a) >= (0) ? (a) : -(a) )
#endif
#ifndef SIGN
#define SIGN(a)    ( (a) >= (0) ? (1) : -(1) )
#endif

int *bcnodes;
int eoffset;
int **elem;
int k0;
struct Nodelist *n;

static float getVProd(float *x1, float *x2);
static int SetBCNodes(struct Ilist *nodes);
static int OptimizeLayer(void);

int SmoothRR_Mesh(struct Nodelist *nn, struct Element *e, int ge_num,
struct Ilist *psnod, struct Ilist *ssnod,
struct Ilist *psle, struct Ilist *ssle, struct Ilist *pste,
struct Ilist *sste,struct Ilist *inlet,struct Ilist *outlet)
{
   int i,ig;

   //	return 0;

   // **************************************************
   n = nn;
   if( (bcnodes = (int*)calloc(n->num, sizeof(int))) == NULL)
   {
      fatal("memory for (int)!");
      exit(-1);
   }
   SetBCNodes(psnod);
   SetBCNodes(ssnod);
   SetBCNodes(psle);
   SetBCNodes(ssle);
   SetBCNodes(pste);
   SetBCNodes(sste);
   SetBCNodes(inlet);
   SetBCNodes(outlet);

   eoffset = e->nume / (ge_num-1);
   // **************************************************

   for(i = 0; i < 30; i++)
   {
      fprintf(stderr,"SmoothRR_Mesh(): %d\n",i);
      for(ig = 0; ig < 1; ig++)
      {
         elem = e->e + ig*eoffset;
         OptimizeLayer();
      }                                           // ig
   }

   // **************************************************
   free(bcnodes);
   return 0;
}


static int SetBCNodes(struct Ilist *nodes)
{
   int i;
   for(i = 0; i < nodes->num; i++) bcnodes[nodes->list[i]] = 1;
   return 0;
}


static int OptimizeLayer(void)
{
   int i, j, k, ii[2], jj[2], iicount;

   float l, len, x1[2], x2[2], s[2], dx[2], ds[2];

   for(i = 0; i < eoffset; i++)
   {
      ii[0] = ii[1] = -1;
      iicount = 0;
      len = 0.0;
      for(j = 0; j < 4 && iicount < 2; j++)
      {
         if(!bcnodes[n->n[(*elem)[j]]->index])
         {
            ii[iicount] = n->n[(*elem)[j]]->index;
            jj[iicount] = j;
            iicount++;
         }
         l  = pow(n->n[(*elem)[j]]->arc-
            n->n[(*elem)[(j+1)%4]]->arc,2);
         l += pow(n->n[(*elem)[j]]->l-
            n->n[(*elem)[(j+1)%4]]->l,2);
         len += sqrt(l);
      }
      len *= 0.25;
      dx[0] = dx[1] = 0.02*len;

      for(j = 0; j < iicount; j++)
      {

         for(k = 0; k < 2; k++)
         {
            x1[0] = n->n[(*elem)[(jj[j]+1)%4]]->arc -
               n->n[(*elem)[jj[j]]]->arc;
            x1[1] = n->n[(*elem)[(jj[j]+1)%4]]->l -
               n->n[(*elem)[jj[j]]]->l;
            x2[0] = n->n[(*elem)[(jj[j]-1)%4]]->arc -
               n->n[(*elem)[jj[j]]]->arc;
            x2[1] = n->n[(*elem)[(jj[j]-1)%4]]->l -
               n->n[(*elem)[jj[j]]]->l;

            s[j] = x1[0]*x2[0]+x1[1]*x2[1];

            // deltas
            ds[0] = 2*n->n[(*elem)[jj[j]]]->arc-
               n->n[(*elem)[(jj[j]+1)%4]]->arc-
               n->n[(*elem)[(jj[j]-1)%4]]->arc;
            ds[1] = 2*n->n[(*elem)[jj[j]]]->l-
               n->n[(*elem)[(jj[j]+1)%4]]->l-
               n->n[(*elem)[(jj[j]-1)%4]]->l;
            if( (ds[0]*dx[0]+s[j]) < s[j])
               n->n[(*elem)[jj[j]]]->arc+=dx[0];
            else n->n[(*elem)[jj[j]]]->arc-=dx[0];
            if( (ds[1]*dx[1]+s[j]) < s[j])
               n->n[(*elem)[jj[j]]]->l+=dx[1];
            else n->n[(*elem)[jj[j]]]->l-=dx[1];

         }

      }
      elem++;
   }
   return 0;
}


static float getVProd(float *x1, float *x2)
{
   float vp,l;

   l = 1.0/sqrt(x1[0]*x1[0]+x1[1]*x1[1]);
   x1[0] *= l; x1[1] *= l;
   l = 1.0/sqrt(x2[0]*x2[0]+x2[1]*x2[1]);
   x2[0] *= l; x2[1] *= l;

   vp =  x1[0]*x2[1] - x2[0]*x1[1];

   return vp;
}
