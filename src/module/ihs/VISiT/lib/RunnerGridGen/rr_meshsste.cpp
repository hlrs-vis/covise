#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../General/include/flist.h"
#include "../General/include/points.h"
#include "../General/include/curve.h"
#include "../General/include/nodes.h"
#include "../General/include/fatal.h"
#include "../General/include/v.h"

#include "include/rr_grid.h"

#ifndef SMALL
#define SMALL 0.0001
#endif
#ifndef ABS
#define ABS(a)    ( (a) >= (0) ? (a) : -(a) )
#endif

#ifdef DEBUG_SSTE
#define VPRINT(x) fprintf(stderr,"%10s = [%4f, %4f, %4f]\n",#x,x[0], x[1], x[2])
#define VPRINTF(x,f) fprintf(f,"%10s = [%4f, %4f, %4f]\n",#x,x[0], x[1], x[2])
#endif

int   MeshRR_SSTERegion(struct Nodelist *n, struct curve *ml, struct region *reg, struct region *reg1,
struct Ilist *ssnod, struct Ilist *sste, struct Ilist *outlet)
{

   int i, j;
   int offset, newnodes;                          // here current region starts

   float para;
   float u1[3], u2[3];
   float p[3];

   struct node **tmpnode = NULL;

#ifdef DEBUG_SSTE
   char fn[111];
   FILE *fp;
   static int count = 0;

   sprintf(fn,"rr_debugsste_%02d.txt", count++);
   if( (fp = fopen(fn,"w+")) == NULL)
   {
      fprintf(stderr,"Shit happened opening file '%s'!\n",fn);
      exit(-1);
   }
   fprintf(fp," MeshRR_SSTERegion %d\n",count);
#endif

   // mem. check
   for(i = 0; i < reg->numl; i++)
   {
      if(reg->nodes[i])
      {
         FreeIlistStruct(reg->nodes[i]);
         reg->nodes[i] = NULL;
      }
      reg->nodes[i] = AllocIlistStruct(reg->line[i]->nump+1);
   }
   if(reg->nodes[reg->numl])
   {
      FreeIlistStruct(reg->nodes[reg->numl]);
      reg->nodes[reg->numl+1] = NULL;
   }
   reg->nodes[reg->numl] = AllocIlistStruct(reg->line[0]->nump * reg->line[1]->nump + 1);

   offset = n->num;

   // trailing edge
   for(i = 0; i < (reg->line[0]->nump - reg1->line[3]->nump); i++)
   {
      Add2Ilist(ssnod, n->num);
      Add2Ilist(reg->nodes[0], n->num);
      Add2Ilist(reg->nodes[reg->numl], n->num);
      AddNode(n, reg->arc[0]->list[i], reg->line[0]->y[i], reg->line[0]->z[i], ARC);
   }
   // boundaries
   for(i = reg1->line[3]->nump-1; i >= 0; i--)
   {
      Add2Ilist(reg->nodes[0], reg1->nodes[3]->list[i]);
      Add2Ilist(reg->nodes[reg->numl], reg1->nodes[3]->list[i]);
   }

   // new nodes
   u1[2] = u2[2] = p[2] = 0.0;
   for(i = 1; i < reg->line[1]->nump; i++)
   {
      u1[0] = reg->arc[2]->list[i] - reg->arc[1]->list[i];
      u1[1] = reg->line[2]->y[i]   - reg->line[1]->y[i];
      u2[0] = reg->arc[1]->list[i];
      u2[1] = reg->line[1]->y[i];
      for(j = 0; j < reg->line[0]->nump; j++)
      {
         para = (1.0 - reg->para[1]->list[i])*reg->para[0]->list[j]
            + reg->para[1]->list[i]*reg->para[3]->list[j];
         p[0] = u1[0]*para + u2[0];
         p[1] = u1[1]*para + u2[1];
         Add2Ilist(reg->nodes[reg->numl], n->num);
         AddVNode(n, p, ARC);
      }
   }
   newnodes = n->num - offset;
   CalcNodeRadius(&n->n[n->num-newnodes], ml, newnodes);
   CalcNodeAngle(&n->n[n->num-newnodes], ml, newnodes);

   tmpnode = n->n + offset;
   if(reg->line[0]->nump == reg1->line[3]->nump)  // te-thickness = 0
      Add2Ilist(sste,reg->nodes[0]->list[0]);
   Add2Ilist(sste,(*tmpnode)->index);
   Add2Ilist(reg->nodes[1],(*tmpnode)->index);
   tmpnode = n->n + offset + reg->line[0]->nump - reg1->line[3]->nump;
   for(i = 1; i < reg->line[1]->nump; i++)
   {
      Add2Ilist(sste,(*tmpnode)->index);
      Add2Ilist(reg->nodes[1],(*tmpnode)->index);
      tmpnode += reg->line[0]->nump;
   }
   Add2Ilist(reg->nodes[2], reg1->nodes[3]->list[0]);
   tmpnode  = n->n + offset + reg->line[0]->nump - reg1->line[3]->nump;
   tmpnode += reg->line[0]->nump-1;
   for(i = 1; i < reg->line[2]->nump; i++)
   {
      Add2Ilist(reg->nodes[2], (*tmpnode)->index);
      tmpnode += reg->line[0]->nump;
   }
   tmpnode = n->n + (n->num - reg->line[3]->nump);
   for(i = 0; i < reg->line[3]->nump; i++)
   {
      Add2Ilist(outlet, (*tmpnode)->index);
      Add2Ilist(reg->nodes[3], (*tmpnode)->index);
      tmpnode++;
   }

#ifdef DEBUG_SSTE
   fprintf(fp,"ssnod\n");
   DumpIlist2File(ssnod, fp);
   fprintf(fp,"sste\n");
   DumpIlist2File(sste, fp);

   for(i = 0; i < reg->numl+1; i++)
   {
      fprintf(fp," ** reg->nodes[%d] **\n", i);
      DumpIlist2File(reg->nodes[i], fp);
   }
#endif

#ifdef DEBUG_SSTE
   fclose(fp);
#endif

   return(0);
}
