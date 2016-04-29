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

#ifdef DEBUG_OUTLET
#define VPRINT(x) fprintf(stderr,"%10s = [%4f, %4f, %4f]\n",#x,x[0], x[1], x[2])
#define VPRINTF(x,f) fprintf(f,"%10s = [%4f, %4f, %4f]\n",#x,x[0], x[1], x[2])
#endif

int MeshRR_OutletRegion(struct Nodelist *n, struct curve *ml, struct region *reg, struct region *reg1,
struct region *reg4, struct region *reg2, struct Ilist *outlet)
{
   int i, j, ii1;
   int offset, newnodes;

   float u1[3], u2[3];
   float p[3];
   float para;

   struct node **tmpnode = NULL;

#ifdef DEBUG_OUTLET
   char fn[111];
   FILE *fp;
   static int count = 0;

   sprintf(fn,"rr_debugoutlet_%02d.txt", count++);
   if( (fp = fopen(fn,"w+")) == NULL)
   {
      fprintf(stderr,"Shit happened opening file '%s'!\n",fn);
      exit(-1);
   }
   fprintf(fp," MeshRR_OutletRegion %d\n",count);
#endif

   for(i = 0; i < reg->numl; i++)
   {
      if(reg->nodes[i])
      {
         FreeIlistStruct(reg->nodes[i]);
         reg->nodes[i] = NULL;
      }
   }
   if(reg->nodes[reg->numl])
   {
      FreeIlistStruct(reg->nodes[reg->numl]);
      reg->nodes[reg->numl+1] = NULL;
   }

   offset = n->num;

   // boundaries, 1st part, reg->nodes[1] needed later
   reg->nodes[0] = CopyIlistStruct(reg2->nodes[3]);
   reg->nodes[reg->numl] = CopyIlistStruct(reg2->nodes[3]);
   // suct. side
   ii1 = (reg1->nodes[2]->num + reg4->nodes[2]->num - 1)
      - reg->arc[1]->num;
   reg->nodes[1] = AllocIlistStruct(reg->line[1]->nump+1);
   for(i = ii1; i < (reg1->nodes[2]->num); i++)
   {
      Add2Ilist(reg->nodes[1], reg1->nodes[2]->list[i]);
   }
   for(i = 1; i < reg4->nodes[2]->num; i++)
   {
      Add2Ilist(reg->nodes[1], reg4->nodes[2]->list[i]);
   }

   // new nodes
   u1[2] = u2[2] = p[2] = 0.0;
   for(i = 1; i < reg->line[2]->nump; i++)
   {
      u1[0] = reg->arc[2]->list[i] - reg->arc[1]->list[i];
      u1[1] = reg->line[2]->y[i]   - reg->line[1]->y[i];
      u2[0] = reg->arc[1]->list[i];
      u2[1] = reg->line[1]->y[i];
      Add2Ilist(reg->nodes[reg->numl], reg->nodes[1]->list[i]);
      for(j = 1; j < reg->line[0]->nump; j++)
      {
         para = (1.0 - reg->para[2]->list[i]) * reg->para[0]->list[j]
            + reg->para[2]->list[i] * reg->para[3]->list[j];
         p[0] = u1[0] * para + u2[0];
         p[1] = u1[1] * para + u2[1];
         Add2Ilist(reg->nodes[reg->numl], n->num);
         AddVNode(n, p, ARC);
      }
   }
   newnodes = n->num - offset;
   CalcNodeRadius(&n->n[n->num-newnodes], ml, newnodes);
   CalcNodeAngle(&n->n[n->num-newnodes], ml, newnodes);

   // boundaries, 2nd part
   // pres. side
   reg->nodes[2] = AllocIlistStruct(reg->line[2]->nump+1);
   Add2Ilist(reg->nodes[2], reg->nodes[0]->list[reg->nodes[0]->num-1]);
   tmpnode = n->n + offset + reg->line[0]->nump-2;
   for(i = 1; i < reg->line[2]->nump; i++)
   {
      Add2Ilist(reg->nodes[2], (*tmpnode)->index);
      tmpnode += reg->line[0]->nump-1;
   }
   // outlet
   reg->nodes[3] = AllocIlistStruct(reg->line[3]->nump+1);
   Add2Ilist(reg->nodes[3], reg->nodes[1]->list[reg->nodes[1]->num-1]);
   tmpnode = n->n + (n->num - reg->line[3]->nump+1);
   for(i = 1; i < reg->line[3]->nump; i++)
   {
      Add2Ilist(outlet, (*tmpnode)->index);
      Add2Ilist(reg->nodes[3], (*tmpnode)->index);
      tmpnode++;
   }

#ifdef DEBUG_OUTLET
   fprintf(fp,"outlet\n");
   DumpIlist2File(outlet, fp);
   for(i = 0; i < reg->numl+1; i++)
   {
      fprintf(fp," ** reg->nodes[%d] **\n", i);
      DumpIlist2File(reg->nodes[i], fp);
   }
#endif

#ifdef DEBUG_OUTLET
   fclose(fp);
#endif

   return(0);
}
