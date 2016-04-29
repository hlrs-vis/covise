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

#ifdef DEBUG_PSTE
#define VPRINT(x) fprintf(stderr,"%10s = [%4f, %4f, %4f]\n",#x,x[0], x[1], x[2])
#define VPRINTF(x,f) fprintf(f,"%10s = [%4f, %4f, %4f]\n",#x,x[0], x[1], x[2])
#endif

int MeshRR_PSTERegion(struct Nodelist *n, struct curve *ml, struct region *reg, struct region *reg3,
struct region *reg5, struct Ilist *psnod, struct Ilist *pste, struct Ilist *outlet)
{

   int i, j;
   int offset, newnodes;                          // here current region starts

   float para;
   float u1[3], u2[3];
   float p[3];

   struct node **tmpnode = NULL;

#ifdef DEBUG_PSTE
   char fn[111];
   FILE *fp;
   static int count = 0;

   sprintf(fn,"rr_debugpste_%02d.txt", count++);
   if( (fp = fopen(fn,"w+")) == NULL)
   {
      fprintf(stderr,"Shit happened opening file '%s'!\n",fn);
      exit(-1);
   }
   fprintf(fp," MeshRR_PSTERegion %d\n",count);
#endif

   // mem. check
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
   reg->nodes[reg->numl] = AllocIlistStruct(reg->line[0]->nump * reg->line[1]->nump + 1);

   offset = n->num;

   // trailing edge boundary line, first part
   reg->nodes[0] = AllocIlistStruct(reg->line[0]->nump+1);
   for(i = 0; i < reg3->nodes[3]->num; i++)
   {
      Add2Ilist(reg->nodes[0], reg3->nodes[3]->list[i]);
      Add2Ilist(reg->nodes[reg->numl], reg3->nodes[3]->list[i]);
#ifdef DEBUG_PSTE
      fprintf(fp,"i: %d, reg3->nodes[3]->list[i] = %d\n",i,reg3->nodes[3]->list[i]);
#endif
   }

   // new nodes
   for(i = reg3->line[3]->nump; i < reg->line[0]->nump; i++)
   {
      Add2Ilist(psnod, n->num);
      Add2Ilist(reg->nodes[0], n->num);
      Add2Ilist(reg->nodes[reg->numl], n->num);
      AddNode(n, reg->arc[0]->list[i], reg->line[0]->y[i], 0.0, ARC);
   }

   // create nodes
   u1[2] = u2[2] = p[2] = 0.0;
   for(i = 1; i < reg->line[1]->nump; i++)
   {
      u1[0] = reg->arc[2]->list[i] - reg->arc[1]->list[i];
      u1[1] = reg->line[2]->y[i]   - reg->line[1]->y[i];
      u2[0] = reg->arc[1]->list[i];
      u2[1] = reg->line[1]->y[i];
      Add2Ilist(reg->nodes[reg->numl], reg5->nodes[2]->list[i]);
      for(j = 1; j < reg->line[0]->nump; j++)
      {
         para = (1.0 - reg->para[1]->list[i]) * reg->para[0]->list[j]
            + reg->para[1]->list[i] * reg->para[3]->list[j];
#ifdef DEBUG_PSTE
         fprintf(fp,"para = %f\n",para);
#endif
         p[0] = u1[0] * para + u2[0];
         p[1] = u1[1] * para + u2[1];
         Add2Ilist(reg->nodes[reg->numl], n->num);
         AddVNode(n, p, ARC);
      }
   }
   newnodes = n->num - offset;
   CalcNodeRadius(&n->n[n->num-newnodes], ml, newnodes);
   CalcNodeAngle(&n->n[n->num-newnodes], ml, newnodes);

   // boundaries
   // inner boundary
   reg->nodes[1] = CopyIlistStruct(reg5->nodes[2]);
   // outer boundary
   reg->nodes[2] = AllocIlistStruct(reg->line[2]->nump+1);
   Add2Ilist(pste, reg->nodes[0]->list[reg->nodes[0]->num-1]);
   Add2Ilist(reg->nodes[2], reg->nodes[0]->list[reg->nodes[0]->num-1]);
   tmpnode = n->n + (offset + 2*reg->line[0]->nump - reg3->line[3]->nump-2);
   for(i = 1; i < reg->line[2]->nump; i++)
   {
      Add2Ilist(pste, (*tmpnode)->index);
      Add2Ilist(reg->nodes[2], (*tmpnode)->index);
      tmpnode += reg->line[0]->nump-1;
   }
   // outlet
   reg->nodes[3] = AllocIlistStruct(reg->line[3]->nump+1);
   Add2Ilist(reg->nodes[3], reg5->nodes[3]->list[reg5->nodes[3]->num-1]);
   tmpnode = n->n + (n->num - reg->line[3]->nump+1);
   for(i = 1; i < reg->line[3]->nump; i++)
   {
      Add2Ilist(outlet,(*tmpnode)->index);
      Add2Ilist(reg->nodes[3],(*tmpnode)->index);
      tmpnode++;
   }

#ifdef DEBUG_PSTE
   fprintf(fp,"psnod\n");
   DumpIlist2File(psnod, fp);
   fprintf(fp,"pste\n");
   DumpIlist2File(pste, fp);
   fprintf(fp,"outlet\n");
   DumpIlist2File(outlet, fp);
   for(i = 0; i < reg->numl+1; i++)
   {
      fprintf(fp," ** reg->nodes[%d] **\n", i);
      DumpIlist2File(reg->nodes[i], fp);
   }
#endif

#ifdef DEBUG_PSTE
   fclose(fp);
#endif

   return(0);
}
