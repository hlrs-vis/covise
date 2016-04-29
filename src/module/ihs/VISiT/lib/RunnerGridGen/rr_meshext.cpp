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

#ifdef DEBUG_INLET
#define VPRINT(x) fprintf(stderr,"%10s = [%4f, %4f, %4f]\n",#x,x[0], x[1], x[2])
#define VPRINTF(x,f) fprintf(f,"%10s = [%4f, %4f, %4f]\n",#x,x[0], x[1], x[2])
#endif

int MeshRR_ExtRegion(struct Nodelist *n, struct curve *ml, struct Ilist *inlet,
struct Ilist *psle, struct Ilist *ssle, struct region *reg, struct region *reg0)
{
   int i, j;
   int offset, newnodes;

   float u1[3], u2[3];
   float p[3];

   struct node **tmpnode = NULL;

#ifdef DEBUG_EXT
   char fn[111];
   FILE *fp;
   static int count = 0;

   sprintf(fn,"rr_debugext_%02d.txt", count++);
   if( (fp = fopen(fn,"w+")) == NULL)
   {
      fprintf(stderr,"Shit happened opening file '%s'!\n",fn);
      exit(-1);
   }
   fprintf(fp," MeshRR_ExtRegion %d\n",count);
#endif

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

   // nodes, alongside line 8.2
   u1[2] = u2[2] = p[2] = 0.0;
   for(i = 0; i < reg->line[0]->nump; i++)
   {
      u1[0] = reg->line[3]->x[i] - reg->line[0]->x[i];
      u1[1] = reg->line[3]->y[i] - reg->line[0]->y[i];
      u2[0] = reg->line[0]->x[i];
      u2[1] = reg->line[0]->y[i];
      for(j = 0; j < reg->line[1]->nump-1; j++)
      {
         p[0] = u2[0] + reg->para[1]->list[j]*u1[0];
         p[1] = u2[1] + reg->para[1]->list[j]*u1[1];
         AddVNode(n, p, PHI);
      }
   }
   newnodes = n->num - offset;
   CalcNodeRadius(&n->n[n->num-newnodes], ml, newnodes);

   for(i = 0; i < reg->line[1]->nump-1; i++)
   {
      tmpnode = n->n + offset + i;
      for(j = 0; j < reg->line[0]->nump; j++)
      {
         Add2Ilist(reg->nodes[reg->numl], (*tmpnode)->index);
         tmpnode += reg->line[1]->nump-1;
      }
   }
   for(i = 0; i < reg->line[0]->nump; i++)
   {
      Add2Ilist(reg->nodes[reg->numl], reg0->nodes[0]->list[i]);
   }

   // get boundary nodes
   tmpnode = n->n + offset;
   for(i = 0; i < reg->line[0]->nump; i++)
   {
      Add2Ilist(inlet, (*tmpnode)->index);
      Add2Ilist(reg->nodes[0], (*tmpnode)->index);
      tmpnode += reg->line[1]->nump-1;
   }
   tmpnode = n->n + offset;
   for(i = 0; i < reg->line[1]->nump-1; i++)
   {
      Add2Ilist(ssle, (*tmpnode)->index);
      Add2Ilist(reg->nodes[1], (*tmpnode)->index);
      tmpnode++;
   }
   //Add2Ilist(ssle, reg0->nodes[0]->list[0]);
   Add2Ilist(reg->nodes[1], reg0->nodes[0]->list[0]);

   tmpnode = n->n + (n->num - reg->line[1]->nump+1);
   for(i = 0; i < reg->line[2]->nump-1; i++)
   {
      Add2Ilist(psle, (*tmpnode)->index);
      Add2Ilist(reg->nodes[2], (*tmpnode)->index);
      tmpnode++;
   }
   //Add2Ilist(psle, reg0->nodes[0]->list[reg0->nodes[0]->num-1]);
   Add2Ilist(reg->nodes[2], reg0->nodes[0]->list[reg0->nodes[0]->num-1]);

   FreeIlistStruct(reg->nodes[3]);
   reg->nodes[3] = CopyIlistStruct(reg0->nodes[0]);

#ifdef DEBUG_EXT
   fprintf(fp,"inlet\n");
   DumpIlist2File(inlet, fp);
   fprintf(fp,"psle\n");
   DumpIlist2File(psle, fp);
   fprintf(fp,"ssle\n");
   DumpIlist2File(ssle, fp);
   for(i = 0; i < reg->numl+1; i++)
   {
      fprintf(fp," ** reg->nodes[%d] **\n", i);
      DumpIlist2File(reg->nodes[i], fp);
   }
#endif

#ifdef DEBUG_EXT
   fclose(fp);
#endif

   return(0);
}
