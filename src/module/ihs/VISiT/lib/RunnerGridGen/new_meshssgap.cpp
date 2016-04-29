#ifdef GAP

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../General/include/flist.h"
#include "../General/include/points.h"
#include "../General/include/curve.h"
#include "../General/include/nodes.h"
#include "../General/include/fatal.h"
#include "../General/include/v.h"
#include "../General/include/plane_geo.h"

#include "include/rr_grid.h"

#define PARA_FINE_LE 0.03
#define PARA_FINE_TE 0.75
#define FINE  5
#define SHIFT 0.5
#define INTERFACE_SHIFT 0.25
#define DELTA 0.7                                 // < 1.0 !!
#define REFINE 1
#define COARSEN -1

#define EVEN 0
#define ODD  1

#ifndef SMALL
#define SMALL 0.0001
#endif
#ifndef ABS
#define ABS(a)    ( (a) >= (0) ? (a) : -(a) )
#endif

#ifdef DEBUG_SSGAP
#define VPRINT(x) fprintf(stderr,"%10s = [%4f, %4f, %4f]\n",#x,x[0], x[1], x[2])
#define VPRINTF(x,f) fprintf(f,"%10s = [%4f, %4f, %4f]\n",#x,x[0], x[1], x[2])
#endif

// meshes the ss-part between shroud and tip. will start with three nodes
// across the blade and refine it to FINE (ODD), starting at the blade
// curve para. PARA_FINE_LE up to PARA_FINE_TE, then we'll reduce the number
// of nodes to the number of nodes on the te-line
int MeshNew_SSGapRegion(struct Nodelist *n, struct curve *ml,
struct Ilist *sste, struct Ilist *ssnod,
struct region *reg, struct region *reg1,
struct region *reg4, int itip)
{
   int i, j;
   int offset, newnodes;
   int first, last, fine_num, coarse_num = 3;
   int steps4reduc, steps4refin;
   static int ifine[4];

   float u1[3], u2[3];
   float p1[3], p2[3], p[3];
   float alpha, beta, scale, shift, ratio;

   struct Ilist *regnodes;

   int OddExamine(int n);
   int RefineChord(struct Flist *arc1, struct Flist *arc2,
      struct Point *line1, struct Point *line2,
      struct Nodelist *n, struct Ilist *regnodes,
      struct Ilist *nodes0, struct Ilist *nodes1,
      struct Ilist *nodes2, float *p, int istart,
      int iend, int fine, int flag
   #ifdef DEBUG_SSGAP
      , FILE *fp
   #endif
      );

#ifdef DEBUG_SSGAP
   char fn[111];
   FILE *fp;
   static int count = 0;

   sprintf(fn,"rr_debugssgap_%02d.txt", count++);
   if( (fp = fopen(fn,"w+")) == NULL)
   {
      fprintf(stderr,"Shit happened opening file '%s'!\n",fn);
      exit(-1);
   }
   fprintf(fp,"# MeshNew_SSGapRegion %d\n",count);
   fprintf(stderr," MeshNew_SSGapRegion %d\n",count);
#endif

   // mem. check
   for(i = 1; i < reg->numl; i++)
   {
      if(reg->nodes[i])
      {
         FreeIlistStruct(reg->nodes[i]);
         reg->nodes[i] = NULL;
      }
      reg->nodes[i] = AllocIlistStruct(reg->line[i]->nump+1);
   }
   // nodes[0] will be used to store discretization ratios!
   if(reg->nodes[0])
   {
      FreeIlistStruct(reg->nodes[0]);
      reg->nodes[0] = NULL;
   }
   reg->nodes[0] = AllocIlistStruct(reg->line[1]->nump);
   // domain nodes, all of them.
   if(reg->nodes[reg->numl])
   {
      FreeIlistStruct(reg->nodes[reg->numl]);
      reg->nodes[reg->numl+1] = NULL;
   }
   reg->nodes[reg->numl] = AllocIlistStruct(reg->line[3]->nump * reg->line[1]->nump);
   regnodes = reg->nodes[reg->numl];

   offset = n->num;

#ifdef DEBUG_SSGAP
   fprintf(fp,"# offset = %d\n", offset);
#endif

   // **************************************************
   reg->nodes[2] = CopyIlistStruct(reg1->nodes[1]);
   reg->nodes[3] = nCopyIlistStruct(reg4->nodes[0], reg->line[3]->nump);
#ifdef DEBUG_SSGAP
   fprintf(fp,"# reg->line[3]->nump = %d\n",reg->line[3]->nump);
#endif

   Add2Ilist(reg->nodes[1], reg->nodes[2]->list[0]);
   Add2Ilist(regnodes, reg->nodes[2]->list[0]);
   Add2Ilist(reg->nodes[0], 1);

   // get node index for refining domain (more nodes on chord), only once!
   if(itip == 0)
   {
      if( (ifine[0] = GetPointIndex(reg->para[1]->num, reg->para[1]->list, PARA_FINE_LE, 0))
         < coarse_num)
      {
         ifine[0] = coarse_num;
      }
      // get node index for reducing number of nodes per chord
      ifine[2] = GetPointIndex(reg->para[1]->num, reg->para[1]->list, PARA_FINE_TE, ifine[0]);
      ifine[3] = ifine[2];

      // middle value
      ifine[1] = (ifine[2] + ifine[0])/2;
   }                                              // end itip == 0
   // get and check parameters for mesh refinement
   fine_num = FINE;
   if(fine_num < reg->line[3]->nump) fine_num = reg->line[3]->nump;
   if(OddExamine(fine_num) == EVEN) fine_num++;
   steps4refin = (fine_num - coarse_num)/2;       // reduce by 2 nodes per step
   if((steps4reduc = (fine_num - reg->line[3]->nump)/2)
      > reg->line[2]->nump - ifine[2] - 2)
   {
      fatal("not enough nodes alongside blade surface!");
      exit(-1);
   }

#ifdef DEBUG_SSGAP
   fprintf(fp,"# ifine[0], ifine[1], ifine[2]: %d, %d, %d\n",
      ifine[0], ifine[1], ifine[2]);
#endif
   u1[2] = u2[2] = 0.0;
   p1[2] = p2[2] = p[2] = 0.0;
   // **************************************************
   // create nodes on line between cl and blade surface
   // coarse region (3 nodes p. chord), next to leading edge
   u1[0] = reg->arc[2]->list[1] - reg->arc[1]->list[1];
   u1[1] = reg->line[2]->y[1]   - reg->line[1]->y[1];
   p2[0] = reg->arc[1]->list[1] + 0.5 * u1[0];
   p2[1] = reg->line[1]->y[1]   + 0.5 * u1[1];
#ifdef DEBUG_SSGAP
   fprintf(fp," %f   %f\n", reg->arc[1]->list[0], reg->line[1]->y[0]);
   fprintf(fp," %f   %f\n\n", reg->arc[1]->list[1], reg->line[1]->y[1]);
   fprintf(fp," %f   %f\n", reg->arc[2]->list[0], reg->line[2]->y[0]);
   fprintf(fp," %f   %f\n", reg->arc[2]->list[1], reg->line[2]->y[1]);
#endif
   for(i = 2; i <= ifine[0]; i++)
   {
      scale = (float)(i-2)/(float)(ifine[0]-2);
      shift = (1.0 - scale) * SHIFT;
      u1[0] = reg->arc[2]->list[i] - reg->arc[1]->list[i];
      u1[1] = reg->line[2]->y[i]   - reg->line[1]->y[i];
      p1[0] = reg->arc[1]->list[i] + 0.5 * u1[0];
      p1[1] = reg->line[1]->y[i]   + 0.5 * u1[1];
      p[0]  = p2[0] + shift * (p1[0] - p2[0]);
      p[1]  = p2[1] + shift * (p1[1] - p2[1]);
      p2[0] = p1[0];
      p2[1] = p1[1];
      Add2Ilist(reg->nodes[0], 3);
      // node on cl
      Add2Ilist(regnodes, n->num);
      Add2Ilist(reg->nodes[1], n->num);
      AddNode(n, reg->arc[1]->list[i-1], reg->line[1]->y[i-1], 0.0, ARC);
      // new node
      Add2Ilist(regnodes, n->num);
      AddVNode(n, p, ARC);
      // node on blade surface
      Add2Ilist(regnodes, reg->nodes[2]->list[i-1]);

#ifdef DEBUG_SSGAP
      fprintf(fp,"# scale, shift: %f, %f\n", scale, shift);
      fprintf(fp," %f   %f\n", p[0], p[1]);
      fprintf(fp," %f   %f\n", reg->arc[1]->list[i-1], reg->line[1]->y[i-1]);
      fprintf(fp," %f   %f\n\n", reg->arc[1]->list[i], reg->line[1]->y[i]);
      fprintf(fp," %f   %f\n", reg->arc[2]->list[i], reg->line[2]->y[i]);
      fprintf(fp," %f   %f\n", reg->arc[2]->list[i-1], reg->line[2]->y[i-1]);
      fprintf(fp," %f   %f\n", p[0], p[1]);
#endif
   }                                              // i < ifine[0]

   // **************************************************
   // refine
   steps4refin += ifine[0];
   RefineChord(reg->arc[1], reg->arc[2], reg->line[1], reg->line[2],
      n, regnodes, reg->nodes[0], reg->nodes[1], reg->nodes[2],
      p, ifine[0], steps4refin, coarse_num, REFINE
   #ifdef DEBUG_SSGAP
      , fp
   #endif
      );

   // mesh refined region
   first = 1;
   last  = fine_num - 1;
   alpha = M_PI / (float)(last);
   beta  = M_PI / (float)(ifine[2]-steps4refin-1);
   for(i = steps4refin; i < ifine[2]; i++)
   {
      u1[0] = reg->arc[2]->list[i] - reg->arc[1]->list[i];
      u1[1] = reg->line[2]->y[i]   - reg->line[1]->y[i];
      p1[0] = reg->arc[1]->list[i];
      p1[1] = reg->line[1]->y[i];
      Add2Ilist(reg->nodes[0], fine_num);
      Add2Ilist(regnodes, n->num);
      Add2Ilist(reg->nodes[1], n->num);
      AddVNode(n, p1, ARC);
      ratio = sin(beta*(float)(i-steps4refin));
#ifdef DEBUG_SSGAP
      fprintf(fp,"\n# i: %d, ratio = %f\n",i, ratio);
      fprintf(fp," %f   %f\n", p1[0], p1[1]);
#endif
      for(j = first; j < last; j++)
      {
         scale = (1.0 - ratio) * (0.5 * (1.0- cos(alpha*j))) +
            ratio * (float)(j)/(float)(last);
         p[0]  = p1[0] + u1[0] * scale;
         p[1]  = p1[1] + u1[1] * scale;
         Add2Ilist(regnodes, n->num);
         AddVNode(n, p, ARC);
#ifdef DEBUG_SSGAP
         fprintf(fp," %f   %f\n", p[0], p[1]);
#endif
      }
      Add2Ilist(regnodes, reg->nodes[2]->list[i]);
#ifdef DEBUG_SSGAP
      fprintf(fp," %f   %f\n",
         reg->arc[2]->list[i], reg->line[2]->y[i]);
#endif
   }
   // chord center point
   p[0]  = p1[0] + u1[0] * 0.5;
   p[1]  = p1[1] + u1[1] * 0.5;

   // **************************************************
   // coarsening mesh to number of nodes on te.
   // make number of nodes even, if necessary
   if(OddExamine(fine_num - reg->line[3]->nump) == ODD)
   {
#ifdef DEBUG_SSGAP
      fprintf(fp,"\n# fine_num - reg->line[3]->nump = %d\n",
         fine_num - reg->line[3]->nump);
      fprintf(fp,"# --> make number EVEN!\n");
#endif
      // make number of nodes on chord EVEN
      fine_num--;                                 // make fine_num EVEN

      // interface layer
      i = ifine[2]-1;
      shift = 1.0 - INTERFACE_SHIFT;
      fprintf(fp,"# interface nodes\n");
      p1[0] = reg->arc[1]->list[i]
         + shift * (reg->arc[1]->list[ifine[2]]
         - reg->arc[1]->list[i]);
      p1[1] = reg->line[1]->y[i]
         + shift * (reg->line[1]->y[ifine[2]]
         - reg->line[1]->y[i]);
      u1[0] = DELTA*(reg->arc[2]->list[ifine[2]]
         - reg->arc[1]->list[ifine[2]]);
      u1[1] = DELTA*(reg->line[2]->y[ifine[2]]
         - reg->line[1]->y[ifine[2]]);

      Add2Ilist(reg->nodes[0], fine_num);
      Add2Ilist(regnodes, n->num);
      Add2Ilist(reg->nodes[1], n->num);           // extra node on cl!
      AddVNode(n, p1, ARC);
      last = fine_num-1;
#ifdef DEBUG_SSGAP
      fprintf(fp,"# fine_num = %d\n",fine_num);
      fprintf(fp," %f   %f\n", p1[0], p1[1]);
#endif
      for(i = 1; i < fine_num; i++)
      {
         scale = (float)(i)/(float)(last);
         p[0] = p1[0] + scale*u1[0];
         p[1] = p1[1] + scale*u1[1];
         Add2Ilist(regnodes, n->num);
         AddVNode(n, p, ARC);
#ifdef DEBUG_SSGAP
         fprintf(fp," %f   %f\n", p[0], p[1]);
#endif
      }

      // next chord (EVEN number of nodes)
      fprintf(fp,"\n# new chord (no. %d = ifine[2])\n",ifine[2]);
      p1[0] = reg->arc[1]->list[ifine[2]];
      p1[1] = reg->line[1]->y[ifine[2]];
      u1[0] = reg->arc[2]->list[ifine[2]]
         - reg->arc[1]->list[ifine[2]];
      u1[1] = reg->line[2]->y[ifine[2]]
         - reg->line[1]->y[ifine[2]];
      Add2Ilist(reg->nodes[0], fine_num);
      Add2Ilist(regnodes, n->num);
      Add2Ilist(reg->nodes[1], n->num);
      AddVNode(n, p1, ARC);
      alpha = M_PI / (float)(last);
#ifdef DEBUG_SSGAP
      fprintf(fp," %f   %f\n", p1[0], p1[1]);
#endif
      for(i = 1; i < last; i++)
      {
         scale = 0.5 * (1.0 - cos(alpha*i));
         p[0]  = p1[0] + u1[0] * scale;
         p[1]  = p1[1] + u1[1] * scale;
         Add2Ilist(regnodes, n->num);
         AddVNode(n, p, ARC);
#ifdef DEBUG_SSGAP
         fprintf(fp," %f   %f\n", p[0], p[1]);
#endif
      }
      Add2Ilist(regnodes, reg->nodes[2]->list[ifine[2]]);
#ifdef DEBUG_SSGAP
      fprintf(fp," %f   %f\n",
         reg->arc[2]->list[ifine[2]], reg->line[2]->y[ifine[2]]);
#endif

      ifine[2]++;
      p[0] = p1[0] + u1[0] * 0.5;                 // chord center
      p[1] = p1[1] + u1[1] * 0.5;
   }

   // **************************************************
   // reduce number of nodes on chord to line[3]->nump
   steps4reduc += ifine[2];
#ifdef DEBUG_SSGAP
   fprintf(fp,"\n# steps4reduc = %d\n",steps4reduc);
#endif
   RefineChord(reg->arc[1], reg->arc[2], reg->line[1], reg->line[2],
      n, regnodes, reg->nodes[0], reg->nodes[1], reg->nodes[2],
      p, ifine[2], steps4reduc, fine_num, COARSEN
   #ifdef DEBUG_SSGAP
      , fp
   #endif
      );

   // **************************************************
   // mesh rest of blade to trailing edge
   fine_num = reg->line[3]->nump;
   first = 1;
   last  = fine_num - 1;
   for(i = steps4reduc; i < reg->line[2]->nump-1; i++)
   {
      u1[0] = reg->arc[2]->list[i] - reg->arc[1]->list[i];
      u1[1] = reg->line[2]->y[i]   - reg->line[1]->y[i];
      p1[0] = reg->arc[1]->list[i];
      p1[1] = reg->line[1]->y[i];
      Add2Ilist(reg->nodes[0], fine_num);
      Add2Ilist(regnodes, n->num);
      Add2Ilist(reg->nodes[1], n->num);
      AddVNode(n, p1, ARC);
#ifdef DEBUG_SSGAP
      fprintf(fp,"\n# i: %d\n",i);
      fprintf(fp," %f   %f\n", p1[0], p1[1]);
#endif
      for(j = first; j < last; j++)
      {
         scale = (float)(j)/(float)(last);
         p[0]  = p1[0] + u1[0] * scale;
         p[1]  = p1[1] + u1[1] * scale;
         Add2Ilist(regnodes, n->num);
         AddVNode(n, p, ARC);
#ifdef DEBUG_SSGAP
         fprintf(fp," %f   %f   %f\n", p[0], p[1], scale);
#endif
      }
      Add2Ilist(regnodes, reg->nodes[2]->list[i]);
#ifdef DEBUG_SSGAP
      fprintf(fp," %f   %f\n",
         reg->arc[2]->list[i], reg->line[2]->y[i]);
#endif
   }                                              // end i

   // trailing edge nodes
   Add2Ilist(reg->nodes[0], reg->nodes[3]->num);
   Add2Ilist(reg->nodes[1], reg->nodes[3]->list[0]);
   for(i = 0; i < reg->nodes[3]->num; i++)
   {
      Add2Ilist(regnodes, reg->nodes[3]->list[i]);
   }
   Add2Ilist(reg->nodes[2], reg->nodes[3]->list[reg->nodes[3]->num-1]);
   // **************************************************
   // nodes for boundary conditions
   // periodic boundary (ss)
   for(i = 0; i < reg->nodes[1]->num; i++)
   {
      Add2Ilist(sste, reg->nodes[1]->list[i]);
   }
   // solid blade surface
   if(itip == 0)
   {
      for(i = 0; i < regnodes->num; i++)
      {
         Add2Ilist(ssnod, regnodes->list[i]);
      }
   }
   // **************************************************
   // final works
   ifine[2] = ifine[3];
   newnodes = n->num - offset;
   CalcNodeRadius(&n->n[n->num-newnodes], ml, newnodes);
   CalcNodeAngle(&n->n[n->num-newnodes], ml, newnodes);
   // **************************************************

#ifdef DEBUG_SSGAP
   fprintf(fp,"newnodes = %d\n",newnodes);
   fprintf(fp,"sste\n");
   DumpIlist2File(sste, fp);
   fprintf(fp,"ssnod\n");
   DumpIlist2File(ssnod, fp);

   for(i = 0; i < reg->numl+1; i++)
   {
      fprintf(fp," ** reg->nodes[%d] **\n", i);
      DumpIlist2File(reg->nodes[i], fp);
   }

   fclose(fp);
#endif
   return(0);
}


// **************************************************
int OddExamine(int n)
{
   if(pow(-1.0, n) < 0.0) return(ODD);
   else return(EVEN);
}


// **************************************************
// change discretization of chords between blade surf. and cl.
// interface nodes in between two chords will be created, there
// must be as many interface nodes as on the coarser meshed chord.
int RefineChord(struct Flist *arc1, struct Flist *arc2,
struct Point *line1, struct Point *line2,
struct Nodelist *n, struct Ilist *regnodes,
struct Ilist *nodes0, struct Ilist *nodes1,
struct Ilist *nodes2, float *p, int istart,
int iend, int fine, int flag
#ifdef DEBUG_SSGAP
, FILE *fp
#endif
)
{
   int i, j, first, last;
   float scale, alpha, shift;
   float p1[3], p2[3], u1[3], u2[3];

   p1[2] = p2[2] = u1[2] = u2[2] = 0.0;
   if(flag == COARSEN) shift = 1.0 - INTERFACE_SHIFT;
   else if(flag == REFINE) shift = INTERFACE_SHIFT;
   else
   {
      fprintf(stderr,"\n RefineChord: unknown flag: %d\n\n",flag);
      exit(-1);
   }
   for(i = istart; i < iend; i++)
   {
      if(flag == COARSEN) fine -= 2;
      u1[0] = arc2->list[i] - arc1->list[i];
      u1[1] = line2->y[i]   - line1->y[i];
      u2[0] = DELTA*u1[0];                        // interface chord
      u2[1] = DELTA*u1[1];                        // vector
      p1[0] = arc1->list[i];                      // cl point
      p1[1] = line1->y[i];
      p2[0] = p1[0] + 0.5*u1[0];                  // chord center
      p2[1] = p1[1] + 0.5*u1[1];
      p1[0] = p[0] + shift * (p2[0] - p[0]);      // center pt.
      p1[1] = p[1] + shift * (p2[1] - p[1]);
      p2[0] = p1[0] - 0.5*u2[0];                  // start pt.
      p2[1] = p1[1] - 0.5*u2[1];
#ifdef DEBUG_SSGAP
      fprintf(fp,"\n# i: %d: interface nodes, fine = %d\n",
         i, fine);
#endif
      // interface nodes
      last  = fine - 1;
      Add2Ilist(nodes0, fine);
      for(j = 0; j < fine; j++)
      {
         scale = (float)(j)/(float)(last);
         p[0] = p2[0] + scale * u2[0];
         p[1] = p2[1] + scale * u2[1];
         Add2Ilist(regnodes, n->num);
         AddVNode(n, p, ARC);
#ifdef DEBUG_SSGAP
         fprintf(fp," %f   %f\n", p[0], p[1]);
#endif
      }
      // new chord
      p1[0] = arc1->list[i];
      p1[1] = line1->y[i];
      Add2Ilist(regnodes, n->num);
      Add2Ilist(nodes1, n->num);
      AddVNode(n, p1, ARC);
#ifdef DEBUG_SSGAP
      fprintf(fp,"\n# chord nodes\n");
      fprintf(fp," %f   %f\n", p1[0], p1[1]);
#endif
      if(flag == REFINE) fine += 2;
      first = 1;
      last  = fine - 1;
      alpha = M_PI / (float)(last);
      Add2Ilist(nodes0, fine);
      for(j = first; j < last; j++)
      {
         scale = 0.5 * (1.0- cos(alpha*j));
         p[0]  = p1[0] + u1[0] * scale;
         p[1]  = p1[1] + u1[1] * scale;
         Add2Ilist(regnodes, n->num);
         AddVNode(n, p, ARC);
#ifdef DEBUG_SSGAP
         fprintf(fp," %f   %f\n", p[0], p[1]);
#endif
      }
      p[0]  = p1[0] + u1[0] * 0.5;
      p[1]  = p1[1] + u1[1] * 0.5;
      Add2Ilist(regnodes, nodes2->list[i]);
#ifdef DEBUG_SSGAP
      fprintf(fp," %f   %f\n",
         arc2->list[i], line2->y[i]);
#endif
   }                                              // end i < steps4refin

   return(0);
}
#endif                                            // GAP
