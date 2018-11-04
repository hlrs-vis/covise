#include <stdio.h>
#include <stdlib.h>
#ifdef WIN32
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#endif
#include <math.h>

#include "../General/include/flist.h"
#include "../General/include/ilist.h"
#include "../General/include/points.h"
#include "../General/include/curve.h"
#include "../General/include/nodes.h"
#include "../General/include/plane_geo.h"
#include "../General/include/curvepoly.h"
#include "../BSpline/include/bspline.h"
#include "../General/include/fatal.h"
#include "../General/include/v.h"

#include "include/rr_grid.h"

#ifndef SMALL
#define SMALL 0.0001
#endif
#ifndef ABS
#define ABS(a)    ( (a) >= (0) ? (a) : -(a) )
#endif

#ifndef BSPLN_DEGREE
#define BSPLN_DEGREE 3
#endif

#ifdef DEBUG_PS
#define VPRINT(x) fprintf(stderr,"%10s = [%4f, %4f, %4f]\n",#x,x[0], x[1], x[2])
#define VPRINTF(x,f) fprintf(f,"%10s = [%4f, %4f, %4f]\n",#x,x[0], x[1], x[2])
#endif

int MeshNew_PSRegion(struct Nodelist *n, struct curve *ml, struct Ilist *psnod,
struct Ilist *psle, struct region *reg, struct region *reg0, int le_dis)
{
   int i, j, jadd;
   int offset, newnodes;                          // here current region starts

   float u1[3], u2[3];
   float p[3];
   float p1[3], p2[3], p3[3];
   float alpha[2], beta[2], delta, para;

   struct node **tmpnode = NULL;

   struct Point *poly;
   struct Flist *knot;

#ifdef DEBUG_PS
   char fn[111];
   FILE *fp;
   static int count = 0;

   sprintf(fn,"rr_debugps_%02d.txt", count++);
   if( (fp = fopen(fn,"w+")) == NULL)
   {
      fprintf(stderr,"Shit happened opening file '%s'!\n",fn);
      exit(-1);
   }
   fprintf(fp," MeshNew_PSRegion %d\n",count);
#endif

   // **************************************************
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

   // **************************************************
   // initialize some params.
   offset = n->num;
   jadd   = reg->line[0]->nump/2;
   p[2] = p1[2] = p2[2] = p3[2] = u1[2] = u2[2] = 0.0;

#ifdef DEBUG_PS
   fprintf(fp," offset = %d\n", offset);
   fprintf(fp," jadd = %d\n",jadd);
#endif

   for(j = reg0->nodes[3]->num-jadd; j < reg0->nodes[3]->num; j++)
   {
#ifdef DEBUG_PS
      fprintf(fp,"j = %3d, reg0->nodes[3]->list[j] = %5d\n",j,reg0->nodes[3]->list[j]);
#endif
      Add2Ilist(reg->nodes[reg->numl], reg0->nodes[3]->list[j]);
   }
   // create nodes, leave out first node.
   for(j = jadd; j < reg->line[0]->nump; j++)
   {
      p[0] = reg->arc[0]->list[j];
      p[1] = reg->line[0]->y[j];
      Add2Ilist(reg->nodes[reg->numl], n->num);
      AddVNode(n, p, ARC);
   }
   // create new nodes.
   // create nodes on spline for i < le_dis,
   // angles for spline
   u1[0] = reg->arc[0]->list[1] - reg->arc[0]->list[0];
   u1[1] = reg->line[0]->y[1] - reg->line[0]->y[0];
   alpha[0] = float(atan(u1[1]/u1[0]));
   u1[0] = reg->arc[1]->list[le_dis-1] - reg->arc[2]->list[le_dis-1];
   u1[1] = reg->line[1]->y[le_dis-1] - reg->line[2]->y[le_dis-1];
   alpha[1] = beta[1] = float(atan(u1[1]/u1[0]));
   u1[0] = reg->arc[0]->list[reg->line[0]->nump-1]
      - reg->arc[0]->list[reg->line[0]->nump-2];
   u1[1] = reg->line[0]->y[reg->line[0]->nump-1]
      - reg->line[0]->y[reg->line[0]->nump-2];
   if( (beta[0] = float(atan(u1[1]/u1[0]))) < 0.0f) beta[0] += (float) M_PI;
#ifdef DEBUG_PS
   fprintf(fp,"alpha = [%14.4f  %14.4f], beta = [%14.4f  %14.4f]\n",
      alpha[0]*180/M_PI, alpha[1]*180/M_PI,
      beta[0]*180/M_PI, alpha[1]*180/M_PI);
#endif
   // spline curves
   for(i = 1; i < le_dis-1; i++)
   {
      p1[0] = reg->arc[1]->list[i];
      p1[1] = reg->line[1]->y[i];
      p3[0] = reg->arc[2]->list[i];
      p3[1] = reg->line[2]->y[i];
      para  = reg->para[1]->list[i]/reg->para[1]->list[le_dis-1];
      delta = (1.0f-para)*alpha[0] + para*alpha[1];
      u1[0] = float(cos(delta));
      u1[1] = float(sin(delta));
#ifdef DEBUG_PS
      fprintf(fp,"# delta_1 = %f\n#",delta*180/M_PI);
      VPRINTF(u1,fp);
#endif
      delta = (1.0f-para)*beta[0] + para*beta[1];
      u2[0] = float(cos(delta));
      u2[1] = float(sin(delta));
#ifdef DEBUG_PS
      fprintf(fp,"# delta_2 = %f\n#",delta*180/M_PI);
      VPRINTF(u2,fp);
#endif
      LineIntersect(p1,u1, p3,u2, p2);
      if( ((p2[0]-p3[0])*u2[0] >= 0.0) ||
         ((p2[0]-p1[0])*u1[0] <= 0.0) )
      {
#ifdef DEBUG_PS
         fprintf(fp,"# p2 is on wrong side!\n");
         fprintf(fp,"# (p2[0]-p3[0]) = %f\n",(p2[0]-p3[0]));
         fprintf(fp,"# u2[0]         = %f\n",u2[0]);
         fprintf(fp,"# (p2[0]-p1[0]) = %f\n",(p2[0]-p1[0]));
         fprintf(fp,"# u1[0]         = %f\n",u1[0]);
#endif
         le_dis = i+1;
         break;
      }
      poly = CurvePolygon(p1,p2,p3,0.5,0.5);
      knot = BSplineKnot(poly,BSPLN_DEGREE);
      for(j = 0; j <  reg->line[0]->nump; j++)
      {
         BSplinePoint(BSPLN_DEGREE, poly, knot,
            reg->para[0]->list[j], p);
         Add2Ilist(reg->nodes[reg->numl], n->num);
         AddVNode(n,p,ARC);
      }
#ifdef DEBUG_PS
      fprintf(fp,"# %3d (%d)\n",i,le_dis-1);
      for(j = 0; j < 100; j++)
      {
         BSplinePoint(BSPLN_DEGREE, poly, knot,
            (float)j/99.0, p);
         fprintf(fp,"%4d %12.6f %12.6f %12.6f\n",j,
            p[0], p[1], p[2]);
      }
#endif
      FreePointStruct(poly);
      FreeFlistStruct(knot);
      continue;
   }                                              // end i
   // straight line
   for(i = le_dis-1; i < reg->line[1]->nump; i++)
   {
      u1[0] = reg->arc[2]->list[i] - reg->arc[1]->list[i];
      u1[1] = reg->line[2]->y[i]   - reg->line[1]->y[i];
      u2[0] = reg->arc[1]->list[i];
      u2[1] = reg->line[1]->y[i];
      for(j = 0; j < reg->line[0]->nump; j++)
      {
         p[0] = u2[0] + reg->para[0]->list[j]*u1[0];
         p[1] = u2[1] + reg->para[0]->list[j]*u1[1];
         Add2Ilist(reg->nodes[reg->numl], n->num);
         AddVNode(n, p, ARC);
      }
   }
   newnodes = n->num - offset;
   CalcNodeRadius(&n->n[n->num-newnodes], ml, newnodes);
   CalcNodeAngle(&n->n[n->num-newnodes], ml, newnodes);
   // get boundary nodes
   // extension
   for(i = reg0->nodes[3]->num-jadd; i < reg0->nodes[3]->num; i++)
      Add2Ilist(reg->nodes[0], reg0->nodes[3]->list[i]);
   tmpnode = n->n + offset;
   for(i = jadd; i < reg->line[0]->nump; i++)
   {
      Add2Ilist(psle, (*tmpnode)->index);
      Add2Ilist(reg->nodes[0], (*tmpnode)->index);
      tmpnode++;
   }
   // envelope
   Add2Ilist(reg->nodes[1], reg0->nodes[3]->list[reg0->nodes[3]->num-jadd]);
   tmpnode = n->n + offset + reg->line[0]->nump-jadd;
   for(i = 1; i < reg->line[1]->nump; i++)
   {
      Add2Ilist(reg->nodes[1], (*tmpnode)->index);
      tmpnode += reg->line[0]->nump;
   }
   // blade surface
   tmpnode = n->n + offset + reg->line[0]->nump-jadd-1;
   for(i = 0; i < reg->line[2]->nump; i++)
   {
      Add2Ilist(psnod, (*tmpnode)->index);
      Add2Ilist(reg->nodes[2], (*tmpnode)->index);
      tmpnode += reg->line[0]->nump;
   }
   // te ext.
   tmpnode = n->n + (n->num - reg->line[3]->nump);
   for(i = 0; i < reg->line[0]->nump; i++)
   {
      Add2Ilist(reg->nodes[3], (*tmpnode)->index);
      tmpnode++;
   }

#ifdef DEBUG_PS
   fprintf(fp,"psle\n");
   DumpIlist2File(psle, fp);
   fprintf(fp,"psnod\n");
   DumpIlist2File(psnod, fp);

   for(i = 0; i < reg->numl+1; i++)
   {
      fprintf(fp," ** reg->nodes[%d] **\n", i);
      DumpIlist2File(reg->nodes[i], fp);
   }
#endif

#ifdef DEBUG_PS
   fclose(fp);
#endif

#ifdef DEBUG_PS
   sprintf(fn,"rr_debugpsnodes_%02d.txt", count-1);
   if( (fp = fopen(fn,"w+")) == NULL)
   {
      fprintf(stderr,"Shit happened opening file '%s'!\n",fn);
      exit(-1);
   }
   for(i = 0; i < le_dis+3; i++)
   {
      for(j = 0; j < reg->nodes[0]->num; j++)
      {
         fprintf(fp,"%16.6f  %16.6f\n",
            n->n[reg->nodes[reg->numl]->list[i*reg->nodes[0]->num+j]]->arc,
            n->n[reg->nodes[reg->numl]->list[i*reg->nodes[0]->num+j]]->l);
      }
      fprintf(fp,"\n");
   }
   fclose(fp);
#endif

   return 0;
}
