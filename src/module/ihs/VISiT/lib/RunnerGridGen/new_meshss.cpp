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
#include "../General/include/plane_geo.h"
#include "../General/include/curvepoly.h"
#include "../BSpline/include/bspline.h"
#include "../General/include/curve.h"
#include "../General/include/nodes.h"
#include "../General/include/fatal.h"
#include "../General/include/v.h"

#include "include/rr_grid.h"

#ifndef BSPLN_DEGREE
#define BSPLN_DEGREE 3
#endif

#ifndef SMALL
#define SMALL 0.0001
#endif
#ifndef ABS
#define ABS(a)    ( (a) >= (0) ? (a) : -(a) )
#endif

#ifdef DEBUG_SS
#define VPRINT(x) fprintf(stderr,"%10s = [%4f, %4f, %4f]\n",#x,x[0], x[1], x[2])
#define VPRINTF(x,f) fprintf(f,"%10s = [%4f, %4f, %4f]\n",#x,x[0], x[1], x[2])
#endif

int MeshNew_SSRegion(struct Nodelist *n, struct curve *ml, struct Ilist *ssnod,
struct Ilist *ssle, struct region *reg, struct region *reg0, int le_dis)
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

#ifdef DEBUG_SS
   char fn[111];
   FILE *fp;
   static int count = 0;

   sprintf(fn,"rr_debugss_%02d.txt", count++);
   if( (fp = fopen(fn,"w+")) == NULL)
   {
      fprintf(stderr,"Shit happened opening file '%s'!\n",fn);
      exit(-1);
   }
   fprintf(fp," MeshNew_SSRegion %d\n",count);
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

   // **************************************************
   // initialize some params.
   offset = n->num;
   jadd   = reg->line[0]->nump/2;
   p[2] = p1[2] = p2[2] = p3[2] = u1[2] = u2[2] = 0.0;

#ifdef DEBUG_SS
   fprintf(fp," offset = %d\n", offset);
   fprintf(fp," jadd = %d\n",jadd);
#endif

   for(j = jadd-1; j >= 0; j--)
      Add2Ilist(reg->nodes[reg->numl], reg0->nodes[3]->list[j]);
   // copy points
   for(j = jadd; j < reg->line[0]->nump; j++)
   {
      p[0] = reg->arc[0]->list[j];
      p[1] = reg->line[0]->y[j];
      Add2Ilist(reg->nodes[reg->numl], n->num);
      AddVNode(n, p, ARC);
   }
   // create new nodes, bound. layer
   // create nodes on spline for i < le_dis,
   // angles for spline
   u1[0] = reg->arc[0]->list[1] - reg->arc[0]->list[0];
   u1[1] = reg->line[0]->y[1] - reg->line[0]->y[0];
   alpha[0] = float(atan(u1[1]/u1[0]) + M_PI);
   u1[0] = reg->arc[1]->list[le_dis-1] - reg->arc[2]->list[le_dis-1];
   u1[1] = reg->line[1]->y[le_dis-1] - reg->line[2]->y[le_dis-1];
   alpha[1] = float(atan(u1[1]/u1[0]) + M_PI);
   beta[1]  = float(atan(u1[1]/u1[0]));
   u1[0] = reg->arc[0]->list[reg->line[0]->nump-1]
      - reg->arc[0]->list[reg->line[0]->nump-2];
   u1[1] = reg->line[0]->y[reg->line[0]->nump-1]
      - reg->line[0]->y[reg->line[0]->nump-2];
   beta[0] = float(atan(u1[1]/u1[0]));
#ifdef DEBUG_SS
   fprintf(fp,"alpha = [%14.4f  %14.4f], beta = [%14.4f  %14.4f]\n",
      alpha[0]*180/M_PI, alpha[1]*180/M_PI,
      beta[0]*180/M_PI, alpha[1]*180/M_PI);
#endif
   // spline curves
   for(i = 1; i < le_dis-1; i++)
   {
      p1[0] = reg->arc[2]->list[i];
      p1[1] = reg->line[2]->y[i];
      p3[0] = reg->arc[1]->list[i];
      p3[1] = reg->line[1]->y[i];
      para  = reg->para[2]->list[i]/reg->para[2]->list[le_dis-1];
      delta = (1.0f-para)*alpha[0] + para*alpha[1];
      u1[0] = float(cos(delta));
      u1[1] = float(sin(delta));
#ifdef DEBUG_SS
      fprintf(fp,"delta_1 = %f\n",delta*180/M_PI);
      VPRINTF(u1,fp);
#endif
      delta = (1.0f-para)*beta[0] + para*beta[1];
      u2[0] = float(cos(delta));
      u2[1] = float(sin(delta));
#ifdef DEBUG_SS
      fprintf(fp,"delta_2 = %f\n",delta*180/M_PI);
      VPRINTF(u2,fp);
#endif
      LineIntersect(p1,u1, p3,u2, p2);
      if(p2[0] > p1[0])
      {
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
      FreePointStruct(poly);
      FreeFlistStruct(knot);
   }                                              // end i
   // straight line
   for(i = le_dis-1; i < reg->line[1]->nump; i++)
   {
      u1[0] = reg->arc[1]->list[i] - reg->arc[2]->list[i];
      u1[1] = reg->line[1]->y[i]   - reg->line[2]->y[i];
      u2[0] = reg->arc[2]->list[i];
      u2[1] = reg->line[2]->y[i];
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
   for(i = jadd-1; i >= 0; i--)
      Add2Ilist(reg->nodes[0], reg0->nodes[3]->list[i]);
   tmpnode = n->n + offset;
   for(i = jadd; i < reg->line[0]->nump; i++)
   {
      Add2Ilist(ssle, (*tmpnode)->index);
      Add2Ilist(reg->nodes[0], (*tmpnode)->index);
      tmpnode++;
   }
   // blade surface
   tmpnode = n->n + offset + reg->line[0]->nump-jadd-1;
   for(i = 0; i < reg->line[1]->nump; i++)
   {
      Add2Ilist(ssnod, (*tmpnode)->index);
      Add2Ilist(reg->nodes[1], (*tmpnode)->index);
      tmpnode += reg->line[0]->nump;
   }
   // envelope
   Add2Ilist(reg->nodes[2], reg0->nodes[3]->list[jadd-1]);
   tmpnode = n->n + offset + reg->line[0]->nump-jadd;
   for(i = 1; i < reg->line[1]->nump; i++)
   {
      Add2Ilist(reg->nodes[2], (*tmpnode)->index);
      tmpnode += reg->line[0]->nump;
   }
   // trailing edge extension
   tmpnode = n->n + (n->num - reg->line[3]->nump);
   for(i = 0; i < reg->line[0]->nump; i++)
   {
      Add2Ilist(reg->nodes[3], (*tmpnode)->index);
      tmpnode++;
   }

#ifdef DEBUG_SS
   fprintf(fp,"ssle\n");
   DumpIlist2File(ssle, fp);
   fprintf(fp,"ssnod\n");
   DumpIlist2File(ssnod, fp);

   for(i = 0; i < reg->numl+1; i++)
   {
      fprintf(fp," ** reg->nodes[%d] **\n", i);
      DumpIlist2File(reg->nodes[i], fp);
   }
#endif

#ifdef DEBUG_SS
   fclose(fp);
#endif

#ifdef DEBUG_SS
   sprintf(fn,"rr_debugssnodes_%02d.txt", count-1);
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
