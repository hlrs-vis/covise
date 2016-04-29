#include <stdio.h>
#include <stdlib.h>
#ifdef WIN32
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#endif
#include <math.h>

#include "../General/include/flist.h"
#include "../General/include/points.h"
#include "../General/include/curve.h"
#include "../General/include/nodes.h"
#include "../General/include/plane_geo.h"
#include "../General/include/curvepoly.h"
#include "../BSpline/include/bspline.h"
#include "../General/include/fatal.h"
#include "../General/include/v.h"

#include "include/rr_grid.h"

#ifndef BSPLN_DEGREE
#define BSPLN_DEGREE 3
#endif

#ifndef TINY
#define TINY 1.0e-8
#endif
#ifndef SIGN
#define SIGN(a,b)	 ( (a) >= (b) ? (1.0) : -(1.0) )
#endif
#ifndef ABS
#define ABS(a)	  ( (a) >= (0) ? (a) : -(a) )
#endif
#ifndef MAX
#define MAX(a,b)	( (a) > (b) ? (a) : (b) )
#endif

#define SHIFT_CURVE

#ifdef DEBUG_CORE
#define VPRINT(x) fprintf(stderr,"%10s = [%4f, %4f, %4f]\n",#x,x[0], x[1], x[2])
#define VPRINTF(x,f) fprintf(f,"%10s = [%4f, %4f, %4f]\n",#x,x[0], x[1], x[2])
#endif

int MeshRR_CoreRegion(struct Nodelist *n, struct curve *ml, struct region *reg,
					  struct region *reg0, struct region *reg1,
					  struct region *reg3, float angle14, int jadd)
{
	int i, j, ix;
	int offset, newnodes;

	int ispline, double_curved;
#ifdef MODIFY_CORE
	int node, prevnode;
	float delta, ddelta;
#endif

	float u1[3], v1[3], v2[3], v3[3];
	float p[3], p1[3], p2[3], p3[3];
	float alpha[2], beta[2], para, invpara, ppara;
	float slen;

	// must be identical to spline paras of line 1.4 = 3.1
	float t[] = {0.5f, 0.4f};

	struct Point *poly = NULL;
	struct Flist *knot = NULL;

	struct Point *sspoly;
	struct Point *pspoly;

	struct Flist *spara;
	struct Point *sline;

	struct node **tmpnode = NULL;

#ifdef DEBUG_CORE
	char fn[111];
	char fngnu[111];
	FILE *fp;
	FILE *fpgnu;
	int jx;
	static int fcount = 0;

	sprintf(fngnu,"rr_coregnu_%02d.txt", fcount);
	if( (fpgnu = fopen(fngnu,"w+")) == NULL) {
		fprintf(stderr,"Shit happened opening file '%s'!\n",fngnu);
		exit(-1);
	}
	sprintf(fn,"rr_debugcore_%02d.txt", fcount++);
	if( (fp = fopen(fn,"w+")) == NULL) {
		fprintf(stderr,"Shit happened opening file '%s'!\n",fn);
		exit(-1);
	}
	fprintf(fp," MeshRR_CoreRegion %d\n",fcount);
	fprintf(fpgnu,"# line no. phi*r	 l	z	para\n");
#endif

	// **************************************************
	// init
	u1[2] = v1[2] = v2[2] = v3[2] = 0.0;
	p[2]  = p1[2] = p2[2] = p3[2] = 0.0;
#ifdef MODIFY_CORE
	ddelta = 90.0/180.0*M_PI;
#endif

	// **************************************************
	// mem. check
	for(i = 0; i < reg->numl; i++) {
		if(reg->nodes[i]) {
			FreeIlistStruct(reg->nodes[i]);
			reg->nodes[i] = NULL;
		}
	}
	if(reg->nodes[reg->numl]) {
		FreeIlistStruct(reg->nodes[reg->numl]);
		reg->nodes[reg->numl] = NULL;
	}

	offset = n->num;
#ifdef DEBUG_CORE
	fprintf(fp," offset = %d\n", offset);
#endif

	// **************************************************
	// copy existing nodes
	reg->nodes[0] = nCopynIlistStruct(reg0->nodes[3],jadd,
									  reg0->nodes[3]->num-2*jadd);
	reg->nodes[1] = nCopynIlistStruct(reg1->nodes[2], jadd,
									  reg3->nodes[1]->num-jadd);
	reg->nodes[2] = nCopynIlistStruct(reg3->nodes[1],jadd,
									  reg3->nodes[1]->num-jadd);
	reg->nodes[3] = AllocIlistStruct(reg->line[3]->nump+1);
	reg->nodes[reg->numl] = nCopynIlistStruct(reg0->nodes[3],jadd,
									  reg0->nodes[3]->num-2*jadd);
	// **************************************************
	// get index where to start with straight lines
	if( (ispline = GetPointIndex(reg->line[1]->nump, reg->para[1]->list, 1.0, 0)) < 0) {
		fprintf(stderr,"\n point not found!!\n");
	}
	// real spline para
	spara	   = AllocFlistStruct(reg->line[0]->nump+1);
	sline	   = AllocPointStruct();			   // temporary line
	// **************************************************
	// get angles for splines between 3.2 & 3.3
	// ss angles
	u1[0] = reg->arc[0]->list[1] - reg->arc[0]->list[0];
	u1[1] = reg->line[0]->y[1]	 - reg->line[0]->y[0];
	alpha[0] = atan(u1[1]/u1[0]);
	ix = reg->line[1]->nump-1;
	u1[0] = reg->arc[2]->list[ix] - reg->arc[1]->list[ix];
	u1[1] = reg->line[2]->y[ix]	  - reg->line[1]->y[ix];
	alpha[1] = atan(u1[1]/u1[0]);
	// ps angles
	ix = reg->line[0]->nump-1;
	u1[0] = reg->arc[0]->list[ix-1] - reg->arc[0]->list[ix];
	u1[1] = reg->line[0]->y[ix-1]	- reg->line[0]->y[ix];
	beta[0] = atan(u1[1]/u1[0]);
	if(u1[0] < 0.0f && u1[1] < 0.0f) beta[0] -= (float) M_PI;
	beta[1] = alpha[1]-(float) M_PI;
#ifdef DEBUG_CORE
	fprintf(fp," alpha = [ %f  %f], beta = [ %f	 %f]\n",
			alpha[0]*180/M_PI, alpha[1]*180/M_PI,
			beta[0]*180/M_PI, beta[1]*180/M_PI);
#endif

	// **************************************************
	// create splines
	double_curved = 0;
	for(i = 1; i < ispline; i++) {
		para	= reg->para[1]->list[i];
		invpara = 1.0 - para;
		sline->nump = 0;
		spara->num	= 0;
#ifdef DEBUG_CORE
		fprintf(fp,"i: %3d: para, invpara: %10.5f, %10.5f\n",
				i,para,invpara);
		fprintf(fp,"\tss-, ps-angle: %10.5f, %10.5f\n",
				(alpha[0]*invpara + alpha[1]*para)*180/M_PI,
				(beta[0]*invpara  + beta[1]*para)*180/M_PI);
#endif
		// spline points and vectors
		p1[0] = reg->arc[1]->list[i];
		p1[1] = reg->line[1]->y[i];
		p3[0] = reg->arc[2]->list[i];
		p3[1] = reg->line[2]->y[i];

		v1[0] = cos(alpha[0]*invpara + alpha[1]*para);
		v1[1] = sin(alpha[0]*invpara + alpha[1]*para);
		v3[0] = cos(beta[0]*invpara	 + beta[1]*para);
		v3[1] = sin(beta[0]*invpara	 + beta[1]*para);

		// polygon
		if(v1[1] > 0.0 && (double_curved || i == 1)) {
			double_curved = 1;
			p[0]  = p1[0] + 0.25*(p3[0] - p1[0]);
			p[1]  = p1[1] + 0.25*(p3[1] - p1[1]);
			u1[0] = 1.0;
			u1[1] = -tan(angle14*invpara);
			LineIntersect(p1,v1, p,u1, p2);
			sspoly = CurvePolygon(p1,p2,p,0.9f,0.2f);
			LineIntersect(p,u1, p3,v3, p2);
			pspoly = CurvePolygon(p,p2,p3,0.2f,0.9f);
			poly   = AllocPointStruct();
			for(j = 0; j < sspoly->nump; j++)
				AddPoint(poly,sspoly->x[j],sspoly->y[j],sspoly->z[j]);
			for(j = 1; j < pspoly->nump; j++)
				AddPoint(poly,pspoly->x[j],pspoly->y[j],pspoly->z[j]);
			FreePointStruct(sspoly);
			FreePointStruct(pspoly);
		}
		else {
			LineIntersect(p1,v1, p3,v3, p2);
			poly = CurvePolygon(p1, p2, p3, t[0], t[1]);
		}											// end if
		knot = BSplineKnot(poly, BSPLN_DEGREE);
#ifdef DEBUG_CORE
		fprintf(fp,"p1, p2, p3: [%8.4f %8.4f][%8.4f %8.4f][%8.4f %8.4f]\n",
				p1[0],p1[1],p2[0],p2[1],p3[0],p3[1]);
#endif
		if(p2[1] > p3[1]) {
			ispline = i;
			break;
		}

		Add2Flist(spara,0.0);
		AddPoint(sline,reg->arc[1]->list[i], reg->line[1]->y[i], reg->line[1]->z[i]);
		slen = 0.0;
		// get spline, parameters might be wrong!
		for(j = 1; j < reg->line[0]->nump; j++) {
			BSplinePoint(BSPLN_DEGREE, poly, knot,
						 reg->para[0]->list[j]*invpara
						 + reg->para[3]->list[j]*para, p);
			slen += sqrt(pow(sline->x[j-1]-p[0],2) + pow(sline->y[j-1]-p[1],2));
			AddVPoint(sline,p);
			Add2Flist(spara, slen);
#ifdef DEBUG_CORE
			fprintf(fp,"i, j: %3d, %3d: slen = %f\n",i,j,slen);
#endif
		}

		// get real parameters
		for(j = 0; j < reg->line[0]->nump; j++) {
			spara->list[j] /= slen;
#ifdef DEBUG_CORE
			ppara = reg->para[0]->list[j]*invpara + reg->para[3]->list[j]*para;
			if(ppara) {
				fprintf(fp,"i, j: %3d, %3d: spara, ppara: %8.6f %8.6f  %8.3f\n",
						i,j,spara->list[j],ppara, 100*((spara->list[j]/ppara)-1.0));
			}
			else {
				fprintf(fp,"i, j: %3d, %3d: spara, ppara: %8.6f %8.6f  %8.3f\n",
						i,j,spara->list[j],ppara, 0.0);
			}
#endif
		}											// end j

		// move points to correct parameter position and add them to node list
		// pointer on reg->nodes[reg->numl]->list won't work since list is realloc'd!!!
		ix = 1;
		Add2Ilist(reg->nodes[reg->numl], reg->nodes[1]->list[i]);
		for(j = 1; j < reg->line[0]->nump-1; j++) {
			ppara = reg->para[0]->list[j]*invpara + reg->para[3]->list[j]*para;
			ix = GetPointIndex(sline->nump, spara->list, ppara, ix-1);
			p[0] = sline->x[ix] + (ppara - spara->list[ix])*(sline->x[ix+1]-sline->x[ix])/
				(spara->list[ix+1] - spara->list[ix]);
			p[1] = sline->y[ix] + (ppara - spara->list[ix])*(sline->y[ix+1]-sline->y[ix])/
				(spara->list[ix+1] - spara->list[ix]);
#ifdef MODIFY_CORE
			// check on crossing edges.
			// v1 - normal to prev. line, v2 prev. line->line
			node	 = reg->nodes[reg->numl]->list[(i-1)*reg->line[0]->nump+j];
			prevnode = reg->nodes[reg->numl]->list[(i-1)*reg->line[0]->nump+j-1];
			p1[0] = n->n[node]->arc;
			p1[1] = n->n[node]->l;
			v1[0] =	 -(n->n[prevnode]->l   - p1[1]);
			v1[1] =	   n->n[prevnode]->arc - p1[0];
			v2[0] =	   p[0] - p1[0];
			v2[1] =	   p[1] - p1[1];
			delta = acos(V_Angle(v1,v2)) - ddelta;
#ifdef DEBUG_CORE
			fprintf(fp,"# i, j: %d, %d: Angle = %f, delta = %f\n",
					i,j,180/M_PI*(delta+ddelta),180/M_PI*delta);
			fprintf(fp,"# \tnode, prevnode: %d,	 %d\n",node, prevnode);
			fprintf(fp," %f	 %f\n",p1[0], p1[1]);
			fprintf(fp," %f	 %f\n",p1[0]+v1[0], p1[1]+v1[1]);
			fprintf(fp," %f	 %f\n",p1[0], p1[1]);
			fprintf(fp," %f	 %f\n",p1[0]+v2[0], p1[1]+v2[1]);
			fprintf(fp," %f	 %f\n",p1[0], p1[1]);
			//fprintf(fp," %f  %f\n",sline->x[j],sline->y[j]);
			//fprintf(fp," %f  %f\n",p[0], p[1]);
#endif
			if(delta < 0.0) {
				delta *= -2.0;
				v3[0]  =  v2[0]*cos(delta) + v2[1]*sin(delta);
				v3[1]  = -v2[0]*sin(delta) + v2[1]*cos(delta);
				delta  = V_Len(v3);
				V_Norm(v3);
				v3[0] *= delta;
				v3[1] *= delta;
				p[0]   = p1[0]+v3[0];
				p[1]   = p1[1]+v3[1];
#ifdef DEBUG_CORE
				fprintf(fp," %f	 %f\n",p1[0]+v3[0], p1[1]+v3[1]);
				fprintf(fp," %f	 %f\n",p1[0], p1[1]);
				fprintf(fp,"# New Angle = %f\n",180/M_PI*acos(V_Angle(v1,v3)));
				fprintf(fp,"# modified node!\n");
#endif
			}
#endif									 // MODIFY_CORE
			Add2Ilist(reg->nodes[reg->numl],n->num);
			AddVNode(n, p, ARC);
#ifdef DEBUG_CORE
			fprintf(fpgnu,"%3d %10.6f %10.6f %10.6f	  %10.6f\n",
					i,p[0],p[1],p[2],ppara);
#endif
		}
		Add2Ilist(reg->nodes[reg->numl], reg->nodes[2]->list[i]);
#ifdef DEBUG_CORE
		fprintf(fpgnu,"\n");
#endif

		FreePointStruct(poly);
		FreeFlistStruct(knot);
	}											   // end i
	FreePointStruct(sline);
	FreeFlistStruct(spara);

	// **************************************************
	// straight lines
#ifdef DEBUG_CORE
	fprintf(fpgnu,"\n# straight lines\n");
#endif
	for(i = ispline; i < reg->line[1]->nump-1; i++) {
		para	= reg->para[1]->list[i];
		invpara = 1.0 - para;
		u1[0] = reg->arc[2]->list[i] - reg->arc[1]->list[i];
		u1[1] = reg->line[2]->y[i]	 - reg->line[1]->y[i];
		p1[0] = reg->arc[1]->list[i];
		p1[1] = reg->line[1]->y[i];
		Add2Ilist(reg->nodes[reg->numl], reg->nodes[1]->list[i]);
		for(j = 1; j < reg->line[0]->nump-1; j++) {
			ppara = reg->para[0]->list[j]*invpara
				+ reg->para[3]->list[j]*para;
			p[0] = p1[0] + ppara * u1[0];
			p[1] = p1[1] + ppara * u1[1];
			Add2Ilist(reg->nodes[reg->numl],n->num);
			AddVNode(n, p, ARC);
#ifdef DEBUG_CORE
			fprintf(fpgnu,"%3d %10.6f %10.6f %10.6f	  %10.6f\n",
					i,p[0],p[1],p[2],ppara);
#endif
		}
		Add2Ilist(reg->nodes[reg->numl], reg->nodes[2]->list[i]);
#ifdef DEBUG_CORE
		fprintf(fpgnu,"\n");
#endif
	}											   // end i

	// **************************************************
	// points on line 3.4
	Add2Ilist(reg->nodes[reg->numl], reg->nodes[1]->list[reg->nodes[1]->num-1]);
	for(i = 1; i < reg->line[3]->nump-1; i++) {
		p[0] = reg->arc[3]->list[i];
		p[1] = reg->line[3]->y[i];
		Add2Ilist(reg->nodes[reg->numl], n->num);
		AddVNode(n, p, ARC);
		tmpnode++;
	}
	Add2Ilist(reg->nodes[reg->numl], reg->nodes[2]->list[reg->nodes[2]->num-1]);

	// boundary line 3.4
	Add2Ilist(reg->nodes[3], reg->nodes[1]->list[reg->nodes[1]->num-1]);
	tmpnode = n->n + n->num - reg->line[3]->nump+2;
	for(i = 1; i < reg->line[3]->nump-1; i++) {
		Add2Ilist(reg->nodes[3],(*tmpnode)->index);
		tmpnode++;
	}
	Add2Ilist(reg->nodes[3], reg->nodes[2]->list[reg->nodes[2]->num-1]);

	newnodes = n->num - offset;
	CalcNodeRadius(&n->n[n->num-newnodes], ml, newnodes);
	CalcNodeAngle(&n->n[n->num-newnodes], ml, newnodes);

	// **************************************************

#ifdef DEBUG_CORE
	fprintf(fp,"core\n");
	for(i = 0; i < reg->numl+1; i++) {
		fprintf(fp," ** reg->nodes[%d] **\n", i);
		DumpIlist2File(reg->nodes[i], fp);
	}
	fprintf(fp,"n->num - offset = %d\n",n->num-offset);
	fclose(fp);
	fclose(fpgnu);

	sprintf(fn,"rr_corenodes_%02d.txt", fcount-1);
	if( (fp = fopen(fn,"w+")) == NULL) {
		fprintf(stderr,"Shit happened opening file '%s'!\n",fn);
		exit(-1);
	}
	fprintf(fp,"#	ID\t phi\t l\t r\t s\t\t x\t y\t z\t\n");
	for(i = 0; i < reg->nodes[1]->num; i++) {
		ix = i * reg->nodes[0]->num;
		for(j = 0; j < reg->nodes[0]->num; j++) {
			jx = ix + j;
			fprintf(fp," %10d	%10.6f	%10.6f	%10.6f	%10.6f	   %10.6f  %10.6f  %10.6f\n",
					n->n[reg->nodes[reg->numl]->list[jx]]->id, n->n[reg->nodes[reg->numl]->list[jx]]->phi,
					n->n[reg->nodes[reg->numl]->list[jx]]->l, n->n[reg->nodes[reg->numl]->list[jx]]->r,
					n->n[reg->nodes[reg->numl]->list[jx]]->phi*n->n[reg->nodes[reg->numl]->list[jx]]->r,
					n->n[reg->nodes[reg->numl]->list[jx]]->x,
					n->n[reg->nodes[reg->numl]->list[jx]]->y, n->n[reg->nodes[reg->numl]->list[jx]]->z);
		}
		fprintf(fp,"\n");
	}
	fprintf(fp,"\n\n");
	for(i = 0; i < reg->nodes[0]->num; i++) {
		for(j = 0; j < reg->nodes[1]->num; j++) {
			jx = i + j*reg->nodes[0]->num;
			fprintf(fp," %10d	%10.6f	%10.6f	%10.6f	%10.6f	   %10.6f  %10.6f  %10.6f\n",
					n->n[reg->nodes[reg->numl]->list[jx]]->id, n->n[reg->nodes[reg->numl]->list[jx]]->phi,
					n->n[reg->nodes[reg->numl]->list[jx]]->l, n->n[reg->nodes[reg->numl]->list[jx]]->r,
					n->n[reg->nodes[reg->numl]->list[jx]]->phi*n->n[reg->nodes[reg->numl]->list[jx]]->r,
					n->n[reg->nodes[reg->numl]->list[jx]]->x,
					n->n[reg->nodes[reg->numl]->list[jx]]->y, n->n[reg->nodes[reg->numl]->list[jx]]->z);
		}
		fprintf(fp,"\n");
	}
	for(i = 0; i < reg->numl; i++) {
		fprintf(fp,"\n\n");
		for(j = 0; j < reg->nodes[i]->num; j++) {
			fprintf(fp," %10d	%10.6f	%10.6f	%10.6f	%10.6f	   %10.6f  %10.6f  %10.6f\n",
					n->n[reg->nodes[i]->list[j]]->id, n->n[reg->nodes[i]->list[j]]->phi,
					n->n[reg->nodes[i]->list[j]]->l, n->n[reg->nodes[i]->list[j]]->r,
					n->n[reg->nodes[i]->list[j]]->phi*n->n[reg->nodes[i]->list[j]]->r,
					n->n[reg->nodes[i]->list[j]]->x,
					n->n[reg->nodes[i]->list[j]]->y, n->n[reg->nodes[i]->list[j]]->z);
		}
	}
	fclose(fp);
#endif

	return(0);
}
