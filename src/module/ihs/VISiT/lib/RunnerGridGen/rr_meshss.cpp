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
#define ABS(a)	  ( (a) >= (0) ? (a) : -(a) )
#endif

#ifdef DEBUG_SS
#define VPRINT(x) fprintf(stderr,"%10s = [%4f, %4f, %4f]\n",#x,x[0], x[1], x[2])
#define VPRINTF(x,f) fprintf(f,"%10s = [%4f, %4f, %4f]\n",#x,x[0], x[1], x[2])
#endif

int MeshRR_SSRegion(struct Nodelist *n, struct curve *ml, struct Ilist *ssnod,
					struct Ilist *ssle, struct region *reg,struct region *reg0,
					int jadd)
{
	int i, j;
	int offset, newnodes;						   // here current region starts

	float u1[3], u2[3];
	float p[3];

	struct node **tmpnode = NULL;

#ifdef DEBUG_SS
	char fn[111];
	FILE *fp;
	static int count = 0;

	sprintf(fn,"rr_debugss_%02d.txt", count++);
	if( (fp = fopen(fn,"w+")) == NULL) {
		fprintf(stderr,"Shit happened opening file '%s'!\n",fn);
		exit(-1);
	}
	fprintf(fp," MeshRR_SSRegion %d\n",count);
#endif

	// mem. check
	for(i = 0; i < reg->numl; i++) {
		if(reg->nodes[i]) {
			FreeIlistStruct(reg->nodes[i]);
			reg->nodes[i] = NULL;
		}
		reg->nodes[i] = AllocIlistStruct(reg->line[i]->nump+1);
	}
	if(reg->nodes[reg->numl]) {
		FreeIlistStruct(reg->nodes[reg->numl]);
		reg->nodes[reg->numl+1] = NULL;
	}
	reg->nodes[reg->numl] = AllocIlistStruct(reg->line[0]->nump * reg->line[1]->nump + 1);

	offset = n->num;

#ifdef DEBUG_SS
	fprintf(fp," offset = %d\n", offset);
#endif

	
	// create nodes, leave out first node where reg0 and reg1 contact
	for(i = 0; i <= jadd; i++) {
		Add2Ilist(reg->nodes[reg->numl],reg0->nodes[3]->list[i]);
		u1[0] = reg->arc[1]->list[i] - reg->arc[2]->list[i];
		u1[1] = reg->line[1]->y[i]	 - reg->line[2]->y[i];
		u2[0] = reg->arc[2]->list[i];
		u2[1] = reg->line[2]->y[i];
		for(j = 1; j < reg->line[0]->nump; j++) {
			p[0] = u2[0] + reg->para[0]->list[j]*u1[0];
			p[1] = u2[1] + reg->para[0]->list[j]*u1[1];
			Add2Ilist(reg->nodes[reg->numl], n->num);
			AddVNode(n, p, ARC);
		}
	}
	for(i = jadd+1; i < reg->line[1]->nump; i++) {
		u1[0] = reg->arc[1]->list[i] - reg->arc[2]->list[i];
		u1[1] = reg->line[1]->y[i]	 - reg->line[2]->y[i];
		u2[0] = reg->arc[2]->list[i];
		u2[1] = reg->line[2]->y[i];
		for(j = 0; j < reg->line[0]->nump; j++) {
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
	Add2Ilist(reg->nodes[0], reg0->nodes[1]->list[reg0->nodes[1]->num-1]);
	tmpnode = n->n + offset;
	for(i = 0; i < reg->line[0]->nump-1; i++) {
		Add2Ilist(ssle, (*tmpnode)->index);
		Add2Ilist(reg->nodes[0], (*tmpnode)->index);
		tmpnode++;
	}
	// blade surface
	tmpnode = n->n + offset + reg->line[0]->nump-2;
	for(i = 0; i < reg->line[1]->nump; i++) {
		Add2Ilist(ssnod, (*tmpnode)->index);
		Add2Ilist(reg->nodes[1], (*tmpnode)->index);
		tmpnode += reg->line[0]->nump;
		if(i < jadd) tmpnode -= 1;
	}
	// envelope
	for(i = 0; i <= jadd; i++)
		Add2Ilist(reg->nodes[2], reg0->nodes[3]->list[i]);
	tmpnode = n->n + offset + (jadd+1)*(reg->line[0]->nump-1);
	for(i = jadd+1; i < reg->line[1]->nump; i++) {
		Add2Ilist(reg->nodes[2], (*tmpnode)->index);
		tmpnode += reg->line[0]->nump;
	}
	// trailing edge extension
	tmpnode = n->n + (n->num - reg->line[3]->nump);
	for(i = 0; i < reg->line[0]->nump; i++) {
		Add2Ilist(reg->nodes[3], (*tmpnode)->index);
		tmpnode++;
	}

#ifdef DEBUG_SS
	fprintf(fp,"ssle\n");
	DumpIlist2File(ssle, fp);
	fprintf(fp,"ssnod\n");
	DumpIlist2File(ssnod, fp);

	for(i = 0; i < reg->numl+1; i++) {
		fprintf(fp," ** reg->nodes[%d] **\n", i);
		DumpIlist2File(reg->nodes[i], fp);
	}
#endif

#ifdef DEBUG_SS
	fclose(fp);
#endif
	return(0);
}
