#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../General/include/flist.h"
#include "../General/include/ilist.h"
#include "../General/include/points.h"
#include "../General/include/curve.h"
#include "../General/include/nodes.h"
#include "../General/include/elements.h"
#include "../General/include/fatal.h"
#include "../General/include/v.h"

#include "include/rr_grid.h"
#include "include/rr_bcnodes.h"
#include "include/rr_elem.h"

#define NPOIN_EXT 10

#ifdef GAP
#include "include/rr_gapelem.h"
#endif

#ifndef SMALL
#define SMALL 1.0E-04
#endif
#define PERI_MOD_MATCH 1.0
#define PERI_MOD 0.0
#ifndef ABS
#define ABS(a)    ( (a) >= (0) ? (a) : -(a) )
#endif

// **************************************************
#ifdef BLNODES_OUT
int PutBladeNodes(struct Nodelist *n, struct Ilist *ssnod,
				  struct Ilist *psnod, int i, float para)
{
	int j, ssoffset, psoffset;
	int *nlist, *nprev;
	float len, dx, dy, dz;

	char  fn[200];
	FILE *fp;

	sprintf(fn,"rr_blnodes_%02d.dat",i);
	if( (fp = fopen(fn,"w+")) == NULL) {
		fprintf(stderr,"could not open file '%s'!\n", fn);
		exit(-1);
	}

	ssoffset = ssnod->num / (i+1);
	psoffset = psnod->num / (i+1);
	nlist = ssnod->list + (ssoffset * i);
	fprintf(fp,"%7d, %7d, %1.4f\n", ssoffset, psoffset, para);
	len = 0.0;
	fprintf(fp," %7d   %4.4f   %4.4f     %1.4f  %4.4f\n",
			n->n[(*nlist)]->id, len, n->n[(*nlist)]->l,
			n->n[(*nlist)]->phi, n->n[(*nlist)]->r);
	nprev = nlist;
	nlist++;
	for(j = 1; j < ssoffset; j++) {
		dx = n->n[(*nlist)]->x - n->n[(*nprev)]->x;
		dy = n->n[(*nlist)]->y - n->n[(*nprev)]->y;
		dz = n->n[(*nlist)]->z - n->n[(*nprev)]->z;
		len += sqrt(dx*dx + dy*dy + dz*dz);
		fprintf(fp," %7d   %4.4f   %4.4f     %1.4f  %4.4f\n",
				n->n[(*nlist)]->id, len, n->n[(*nlist)]->l,
				n->n[(*nlist)]->phi, n->n[(*nlist)]->r);fflush(fp);
		nprev = nlist;
		nlist++;
	}
	nlist = psnod->list + (psoffset * i);
	len = 0.0;
	fprintf(fp," %7d   %4.4f   %4.4f     %1.4f  %4.4f\n",
			n->n[(*nlist)]->id, len, n->n[(*nlist)]->l,
			n->n[(*nlist)]->phi, n->n[(*nlist)]->r);
	nprev = nlist;
	nlist++;
	for(j = 1; j < psoffset; j++) {
		dx = n->n[(*nlist)]->x - n->n[(*nprev)]->x;
		dy = n->n[(*nlist)]->y - n->n[(*nprev)]->y;
		dz = n->n[(*nlist)]->z - n->n[(*nprev)]->z;
		len += sqrt(dx*dx + dy*dy + dz*dz);
		fprintf(fp," %7d   %4.4f   %4.4f     %1.4f  %4.4f\n",
				n->n[(*nlist)]->id, len, n->n[(*nlist)]->l,
				n->n[(*nlist)]->phi, n->n[(*nlist)]->r);fflush(fp);
		nprev = nlist;
		nlist++;
	}
	return  0;
}
#endif

#ifdef READJUST_PERIODIC
#ifdef GAP
// function for period. readjustment for tip clearance grid, to be put here.
#else
// **************************************************
// check matching and non-matching periodic nodes
int ReadjustPeriodic(struct Nodelist *n, struct Ilist *psle,
					 struct Ilist *ssle,struct Ilist *pste,
					 struct Ilist *sste, int ge_num, int clock)
{
	int i, j, k, knext;
	int ssnum, psnum;

	int *iss, *ips;

	double dphi, invpara;
	float u[3], v[3];

	struct node **tmpps = NULL;
	struct node **tmpss = NULL;
	struct node **tmpssprev = NULL;

#ifdef DEBUG_PERIODIC
	char *fn = "rr_debugperiodic.txt";
	FILE *fp;

	if( (fp = fopen(fn,"w+")) == NULL ) {
		fprintf(stderr, "file '%s'\n", fn);
		exit(-1);
	}
	fprintf(stderr," ReadjustPeriodic:\n");
#endif

	// **************************************************
	// calc. periodic angle from nodes
	u[2] = v[2] = 0.0;

	u[0] = n->n[psle->list[0]]->x;
	u[1] = n->n[psle->list[0]]->y;
	v[0] = n->n[ssle->list[0]]->x;
	v[1] = n->n[ssle->list[0]]->y;
	dphi = 2.0 * M_PI/(int)(floor(2.0 * M_PI/(acos(V_Angle(u,v)))+0.5 ));
	if(clock) dphi *= -1.0;

#ifdef DEBUG_PERIODIC
	fprintf(fp,"acos(V_Angle(u,v)) = %f\n", acos(V_Angle(u,v)) * 180/M_PI);
	fprintf(fp, "psle[0] = %d, ssle[0] = %d, tmpps = %d, tmpss = %d\n"
			, psle->list[0], ssle->list[0],
			n->n[psle->list[0]]->index, n->n[ssle->list[0]]->index);
	fprintf(fp, "(*tmpps)->x = %f, (*tmpps)->y = %f, (*tmpss)->x = %f, (*tmpss)->y = %f, dphi = %f\n",
			n->n[psle->list[0]]->x,
			n->n[psle->list[0]]->y, n->n[ssle->list[0]]->x,
			n->n[ssle->list[0]]->y, dphi * 180/M_PI);
	fprintf(stderr,"dphi = %f\n", dphi *180/M_PI);
#endif

	// **************************************************
	// modify matching periodic nodes to fit better
	// recalc. ps-node's coord. from respective
	// ss node by rotation with dphi
	invpara = 1.0 - PERI_MOD_MATCH;
#ifdef DEBUG_PERIODIC
	fprintf(stderr," matching: invpara = %f\n",invpara);
#endif
	for(i = 0; i < ssle->num; i++) {
#ifdef DEBUG_PERIODIC
		u[0] = n->n[ssle->list[i]]->x;
		u[1] = n->n[ssle->list[i]]->y;
		v[0] = n->n[psle->list[i]]->x;
		v[1] = n->n[psle->list[i]]->y;
		fprintf(fp," ids: %d, %d, angle = %f, phips - phiss = %f\n",
				n->n[ssle->list[i]]->id, n->n[psle->list[i]]->id,
				180 / M_PI * acos(V_Angle(u,v)),
				180 / M_PI * (n->n[psle->list[i]]->phi - n->n[ssle->list[i]]->phi));
#endif
		u[0] = n->n[ssle->list[i]]->x;
		u[1] = n->n[ssle->list[i]]->y;

		v[0] = u[0]*cos(dphi) - u[1]*sin(dphi);
		v[1] = u[0]*sin(dphi) + u[1]*cos(dphi);
#ifdef DEBUG_PERIODIC
		fprintf(fp, "    v[0] = %f, v[1] = %f, x = %f, y = %f\n",
				v[0], v[1], n->n[psle->list[i]]->x,
				n->n[psle->list[i]]->y);
#endif
		n->n[psle->list[i]]->x = invpara * n->n[psle->list[i]]->x
			+ PERI_MOD_MATCH * v[0];
		n->n[psle->list[i]]->y = invpara * n->n[psle->list[i]]->y
			+ PERI_MOD_MATCH * v[1];
		n->n[psle->list[i]]->z = invpara * n->n[psle->list[i]]->z
			+ PERI_MOD_MATCH * n->n[ssle->list[i]]->z;

	}
	// **************************************************
	// modify non-matching nodes
	// interpolate node on ps from z coord. of ss
	// node and rotate this node to obtain new loc.
	// of ss node.
	ssnum = sste->num / ge_num;
	psnum = pste->num / ge_num;
	invpara = (1.0 - PERI_MOD);

#ifdef DEBUG_PERIODIC
	fprintf(fp," ssnum = %d, psnum = %d, invpara = %f\n", ssnum, psnum, invpara);
	fprintf(stderr," non-matching: invpara = %f\n",invpara);
#endif

	for(i = 0; i < ge_num; i++) {
		ips = &pste->list[psnum * i];
		knext = 1;
		for(j = 0; j < psnum; j++) {
			iss = (&sste->list[ssnum * i]) + knext;
			for(k = knext; k < ssnum; k++) {
				tmpps = n->n + (*ips);
				tmpss = n->n + (*iss);
				tmpssprev = n->n + (*(iss-1));
#ifdef DEBUG_PERIODIC
				fprintf(fp," i: %d, j: %d, k: %d:\n", i, j, k);
				fprintf(stderr," i: %d, j: %d, k: %d:\n", i, j, k);
				fprintf(stderr," tmpss, tmpps: %d, %d\n",(*tmpss)->id, (*tmpps)->id);
				fprintf(stderr," iss = %d, iss-1 = %d\n", (*iss), (*(iss-1)) );
				fprintf(stderr," sste->list[%d] = %d\n",ssnum * i + knext-1,
						sste->list[ssnum * i + knext-1]);
				fprintf(stderr," tmpssprev: %d\n",(*tmpssprev)->id);
#endif
				if( ((*tmpps)->z <= (*tmpssprev)->z) &&
					((*tmpps)->z >= (*tmpss)->z))
				{
					if( (knext = k - 1) <= 0) knext = 1;
					u[2] = (*tmpps)->z;
					u[0] = (*tmpssprev)->x +
						( (*tmpss)->x - (*tmpssprev)->x) / ((*tmpss)->z - (*tmpssprev)->z) *
						(u[2] - (*tmpssprev)->z);
					u[1] = (*tmpssprev)->y +
						( (*tmpss)->y - (*tmpssprev)->y) / ((*tmpss)->z - (*tmpssprev)->z) *
						(u[2] - (*tmpssprev)->z);
					v[0] = u[0]*cos(dphi) - u[1]*sin(dphi);
					v[1] = u[0]*sin(dphi) + u[1]*cos(dphi);
#ifdef DEBUG_PERIODIC
					fprintf(fp,"ssnode, psnode: %d, %d\n",(*tmpss)->id, (*tmpps)->id);
					fprintf(fp,"tmpps->z = %f, u[2] = %f, tmpps->x = %f, tmpps->y = %f\n",
							(*tmpps)->z, u[2], (*tmpps)->x, (*tmpps)->y);
					VPRINTF(v,fp);
					fprintf(stderr,"ssnode, psnode: %d, %d\n",(*tmpss)->id, (*tmpps)->id);
					fprintf(stderr,"tmpps->z = %f, u[2] = %f, tmpps->x = %f, tmpps->y = %f\n",
							(*tmpps)->z, u[2], (*tmpps)->x, (*tmpps)->y);
					VPRINT(v);
#endif
					(*tmpps)->x = invpara * ((*tmpps)->x) + PERI_MOD * v[0];
					(*tmpps)->y = invpara * ((*tmpps)->y) + PERI_MOD * v[1];
#ifdef DEBUG_PERIODIC
					fprintf(stderr," ... brech ...\n!");
#endif
					break;
				}
				iss++;
#ifdef DEBUG_PERIODIC
				fprintf(fp," i: %d, j: %d, k: %d:\n", i, j, k);
				fprintf(stderr,"end i: %d, j: %d, k: %d:\n", i, j, k);
#endif
				continue;
			}
			ips++;
		}                                           // end j, psnum

	}                                              // end i, ge_num

	// **************************************************
#ifdef DEBUG_PERIODIC
	fclose(fp);
#endif

	return 0;
}
#endif                                            // GAP
#endif                                            // READJUST_PERIODIC

// **************************************************
#ifdef DEBUG_NODES
int EquivCheck(struct node **n, int offset)
{
	int i, j;

	int equiv = 0;
	static int count = 0;

	struct node **nod = NULL;
	struct node **listnod = NULL;

	char fn[111];
	FILE *fp;

	sprintf(fn,"rr_equiv_%02d.txt",count++);
	if( (fp = fopen(fn,"w+")) == NULL) {
		fprintf(stderr,"Shit happened opening file '%s'!\n",fn);
		exit(-1);
	}

	nod = n;
	for(i = 0; i < offset-1; i++) {
		listnod = nod+1;
		for(j = i+1; j < offset; j++) {
			if( (ABS((*nod)->x - (*listnod)->x) < SMALL) &&
				(ABS((*nod)->y - (*listnod)->y) < SMALL) &&
				(ABS((*nod)->z - (*listnod)->z) < SMALL) ) {
				equiv++;
				fprintf(fp,"count = %d (merid. plane), equiv = %d\n",count,equiv);
				fprintf(fp," nod:     %8d   %10.6f   %10.6f   %10.6f\n", (*nod)->index,
						(*nod)->x, (*nod)->y, (*nod)->z);
				fprintf(fp," listnod: %8d   %10.6f   %10.6f   %10.6f\n", (*listnod)->index,
						(*listnod)->x, (*listnod)->y, (*listnod)->z);
				listnod++;
				break;
			}
			listnod++;
			continue;
		}
		nod++;
	}

	fprintf(fp,"equiv = %4d\n",equiv);
	fclose(fp);
	return(equiv);
}
#endif

// **************************************************
#ifdef DEBUG_BC
int DumpBoundaries(struct Nodelist *n, struct Ilist *inlet, struct Ilist *psle,
				   struct Ilist *ssle, struct Ilist *psnod, struct Ilist *ssnod,
				   struct Ilist *pste, struct Ilist *sste, struct Ilist *outlet,
				   int ge_num)
{
	int i, j;
	int jstart, jend;

	struct node **tmpnode;

	char fn[111];
	FILE *fp;

	for(i = 0; i < ge_num; i++) {
		sprintf(fn,"rr_debugbc_%02d.txt", i);
		if( (fp = fopen(fn,"w+")) == NULL) {
			fprintf(stderr,"Shit happened opening file '%s'!\n",fn);
			exit(-1);
		}

		fprintf(fp,"# inlet\n");
		jstart = i*(inlet->num/ge_num);
		jend   = (i+1)*(inlet->num/ge_num);
		for(j = jstart; j < jend; j++) {
			tmpnode = n->n + inlet->list[j];
			fprintf(fp," %10d    %10.6f  %10.6f  %10.6f  %10.6f      %10.6f  %10.6f  %10.6f\n",
					(*tmpnode)->id, (*tmpnode)->phi, (*tmpnode)->l, (*tmpnode)->r,
					(*tmpnode)->arc, (*tmpnode)->x, (*tmpnode)->y, (*tmpnode)->z);
		}
		fprintf(fp,"\n\n# psle\n");

		jstart = i*(psle->num/ge_num);
		jend   = (i+1)*(psle->num/ge_num);
		for(j = jstart; j < jend; j++) {
			tmpnode = n->n + psle->list[j];
			fprintf(fp," %10d    %10.6f  %10.6f  %10.6f  %10.6f      %10.6f  %10.6f  %10.6f\n",
					(*tmpnode)->id, (*tmpnode)->phi, (*tmpnode)->l, (*tmpnode)->r,
					(*tmpnode)->arc, (*tmpnode)->x, (*tmpnode)->y, (*tmpnode)->z);
		}
		fprintf(fp,"\n\n# ssle\n");

		jstart = i*(ssle->num/ge_num);
		jend   = (i+1)*(ssle->num/ge_num);
		for(j = jstart; j < jend; j++) {
			tmpnode = n->n + ssle->list[j];
			fprintf(fp," %10d    %10.6f  %10.6f  %10.6f  %10.6f      %10.6f  %10.6f  %10.6f\n",
					(*tmpnode)->id, (*tmpnode)->phi, (*tmpnode)->l, (*tmpnode)->r,
					(*tmpnode)->arc, (*tmpnode)->x, (*tmpnode)->y, (*tmpnode)->z);
		}
		fprintf(fp,"\n\n# psnod\n");

		jstart = i*(psnod->num/ge_num);
		jend   = (i+1)*(psnod->num/ge_num);
		for(j = jstart; j < jend; j++) {
			tmpnode = n->n + psnod->list[j];
			fprintf(fp," %10d    %10.6f  %10.6f  %10.6f  %10.6f      %10.6f  %10.6f  %10.6f\n",
					(*tmpnode)->id, (*tmpnode)->phi, (*tmpnode)->l, (*tmpnode)->r,
					(*tmpnode)->arc, (*tmpnode)->x, (*tmpnode)->y, (*tmpnode)->z);
		}
		fprintf(fp,"\n\n# ssnod\n");

		jstart = i*(ssnod->num/ge_num);
		jend   = (i+1)*(ssnod->num/ge_num);
		for(j = jstart; j < jend; j++) {
			tmpnode = n->n + ssnod->list[j];
			fprintf(fp," %10d    %10.6f  %10.6f  %10.6f  %10.6f      %10.6f  %10.6f  %10.6f\n",
					(*tmpnode)->id, (*tmpnode)->phi, (*tmpnode)->l, (*tmpnode)->r,
					(*tmpnode)->arc, (*tmpnode)->x, (*tmpnode)->y, (*tmpnode)->z);
		}
		fprintf(fp,"\n\n# pste\n");

		jstart = i*(pste->num/ge_num);
		jend   = (i+1)*(pste->num/ge_num);
		for(j = jstart; j < jend; j++) {
			tmpnode = n->n + pste->list[j];
			fprintf(fp," %10d    %10.6f  %10.6f  %10.6f  %10.6f      %10.6f  %10.6f  %10.6f\n",
					(*tmpnode)->id, (*tmpnode)->phi, (*tmpnode)->l, (*tmpnode)->r,
					(*tmpnode)->arc, (*tmpnode)->x, (*tmpnode)->y, (*tmpnode)->z);
		}
		fprintf(fp,"\n\n# sste\n");

		jstart = i*(sste->num/ge_num);
		jend   = (i+1)*(sste->num/ge_num);
		for(j = jstart; j < jend; j++) {
			tmpnode = n->n + sste->list[j];
			fprintf(fp," %10d    %10.6f  %10.6f  %10.6f  %10.6f      %10.6f  %10.6f  %10.6f\n",
					(*tmpnode)->id, (*tmpnode)->phi, (*tmpnode)->l, (*tmpnode)->r,
					(*tmpnode)->arc, (*tmpnode)->x, (*tmpnode)->y, (*tmpnode)->z);
		}
		fprintf(fp,"\n\n# outlet\n");

		jstart = i*(outlet->num/ge_num);
		jend   = (i+1)*(outlet->num/ge_num);
		for(j = jstart; j < jend; j++) {
			tmpnode = n->n + outlet->list[j];
			fprintf(fp," %10d    %10.6f  %10.6f  %10.6f  %10.6f      %10.6f  %10.6f  %10.6f\n",
					(*tmpnode)->id, (*tmpnode)->phi, (*tmpnode)->l, (*tmpnode)->r,
					(*tmpnode)->arc, (*tmpnode)->x, (*tmpnode)->y, (*tmpnode)->z);
		}
		fclose(fp);
	}                                              // end i

	return 0;
}
#endif
