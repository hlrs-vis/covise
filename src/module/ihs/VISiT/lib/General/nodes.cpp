#include <stdlib.h>
#include <math.h>
#include "include/fatal.h"
#include "include/curve.h"
#include "include/flist.h"
#include "include/points.h"
#include "include/nodes.h"

#define TINY 1.0E-8

struct Nodelist *AllocNodelistStruct(void)
{
	struct Nodelist *n;

	if( (n = (struct Nodelist *)calloc(1,sizeof(struct Nodelist))) == NULL) {
		fatal("memory for (struct Nodelist)!");
	}
	n->portion = 1000;

	return (n);
}


int AddNode(struct Nodelist *n, float arc, float l, float r, int flag)
{

	int i;

	if(n->num == n->max) {
		n->max += n->portion;
		if( (n->n = (struct node **)realloc(n->n, n->max*sizeof(struct node *))) == NULL) {
			fatal("memory for realloc(struct node*)!");
		}

		if( (n->n[n->num] = (struct node *)calloc(n->portion,sizeof(struct node))) == NULL) {
			fatal("memory for calloc(struct node*)!");
		}

		for(i = n->num+1; i < n->max; i++) {
			n->n[i] = n->n[i-1]+1;
		}
	}
	n->n[n->num]->index = n->num;
	n->n[n->num]->id    = n->num+1;
	if(flag == ARC) {
		n->n[n->num]->arc   = arc;
	}
	else {
		n->n[n->num]->phi   = arc;
	}
	n->n[n->num]->l     = l;
	n->n[n->num]->r     = r;
	n->num++;

	return (n->num-1);
}


int AddVNode(struct Nodelist *n, float x[3], int flag)
{
	return (AddNode(n, x[0], x[1], x[2], flag));
}


struct Nodelist *GetNodelistMemory(struct Nodelist *n)
{
	if(n) {
		FreeNodelistStruct(n);
		n = NULL;
	}
	return(AllocNodelistStruct());
}


void FreeNodelistStruct(struct Nodelist *n)
{
	int i;

	if(n) {
		for(i = 0; i < n->max; i += n->portion) {
			if(n->n[i]) free(n->n[i]);
		}
		free(n->n);
		free(n);
		return;
	}
}


int CalcNodeRadius(struct node **n, struct curve *ml, int nnum)
{
	int i, j, jnext;

	for (i = 0; i < nnum; i++) {
		jnext = 1;
		for (j = jnext; j < ml->p->nump; j++) {
			if ((ml->len[j-1] <= n[i]->l) &&
				(ml->len[j] >= n[i]->l)) {
				jnext = j - 1;
				n[i]->r = (ml->p->x[j-1] - ml->p->x[j]) / (ml->len[j-1] - ml->len[j])
					* (n[i]->l - ml->len[j]) + ml->p->x[j];
				n[i]->z = (ml->p->z[j-1] - ml->p->z[j]) / (ml->len[j-1] - ml->len[j])
					* (n[i]->l - ml->len[j]) + ml->p->z[j];
				if(!n[i]->arc) n[i]->arc = n[i]->r * n[i]->phi;
#ifdef DEBUG_NODES
				if(n[i]->r < TINY) {
					fprintf(stderr,"n[i]->r = %f, n[i]->l = %f, ml->len[j] = %f\n",n[i]->r, n[i]->l ,ml->len[j] );
				}
#endif
				break;
			}
			continue;
		}
	}
	return 0;
}


int CalcNodeAngle(struct node **n, struct curve * /*ml*/, int nnum)
{
	int i;

	for(i = 0; i < nnum; i++) {
		if(n[i]->r < TINY) {
			fprintf(stderr," CalcNodeAngle: Warning: node %d, zero radius %1.3e!\n", n[i]->id, n[i]->r);
			n[i]->phi = 0.0;
		}
		else {
			n[i]->phi = n[i]->arc/n[i]->r;
		}
	}

	return 0 ;
}


int CalcNodeCoords(struct node **n, int nnum, int clock)
{
	int i;

	if(clock) {
		for(i = 0; i < nnum; i++) {
			n[i]->x = n[i]->r * float(cos(n[i]->phi));
			n[i]->y = -n[i]->r * float(sin(n[i]->phi));
		}

	}
	else {
		for(i = 0; i < nnum; i++) {
			n[i]->x = n[i]->r * float(cos(n[i]->phi));
			n[i]->y = n[i]->r * float(sin(n[i]->phi));
		}
	}
	return 0;
}


#ifdef DEBUG_NODES
int DumpNodes(struct node **n, int nnum, FILE *fp)
{
	int i;

	fprintf(fp,"#   ID\t phi\t l\t r\t s\t\t x\t y\t z\t\n");
	for(i = 0; i < nnum; i++) {
		if(!n[i]->arc) {
			n[i]->arc = n[i]->phi * n[i]->r;
		}
		fprintf(fp," %10d   %10.6f  %10.6f  %10.6f  %10.6f     %10.6f  %10.6f  %10.6f\n",
				n[i]->id, n[i]->phi, n[i]->l, n[i]->r, n[i]->arc,
				n[i]->x, n[i]->y, n[i]->z);
	}
	return 0;
}
#endif
