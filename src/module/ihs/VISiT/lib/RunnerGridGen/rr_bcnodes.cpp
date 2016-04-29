#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../General/include/flist.h"
#include "../General/include/ilist.h"
#include "../General/include/points.h"
#include "../General/include/nodes.h"
#include "../General/include/elements.h"
#include "../General/include/fatal.h"

#include "include/rr_grid.h"

#ifndef SMALL
#define SMALL 1.0E-04
#endif
#ifndef ABS
#define ABS(a)    ( (a) >= (0) ? (a) : -(a) )
#endif

#ifndef SET
#define SET 1
#endif

#ifdef DEBUG_ELEMENTS
#define VPRINT(x) fprintf(stderr,"%10s = [%4f, %4f, %4f]\n",#x,x[0], x[1], x[2])
#define VPRINTF(x,f) fprintf(f,"%10s = [%4f, %4f, %4f]\n",#x,x[0], x[1], x[2])
#endif

int *GetBCNodes(struct Ilist *nodes, int nnum)
{
	int i;

	int *bcnodes;

	if( (bcnodes = (int*)calloc(nnum, sizeof(int))) == NULL) {
		fatal("memory for (int)!");
		exit(-1);
	}

	for(i = 0; i < nodes->num; i++) {
		bcnodes[nodes->list[i]] = SET;
	}

	return (&bcnodes[0]);
}


// **************************************************
int AllocBCNodesMemory(struct rr_grid *grid)
{
	grid->inlet  = AllocIlistStruct(grid->cdis+1);
	grid->outlet = AllocIlistStruct(grid->cdis+1);
	grid->psle   = AllocIlistStruct(grid->ssmdis+1);
	grid->ssle   = AllocIlistStruct(grid->ssmdis+1);
	grid->psnod  = AllocIlistStruct(grid->psdis+1);
	grid->ssnod  = AllocIlistStruct(grid->ssdis+1);
	grid->pste   = AllocIlistStruct(grid->ssdis+1);
	grid->sste   = AllocIlistStruct(grid->lowdis+1);

	return(0);
}


// **************************************************
int FreeBCNodesMemory(struct rr_grid *grid)
{
	if(grid->inlet) {
		FreeIlistStruct(grid->inlet);
		grid->inlet = NULL;
	}
	if(grid->outlet) {
		FreeIlistStruct(grid->outlet);
		grid->outlet = NULL;
	}
	if(grid->psle) {
		FreeIlistStruct(grid->psle);
		grid->psle = NULL;
	}
	if(grid->ssle) {
		FreeIlistStruct(grid->ssle);
		grid->ssle = NULL;
	}
	if(grid->psnod) {
		FreeIlistStruct(grid->psnod);
		grid->psnod = NULL;
	}
	if(grid->ssnod) {
		FreeIlistStruct(grid->ssnod);
		grid->ssnod = NULL;
	}
	if(grid->pste) {
		FreeIlistStruct(grid->pste);
		grid->pste = NULL;
	}
	if(grid->sste) {
		FreeIlistStruct(grid->sste);
		grid->sste = NULL;
	}

	return(0);
}
