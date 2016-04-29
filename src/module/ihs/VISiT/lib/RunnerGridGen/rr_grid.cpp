#include <stdlib.h>
#include <stdio.h>

#include "../General/include/curve.h"
#include "../General/include/points.h"
#include "../General/include/flist.h"
#include "../General/include/ilist.h"
#include "../General/include/nodes.h"
#include "../General/include/elements.h"
#include "../General/include/fatal.h"

#include "include/rr_grid.h"

struct region **AllocRRGridRegions(int reg_num, int numl)
{
    int i;

    struct region **reg = NULL;

    if( (reg = (struct region**)calloc(reg_num, sizeof(struct region*))) == NULL) {
	fatal("memory for (struct region*)!");
    }

    if( (reg[0] = (struct region*)calloc(reg_num, sizeof(struct region))) == NULL) {
	fatal("memory for (struct region)!");
    }
    for(i = 1; i < reg_num; i++) {
	reg[i] = reg[i-1]+1;
    }

    for(i = 0; i < reg_num; i++) {

	reg[i]->numl = numl;
	if( (reg[i]->line = (struct Point **)calloc(reg[i]->numl, sizeof(struct Point*))) == NULL)
	{
	    fatal("memory for (struct Point*)!");
	}
	if( (reg[i]->arc = (struct Flist **)calloc(reg[i]->numl, sizeof(struct Flist*))) == NULL)
	{
	    fatal("memory for (struct Flist*)!");
	}
	if( (reg[i]->para = (struct Flist **)calloc(reg[i]->numl, sizeof(struct Flist*))) == NULL)
	{
	    fatal("memory for (struct Flist*)!");
	}
	if( (reg[i]->nodes = (struct Ilist **)calloc(reg[i]->numl+1, sizeof(struct Ilist*))) == NULL)
	{
	    fatal("memory for (struct Ilist*)!");
	}
    }
    return (reg);
}


void FreeRRGridRegions(int reg_num, struct region **reg)
{
    int i, j;

    if(reg) {
	for(i = 0; i < reg_num; i++)
	{
	    if(reg[i])
	    {
		for(j = 0; j < reg[i]->numl; j++)
		{
		    if(reg[i]->line[j])
		    {
			FreePointStruct(reg[i]->line[j]);
		    }
		    if(reg[i]->arc[j])
		    {
			FreeFlistStruct(reg[i]->arc[j]);
		    }
		    if(reg[i]->para[j])
		    {
			FreeFlistStruct(reg[i]->para[j]);
		    }
		    if(reg[i]->nodes[j])
		    {
			FreeIlistStruct(reg[i]->nodes[j]);
		    }
		}
		if(reg[i]->nodes[reg[i]->numl])
		{
		    FreeIlistStruct(reg[i]->nodes[reg[i]->numl]);
		}
		free(reg[i]->line);
		free(reg[i]->para);
		free(reg[i]->arc);
		free(reg[i]->nodes);
	    }
	}
	free(reg);
    }
    return;
}


struct ge **AllocRRGridElements(int ge_num)
{
    int i;
    struct ge **ge = NULL;

    if( (ge = (struct ge**)calloc(ge_num,sizeof(struct ge*))) == NULL) {
	fatal("memory for (struct ge*)!");
    }
    if ( (ge[0] = (struct ge*)calloc(ge_num,sizeof(struct ge))) == NULL) {
	fatal("memory for (struct ge)!");
    }
    for(i = 1; i < ge_num; i++) {
	ge[i] = ge[i-1]+1;
    }
    return (ge);
}


struct cgrid **AllocRRCGridElements(int ge_num)
{
    int i;
    struct cgrid **cge = NULL;

    if( (cge = (struct cgrid**)calloc(ge_num,sizeof(struct cgrid*))) == NULL) {
	fatal("memory for (struct cgrid*)!");
    }
    if ( (cge[0] = (struct cgrid*)calloc(ge_num,sizeof(struct cgrid))) == NULL) {
	fatal("memory for (struct cgrid)!");
    }
    for(i = 1; i < ge_num; i++) {
	cge[i] = cge[i-1]+1;
    }
    return (cge);
}


void FreeRRGridElements(int ge_num, struct ge **ge)
{
    int i;

    if(ge) {
	for(i = 0; i < ge_num; i++)
	{
	    if(ge[i])
	    {
		if(ge[i]->ml)     FreeCurveStruct(ge[i]->ml);
		if(ge[i]->cl)     FreePointStruct(ge[i]->cl);
		if(ge[i]->ps)     FreePointStruct(ge[i]->ps);
		if(ge[i]->ss)     FreePointStruct(ge[i]->ss);
	    }
	}
	free(ge);
    }
    return;
}


void FreeRRCGridElements(int ge_num, struct cgrid **cge)
{
    int i;

    if(cge) {
	for(i = 0; i < ge_num; i++)
	{
	    if(cge[i])
	    {
		if(cge[i]->cl)     FreePointStruct(cge[i]->cl);
		if(cge[i]->clarc)  FreeFlistStruct(cge[i]->clarc);
		if(cge[i]->ps)     FreeCurveStruct(cge[i]->ps);
		if(cge[i]->ss)     FreeCurveStruct(cge[i]->ss);
	    }
	}
	free(cge);
    }
    return;
}


struct rr_grid *AllocRRGrid()
{
    struct rr_grid *grid = NULL;

    if((grid=(struct rr_grid*)calloc(1,sizeof(struct rr_grid))) == NULL) {
	fatal("memory for (struct rr_grid)!");
	exit(1);
    }
    if( (grid->inbc =(struct bc*)calloc(1,sizeof(struct bc))) == NULL) {
	fatal(" no memory for calloc(1,(struct bc))!");
	exit(1);
    }
    return grid;
}


int FreeRRGrid(struct rr_grid *grid)
{
    int i;
    int FreeBCNodesMemory(struct rr_grid *grid);

    if(grid) {
	for(i = 0; i < grid->ge_num; i++)
	{
	    FreeRRGridRegions(grid->cge[i]->reg_num,
			      grid->cge[i]->reg);
	}
	FreeRRCGridElements(grid->ge_num, grid->cge);
	FreeBCNodesMemory(grid);
	FreeNodelistStruct(grid->n);
	FreeElementStruct(grid->e);
	FreeElementStruct(grid->wall);
	FreeElementStruct(grid->shroud);
	FreeElementStruct(grid->shroudext);
	FreeElementStruct(grid->psblade);
	FreeElementStruct(grid->ssblade);
	FreeElementStruct(grid->psleperiodic);
	FreeElementStruct(grid->ssleperiodic);
	FreeElementStruct(grid->psteperiodic);
	FreeElementStruct(grid->ssteperiodic);
	FreeElementStruct(grid->einlet);
	FreeElementStruct(grid->eoutlet);
	FreeElementStruct(grid->frictless);
	if(grid->bcval)
	{
	    free(grid->bcval[0]);
	    free(grid->bcval);
	}
#ifdef RR_IONODES
	FreeElementStruct(grid->rrinlet);
	FreeElementStruct(grid->rroutlet);
#endif
	free(grid->inbc);
	free(grid);
    }
    return 0;
}


int FreeRRGridMesh(struct rr_grid *grid)
{
    int i;
    int FreeBCNodesMemory(struct rr_grid *grid);

    fprintf(stderr,"FreeRRGridMesh()\n");
    if(grid) {
	fprintf(stderr,"grid...exists!\n");
	for(i = 0; i < grid->ge_num; i++) {
	    FreeRRGridRegions(grid->cge[i]->reg_num,
			      grid->cge[i]->reg);
	}
	FreeRRCGridElements(grid->ge_num, grid->cge);
	FreeBCNodesMemory(grid);
	FreeNodelistStruct(grid->n);
	FreeElementStruct(grid->e);
	FreeElementStruct(grid->wall);
	FreeElementStruct(grid->shroud);
	FreeElementStruct(grid->shroudext);
	FreeElementStruct(grid->psblade);
	FreeElementStruct(grid->ssblade);
	FreeElementStruct(grid->psleperiodic);
	FreeElementStruct(grid->ssleperiodic);
	FreeElementStruct(grid->psteperiodic);
	FreeElementStruct(grid->ssteperiodic);
	FreeElementStruct(grid->einlet);
	FreeElementStruct(grid->eoutlet);
	FreeElementStruct(grid->frictless);
	if(grid->bcval)	{
	    fprintf(stderr,"free bc values!\n");
	    free(grid->bcval[0]);
	    free(grid->bcval);
	    grid->bcval = NULL;
	}
#ifdef RR_IONODES
	FreeElementStruct(grid->rrinlet);
	FreeElementStruct(grid->rroutlet);
#endif
    }
    fprintf(stderr," grid freed\n");fflush(stderr);
    return 0;
}
