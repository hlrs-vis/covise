#include <stdio.h>
#include <stdlib.h>

#include <General/include/fatal.h>

#ifdef AXIAL_RUNNER
#include <AxialRunner/include/axial.h>
#endif
#ifdef RADIAL_RUNNER
#include <RadialRunner/include/radial.h>
#endif
#include <RunnerGridGen/include/rr_grid.h>
#include <RunnerGridGen/include/rr_meshinit.h>
#include <RunnerGridGen/include/rr_meshreg.h>
#include <RunnerGridGen/include/rr_meshnod.h>
#include <RunnerGridGen/include/rr_meshmisc.h>
#include <RunnerGridGen/include/new_meshreg.h>
#include <RunnerGridGen/include/new_meshnod.h>
#include <RunnerGridGen/include/mesh.h>

static int last_grid_err = 0;
static const char *grid_err_msg[] = {
    NULL,
    " Could not open grid parameter input file! Using default values.",
    " Grid type does not exist!.",
    " Need at least 4 nodes on boundary layer for this type of grid!",
    " Could not create grid. No geometry available!",
    " Not implemented yet!",
    " Trailing edge thickness might be 0 at one and > 0 at another part!"
};
#ifdef AXIAL_RUNNER
struct rr_grid *CreateAR_Mesh(struct axial *rr)
#elif RADIAL_RUNNER
struct rr_grid *CreateRR_Mesh(struct radial *rr)
#else
struct rr_grid *CreateRR_Mesh(struct radial *rr)
#endif
{

#ifdef AXIAL_RUNNER
    int i;
    float dh;
#endif

    static struct rr_grid *grid = NULL;

    last_grid_err = 0;
    if(!rr) {
	last_grid_err = NO_GEOMETRY_ERROR;
	return grid;
    }
    // **************************************************
    // get memory for grid, only if not available (first call)
    if(!grid) {
#ifdef DEBUG
	fprintf(stderr,"\n AllocRRGrid();\n\n");
#endif
	grid = AllocRRGrid();
	last_grid_err = InitRR_GridParams(grid);
    }

    // **************************************************
    // settings for grid regions, depending on runner type!
    grid->rot_clock = rr->rot_clockwise;
#ifdef AXIAL_RUNNER
    if(rr->mod->inl && rr->mod->bend)
	grid->iinlet = rr->p_hinlet->nump+rr->p_hbend->nump-1;
    else if(rr->mod->bend) grid->iinlet = rr->p_hbend->nump-1;
    else grid->iinlet = 0;

    if(rr->mod->outl) {
	grid->ioutlet = 0;
	dh = rr->p_hcore->z[rr->p_hcore->nump-1]+rr->h_run*rr->ref;
	for(i = rr->me[rr->be_num-1]->ml->p->nump-1; i > 0; i--,
		grid->ioutlet++) {
	    if((rr->me[rr->be_num-1]->ml->p->z[i] <= dh) &&
	       (rr->me[rr->be_num-1]->ml->p->z[i-1] >= dh)) break;
	}
    }
    else grid->ioutlet = 1;
#elif RADIAL_RUNNER
#ifndef NO_INLET_EXT
    grid->iinlet = NPOIN_EXT-1;
#else
    grid->iinlet = 0;
#endif
    grid->ioutlet = NPOIN_EXT;
#endif                                         // RUNNER_TYPE
    // **************************************************
    // init grid, memory
    InitRR_Grid(grid);
    // **************************************************
    // add gap regions, unused for long time!
    // CHECK before using this!
#ifdef GAP
    fprintf(stderr,"WARNING: GAP has not been used and maintained\n");
    fprintf(stderr,"for quite a while\n");
    fprintf(stderr,"CHECK with care!!!\n"); exit(1);
    AddGAP(rr->gp, grid);
#endif
    // **************************************************
    // different calls for different geometry types
#ifdef AXIAL_RUNNER
    InterpolMeridianPlanes(rr->me, rr->be_num, grid);
#elif RADIAL_RUNNER
    InterpolMeridianPlanes(rr->be, rr->be_num, grid);
#else
    InterpolMeridianPlanes(rr->be, rr->be_num, grid);
#endif                                         // RUNNER_TYPE
    // **************************************************
    // runner type independent
    TranslateBladeProfiles(grid);
    if(grid->type == CLASSIC) {
		if( (last_grid_err = CreateRR_GridRegions(rr->nob, grid))) {
			fprintf(stderr,"%s\n",GetLastGridErr());
			return grid;
		}
		MeshRR_GridRegions(grid);
    }
    else if(grid->type == MODIFIED) {
		if((last_grid_err = CreateNew_GridRegions(rr->nob, grid)) ) {
			fprintf(stderr,"%s\n",GetLastGridErr());
			return grid;
		}
		MeshNew_GridRegions(grid);
    }
    else if(grid->type == ISOMESH) {
		last_grid_err = IMPLEMENT_ERROR;
		GetLastGridErr();
		//CreateIso_GridRegions(rr->nob, grid);
		//MeshIso_GridRegions(grid);
		return grid;
    }
    else {
		last_grid_err = GRID_TYPE_ERROR;
    }
    fprintf(stderr," CreateRR_Mesh(): grid created!\n");
#ifdef PARA_OUT
    PutRR_GridParams(grid);
#endif
    return grid;
}


const char *GetLastGridErr()
{
    return grid_err_msg[last_grid_err];
}
