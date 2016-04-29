#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <General/include/flist.h>
#include <General/include/points.h>
#include <General/include/curve.h>
#include <General/include/profile.h>
#include <General/include/bias.h>
#include <General/include/fatal.h>
#include <General/include/v.h>

#include <RunnerGridGen/include/rr_grid.h>
#include <RunnerGridGen/include/rr_meshmisc.h>

// needed because of struct be (radial) || struct meridian (axial) definition
#ifdef AXIAL_RUNNER
#include <AxialRunner/include/axial.h>
#endif
#ifdef RADIAL_RUNNER
#include <RadialRunner/include/radial.h>
#endif
#ifdef DIAGONAL_RUNNER
#include <RadialRunner/include/diagonal.h>
#endif

#ifdef ONLY_GGEN
#define DEFAULT_GRIDTYPE CLASSIC
#define MESH_EXT_INIT 1
#else
#define DEFAULT_GRIDTYPE CLASSIC
#define MESH_EXT_INIT 0
#endif

int InitRR_GridParams(struct rr_grid *grid)
{
	int err = 0;

	grid->type = DEFAULT_GRIDTYPE;
	grid->mesh_ext = MESH_EXT_INIT;
	
	// init discretization and domain pars
	grid->ge_num  =	 6;
	grid->ge_bias = -3.0;
	grid->ge_type =	 2;
#ifdef GAP
	grid->gp_num  =	 3;
	grid->gp_bias = -2.0;
	grid->gp_type =	 2;
	grid->gpreg_num = 2;
#endif
	grid->reg_num = 8;
	grid->numl	  = 4;

	// scaling factors and curve partition factors
	grid->phi_scale[0] = 0.5f;
	grid->phi_scale[1] = 0.5f;
	grid->phi0_ext		=  1.0f;					   // 1.0 for radial inlet ext.
	grid->angle_ext[0]	=  0.2f;					   // spline tangent angle (ss)
	grid->angle_ext[1]	=  1.0f;					   // (ps)
	grid->v14_angle[0] = 0.2f;
	grid->v14_angle[1] = 0.2f;
	grid->bl_v14_part[0] = 0.03f;
	grid->bl_v14_part[1] = 0.97f;
	grid->ssle_part[0] = 0.03f;
	grid->ssle_part[1] = 0.015f;
	grid->psle_part[0] = 0.04f;
	grid->psle_part[1] = 0.02f;
	grid->bl_scale[0]  = 0.95f;
	grid->bl_scale[1]  = 0.95f;
	grid->ss_part[0]  = 0.85f;
	grid->ss_part[1]  = 0.75f;
	grid->ps_part[0]  = 0.4f;
	grid->ps_part[1]  = 0.45f;
	grid->out_part[0] = 0.2f;

	// initialization of discretization pars, change to read pars.
	grid->extdis		=  4;
	grid->extbias		= -2.0;
	grid->extbias_type	=  2;
	grid->cdis			= 6;
	grid->cbias			= -2.0;
	grid->cbias_type	= 2;
	grid->cledis		= grid->cdis;
	grid->clebias		= -3.0;
	grid->clebias_type	= 2;
	grid->ssmdis		= 4;
	grid->ssmbias		= 2.0;
	grid->ssmbias_type	= 1;
	grid->psdis			= 11;
	grid->psbias		= -3.0;
	grid->psbias_type	= 2;
	grid->psedis		= 3;
	grid->psebias		= 2.0;
	grid->psebias_type	= 1;
	grid->ssdis			= 16;
	grid->ssbias		= -2.0;
	grid->ssbias_type	= 2;
	grid->midbias		= -3.0;
	grid->midbias_type	= 2;
	grid->lowdis		= 6;
	grid->lowbias		= -2.0;
	grid->lowbias_type	= 1;
	grid->lowinbias		= grid->lowbias;
	grid->lowin_type	= grid->lowbias_type;
	grid->ssxbias		= 1.0;
	grid->ssxbias_type	= 0;
	grid->psxbias		= 1.0;
	grid->psxbias_type	= 0;
	grid->cxbias		= 1.0;
	grid->cxbias_type	= 0;

#ifdef PARA_IN
	err = GetRR_GridParams(grid);
#endif
#ifdef PARA_OUT
	PutRR_GridParams(grid);
#endif

	return err;
}


//**************************************************

int InitRR_Grid(struct rr_grid *grid)
{
	int i;
	struct Flist *bias;

	if(grid->iinlet == 0) grid->mesh_ext = 0;	   // do not change!!
	if(grid->mesh_ext) grid->reg_num = 8;
	else grid->reg_num = 7;

#ifdef GAP
	grid->ge  = AllocRRGridElements(grid->ge_num + grid->gp_num - 1);
	grid->cge = AllocRRCGridElements(grid->ge_num + grid->gp_num - 1);
#else
	grid->ge  = AllocRRGridElements(grid->ge_num);
	grid->cge = AllocRRCGridElements(grid->ge_num);
#endif
	bias = CalcBladeElementBias(grid->ge_num, 0.0, 1.0, grid->ge_type, grid->ge_bias);
	for(i = 0; i < grid->ge_num; i++) {
		grid->ge[i]->para = bias->list[i];
		grid->cge[i]->reg_num = grid->reg_num;
#ifdef GAP
		if(i == grid->ge_num-1) {
			grid->cge[i]->reg_num += grid->gpreg_num;
		}
#endif
		grid->cge[i]->reg = AllocRRGridRegions(grid->cge[i]->reg_num, grid->numl);
	}
	FreeFlistStruct(bias);
	return 0;
}


//**************************************************
#ifdef GAP
int AddGAP(struct be *gp, struct rr_grid *grid)
{
	int i;
	struct Flist *bias;
	struct ge *getmp;
	struct cgrid *cgetmp;

	bias = CalcBladeElementBias(grid->gp_num, 1.0, gp->para, grid->gp_type, grid->gp_bias);
	getmp  = grid->ge[grid->ge_num];
	cgetmp = grid->cge[grid->ge_num];
	for(i = 1; i < grid->gp_num; i++) {
		getmp->para = bias->list[i];
		cgetmp->reg_num = grid->reg_num + grid->gpreg_num;
		cgetmp->reg = AllocRRGridRegions(cgetmp->reg_num, grid->numl);
		getmp++; cgetmp++;
	}
	grid->ge_num += grid->gp_num-1;
	FreeFlistStruct(bias);
	return 0;
}
#endif

//**************************************************
#ifdef AXIAL_RUNNER
int InterpolMeridianPlanes(struct meridian**be,int be_num,struct rr_grid *grid)

#elif RADIAL_RUNNER
	int InterpolMeridianPlanes(struct be **be, int be_num, struct rr_grid *grid)
#endif
{
	int i, j, k, jnext;
	int ibe_max, clockfactor;

	float t;
	float v[3];
	float p[3];
#ifdef DEBUG_INTERPOL
	char fn[111];
	FILE *fp;
#endif

	v[0] = v[1] = v[2] = 0.0;
	p[0] = p[1] = p[2] = 0.0;
	ibe_max = be_num - 1;
#ifdef GAP
	be_num++;
#endif

	if(grid->rot_clock) clockfactor = -1;
	else clockfactor = 1;

	for(i = 0; i < be[0]->ml->p->nump; i++) {
		if(i == 0) {
			for(j = 0; j < grid->ge_num; j++) {
				grid->ge[j]->ml = GetCurveMemory(grid->ge[j]->ml);
			}
		}
		v[0] = be[ibe_max]->ml->p->x[i] - be[0]->ml->p->x[i];
		v[1] = be[ibe_max]->ml->p->y[i] - be[0]->ml->p->y[i];
		v[2] = be[ibe_max]->ml->p->z[i] - be[0]->ml->p->z[i];
		for(j = 0; j < grid->ge_num; j++) {
			p[0] = be[0]->ml->p->x[i] + grid->ge[j]->para * v[0];
			p[1] = be[0]->ml->p->y[i] + grid->ge[j]->para * v[1];
			p[2] = be[0]->ml->p->z[i] + grid->ge[j]->para * v[2];
			AddCurvePoint(grid->ge[j]->ml, p[0], p[1], p[2], 0.0,
						  grid->ge[j]->para);
		}
	}											   // end i
	for(i = 0; i < grid->ge_num; i++) {
		CalcCurveArclen(grid->ge[i]->ml);
	}

	for(i = 0; i < grid->ge_num; i++) {
		jnext = 1;
		grid->ge[i]->cl = GetPointMemory(grid->ge[i]->cl);
		grid->ge[i]->ps = GetPointMemory(grid->ge[i]->ps);
		grid->ge[i]->ss = GetPointMemory(grid->ge[i]->ss);
		for(j = jnext; j < be_num; j++) {
			if(grid->ge[i]->para <= be[j]->para) {
				jnext = j;
				t = (grid->ge[i]->para - be[j-1]->para)/
					(be[j]->para - be[j-1]->para);
				for(k = 0; k < be[j]->cl->nump; k++)
				{
					v[0] = be[j]->cl->x[k] - be[j-1]->cl->x[k];
					v[1] = be[j]->cl->y[k] - be[j-1]->cl->y[k];
					v[2] = be[j]->cl->z[k] - be[j-1]->cl->z[k];
					p[0] = be[j-1]->cl->x[k] + t*v[0];
					p[1] = be[j-1]->cl->y[k] + t*v[1];
					p[2] = be[j-1]->cl->z[k] + t*v[2];
					p[1] *= clockfactor;
					AddVPoint(grid->ge[i]->cl, p);
				}
				for(k = 0; k < be[j]->ps->nump; k++)
				{
					v[0] = be[j]->ps->x[k] - be[j-1]->ps->x[k];
					v[1] = be[j]->ps->y[k] - be[j-1]->ps->y[k];
					v[2] = be[j]->ps->z[k] - be[j-1]->ps->z[k];
					p[0] = be[j-1]->ps->x[k] + t*v[0];
					p[1] = be[j-1]->ps->y[k] + t*v[1];
					p[2] = be[j-1]->ps->z[k] + t*v[2];
					p[1] *= clockfactor;
					AddVPoint(grid->ge[i]->ps, p);
				}
				for(k = 0; k < be[j]->ss->nump; k++)
				{
					v[0] = be[j]->ss->x[k] - be[j-1]->ss->x[k];
					v[1] = be[j]->ss->y[k] - be[j-1]->ss->y[k];
					v[2] = be[j]->ss->z[k] - be[j-1]->ss->z[k];
					p[0] = be[j-1]->ss->x[k] + t*v[0];
					p[1] = be[j-1]->ss->y[k] + t*v[1];
					p[2] = be[j-1]->ss->z[k] + t*v[2];
					p[1] *= clockfactor;
					AddVPoint(grid->ge[i]->ss, p);
				}									  // end k
#ifdef DEBUG_INTERPOL
				fprintf(stderr,"i, j: %d, %d: t = %f\n", i,j,t);
#endif
				break;
			}										 // end if
			continue;
		}											// end j
	}											   // end i

#ifdef DEBUG_INTERPOL
	for(i = 0; i < grid->ge_num; i++) {
		sprintf(fn,"rr_meridintpol_%02d.txt",i);
		if( (fp = fopen(fn,"w+")) == NULL) {
			fprintf(stderr,"file '%s'!\n",fn);
			exit(-1);
		}
		fprintf(fp,"# grid->ge[%02d]->ml->p->nump = %d\n", i,grid->ge[i]->ml->p->nump);
		DumpCurve(grid->ge[i]->ml, fp);

		fprintf(fp,"\n\n");
		fprintf(fp,"# grid->ge[%02d]->cl->nump = %d\n", i, grid->ge[i]->cl->nump);
		for(j = 0; j < grid->ge[i]->cl->nump; j++) {
			fprintf(fp," %f	  %f   %f\n",grid->ge[i]->cl->x[j],
					grid->ge[i]->cl->y[j] ,grid->ge[i]->cl->z[j]);
		}
		fprintf(fp,"\n\n");
		fprintf(fp,"# grid->ge[%02d]->ps->nump = %d\n", i, grid->ge[i]->ps->nump);
		for(j = 0; j < grid->ge[i]->ps->nump; j++) {
			fprintf(fp," %f	  %f   %f\n",grid->ge[i]->ps->x[j],
					grid->ge[i]->ps->y[j] ,grid->ge[i]->ps->z[j]);
		}
		fprintf(fp,"\n\n");
		fprintf(fp,"# grid->ge[%02d]->ss->nump = %d\n", i, grid->ge[i]->ss->nump);
		for(j = 0; j < grid->ge[i]->ss->nump; j++) {
			fprintf(fp," %f	  %f   %f\n",grid->ge[i]->ss->x[j],
					grid->ge[i]->ss->y[j] ,grid->ge[i]->ss->z[j]);
		}
		fclose(fp);
	}
#endif

	return 0;
}


// translate blade profile data to meridian plane
// r, phi, z --> phi, l, r
int TranslateBladeProfiles(struct rr_grid *grid)
{
	int i;

	float dphi = 0.0;

	extern int InterpolPoint(struct Point *ml, float *len, float dphi,
							 struct Point *cl, struct Point *line);
	extern int InterpolCurve(struct Point *ml, float *len, float dphi,
							 struct Point *cl, struct curve *c);

#ifdef DEBUG_REGIONS
	int j;

	char fn[150];
	FILE *fp;
#endif

	// memory check
	for(i = 0; i < grid->ge_num; i++) {
		if(grid->cge[i]->clarc) {
			FreeFlistStruct(grid->cge[i]->clarc);
			grid->cge[i]->clarc = NULL;
		}
		grid->cge[i]->cl = GetPointMemory(grid->cge[i]->cl);
		grid->cge[i]->ss = GetCurveMemory(grid->cge[i]->ss);
		grid->cge[i]->ps = GetCurveMemory(grid->cge[i]->ps);
	}											   //end i, grid->ge_num

	// transform blade profiles to meridian plane
	for(i = 0; i < grid->ge_num; i++) {			   // loop over merid. planes
		InterpolPoint(grid->ge[i]->ml->p, grid->ge[i]->ml->len, dphi, grid->ge[i]->cl, grid->cge[i]->cl);
		InterpolCurve(grid->ge[i]->ml->p, grid->ge[i]->ml->len, dphi, grid->ge[i]->ss, grid->cge[i]->ss);
		InterpolCurve(grid->ge[i]->ml->p, grid->ge[i]->ml->len, dphi, grid->ge[i]->ps, grid->cge[i]->ps);

		grid->cge[i]->clarc = GetCircleArclen(grid->cge[i]->cl);
	}

#ifdef DEBUG_REGIONS
	for(i = 0; i < grid->ge_num; i++) {			   // loop over merid. planes
		sprintf(fn,"rr_blademerid_%02d.txt",i);
		if( (fp = fopen(fn,"w+")) == NULL) {
			fprintf(stderr, "Shit happenend opening file '%s'!\n",fn);
			exit(-1);
		}
		fprintf(fp,"# ss->nump = %d, ps->nump = %d\n", grid->ge[i]->ss->nump, grid->ge[i]->ps->nump);
		for( j = 0; j < grid->ge[i]->cl->nump; j++) {
			fprintf(fp, "%8.6f	%8.6f  %8.6f\n",
					grid->cge[i]->cl->x[j], grid->cge[i]->cl->y[j], grid->cge[i]->cl->z[j]);
		}
		fprintf(fp,"\n\n");
		fprintf(fp,"# ss->p->nump = %d\n",grid->cge[i]->ss->p->nump);
		DumpCurve(grid->cge[i]->ss, fp);
		fprintf(fp,"\n\n");
		fprintf(fp,"# ps->p->nump = %d\n",grid->cge[i]->ps->p->nump);
		DumpCurve(grid->cge[i]->ps, fp);
		fclose(fp);
	}											   // end i
#endif

	return 0;
}
