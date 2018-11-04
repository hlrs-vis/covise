#include <stdio.h>
#include <stdlib.h>
#ifdef WIN32
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <windows.h>
#else 
#include <strings.h>
#endif
#include <math.h>
#include "../General/include/geo.h"

#ifdef RADIAL_RUNNER
#include "include/radial.h"
#endif
#include "../General/include/points.h"
#include "../General/include/curve.h"
#include "../General/include/flist.h"
#include "../General/include/plane_geo.h"
#include "../General/include/parameter.h"
#include "../General/include/profile.h"
#include "../General/include/bias.h"
#include "../General/include/curvepoly.h"
#include "../BSpline/include/bspline.h"
#include "../General/include/common.h"
#include "../General/include/log.h"
#include "../General/include/fatal.h"
#include "../General/include/v.h"

#define INIT_PORTION 20
#define SHROUD_EXT	0.75f			  // extension factor of shroud height
#define HUB_EXT_RAD	0.1f			  // radius at hub extension end
#define IN_EXT_H	0.05f			  // height factor for inlet ext.
#define IN_EXT_R	0.2f			  // radius factor for inlet ext.
#define BSPLN_DEGREE 3				  // bspline degree
#define SMALL  1.0E-04

#ifdef INTERPOL_BLADE_PROFILE
extern struct profile *InterpolBladeProfile(struct profile *bp_src);
#endif
extern int InitRR_BladeElements(struct radial *rr);
extern int ModifyRR_BladeElements4Covise(struct radial *rr);
extern int CreateRR_MeridianContours(struct radial *rr);
extern int CreateDR_MeridianContours(struct radial *rr);
extern int CreateRR_ConduitAreas(struct radial *rr);
extern float CalcSpecRevs(struct design *desi);
extern int CalcRR_BladeAngles(struct radial *rr);
extern int CreateRR_BladeEdge(struct edge *e);
extern int BladeEdgeMeridianIntersection(struct edge *e, struct be **be,
										 int be_num);
extern int MSurfacesRR_BladeElement(struct be *be, float lepar, float tepar,
									float rle, float rte, float hle, float hte,
									int clock, int camb_flag);
extern int CSurfacesRR_BladeElement(struct be *be, float lepar, float tepar,
									float rle, float rte, float hle, float hte,
									int clock, float camb_flag);
extern int CSurfacesRR_BladeElement2(struct be *be, float lepar, float tepar,
									float rle, float rte, float hle, float hte,
									int clock, float camb_flag);
extern int LSurfRR_BladeElement(struct be *be, float lepar, float rle, float hle,
								struct edge *te, int nob, float rref, int clock);
#ifdef WRITE_WPOY
static int WriteWPOy(struct be **be, int be_num);
#endif

int CreateRR_BladeElements(struct radial *rr)
{
	int i, err = 0;
	static int init = 1;
#ifdef DEBUG_MERIDIANS
	static int ncall = 0;
	FILE *fedge;
	char fname[255];
#endif						   // DEBUG_MERIDIANS

#ifdef DEBUG_MERIDIANS
	sprintf(fname, "rr_edges_%02d.txt", ncall++);
	if ((fedge = fopen(fname, "w")) == NULL)
		fprintf(stderr, "cannot open file %s\n", fname);
#endif						   // DEBUG_MERIDIANS

	/************************************************************/
	// design data
	if(rr->des) rr->des->spec_revs = CalcSpecRevs(rr->des);

	/************************************************************/
	// memory for blade elements and first assignment
	if (init)
		init = InitRR_BladeElements(rr);
	else
		ModifyRR_BladeElements4Covise(rr);

	/************************************************************/
	// meridian contour calculation
	if(!rr->diagonal) {
		CreateRR_MeridianContours(rr);
	}
	else {
		CreateDR_MeridianContours(rr);
	}

	/************************************************************/
	// conduit area calculation
	CreateRR_ConduitAreas(rr);

	/************************************************************/
	// blade edge calculation
	CreateRR_BladeEdge(rr->le);
	CreateRR_BladeEdge(rr->te);

	/************************************************************/
#ifdef DEBUG_MERIDIANS
	if(rr->le->c->p->nump == 0 || rr->te->c->p->nump == 0) {
		fprintf(stderr," ooops, no edge! le->nump: %d  te->nump: %d\n",
				rr->le->c->p->nump, rr->te->c->p->nump);
	}
	DumpCurve(rr->le->c, fedge);
	fprintf(fedge, "\n\n");
	DumpCurve(rr->te->c, fedge);
	fprintf(fedge, "\n\n");
	fprintf(fedge, " %f	 %f	 %f\n",rr->be[0]->ml->p->x[NPOIN_EXT-1],
			rr->be[0]->ml->p->y[NPOIN_EXT-1],rr->be[0]->ml->p->z[NPOIN_EXT-1]);
	fprintf(fedge, " %f	 %f	 %f\n",rr->be[rr->be_num-1]->ml->p->x[NPOIN_EXT-1],
			rr->be[rr->be_num-1]->ml->p->y[NPOIN_EXT-1],rr->be[rr->be_num-1]->ml->p->z[NPOIN_EXT-1]);
	fprintf(fedge, "\n\n");
	fprintf(fedge, " %f	 %f	 %f\n",rr->be[0]->ml->p->x[rr->be[0]->ml->p->nump-NPOIN_EXT],
			rr->be[0]->ml->p->y[rr->be[0]->ml->p->nump-NPOIN_EXT],
			rr->be[0]->ml->p->z[rr->be[0]->ml->p->nump-NPOIN_EXT]);
	fprintf(fedge, " %f	 %f	 %f\n",rr->be[rr->be_num-1]->ml->p->x[rr->be[0]->ml->p->nump-NPOIN_EXT],
			rr->be[rr->be_num-1]->ml->p->y[rr->be[0]->ml->p->nump-NPOIN_EXT],
			rr->be[rr->be_num-1]->ml->p->z[rr->be[0]->ml->p->nump-NPOIN_EXT]);
	fclose(fedge);
#endif						   // DEBUG_MERIDIANS

	/************************************************************/
	// blade edge-meridian contour intersection
	BladeEdgeMeridianIntersection(rr->le, rr->be, rr->be_num);
	BladeEdgeMeridianIntersection(rr->te, rr->be, rr->be_num);

	if((rr->le->bmpar->num == 0 || rr->te->bmpar->num == 0) ||
	   (rr->le->bmpar->num != rr->be_num ||
		rr->te->bmpar->num != rr->be_num)) {
		fprintf(stderr," ooops, no intersection! le->num: %d  te->num: %d\n",
				rr->le->bmpar->num, rr->te->bmpar->num);
		return BLMER_INTERSECT_ERR;
	}
	/************************************************************/
	// blade angles and specific speed
	if(rr->euler) {
		fprintf(stdout," Using Euler's Equation for blade angle calculation\n");
		if( (err = CalcRR_BladeAngles(rr)) ) return err;
	}

	/************************************************************/
	// calculate centre line and blade surfaces
#ifdef GAP
	rr->be_num++;
#endif

	// use camber line function starting at trailing edge
	if(rr->camb2surf == CAMB_FUNC_TEFIX) {
		for (i = 0; i < rr->be_num; i++) {
			if((err =
				CSurfacesRR_BladeElement(rr->be[i], 
										 rr->le->bmpar->list[i], 
										 rr->te->bmpar->list[i], 
										 rr->le->bmint->x[i], 
										 rr->te->bmint->x[i], 
										 rr->le->bmint->z[i], 
										 rr->te->bmint->z[i], 
					                     rr->rot_clockwise,
										 float(rr->camb_flag)))) return err;
		}
	}
	// use camber line function starting at leading edge
	else if(rr->camb2surf == CAMB_FUNC_LEFIX) {
		for (i = 0; i < rr->be_num; i++) {
			if((err =
				CSurfacesRR_BladeElement2(rr->be[i], 
										  rr->le->bmpar->list[i], 
										  rr->te->bmpar->list[i], 
										  rr->le->bmint->x[i], 
										  rr->te->bmint->x[i], 
										  rr->le->bmint->z[i], 
										  rr->te->bmint->z[i], 
										  rr->rot_clockwise,
										  float(rr->camb_flag)))) return err;
		}
	}
	// use camber line function and given blade length
	else if(rr->camb2surf == CAMB_FUNC_LEN) {
		for (i = 0; i < rr->be_num; i++) {
			if((err =
				LSurfRR_BladeElement(rr->be[i], 
									 rr->le->bmpar->list[i], 
									 rr->le->bmint->x[i], 
									 rr->le->bmint->z[i], 
									 rr->te, rr->nob,
									 rr->ref*rr->le->bmint->x[rr->be_num-1],
									 rr->rot_clockwise))) return err;
		}
	}
	// use edges in meridian contour and spline
	else {
		for (i = 0; i < rr->be_num; i++) {
			if((err =
				MSurfacesRR_BladeElement(rr->be[i], 
										 rr->le->bmpar->list[i], 
										 rr->te->bmpar->list[i], 
										 rr->le->bmint->x[i], 
										 rr->te->bmint->x[i], 
										 rr->le->bmint->z[i], 
										 rr->te->bmint->z[i], 
										 rr->rot_clockwise,
										 rr->camb_flag))) return err;
		}
	}
	/************************************************************/

#ifdef GAP
	rr->be_num--;
#endif
#ifdef GNUPLOT
	WriteGNU_RR(rr);
#endif						   // GNUPLOT

	return err;
}

