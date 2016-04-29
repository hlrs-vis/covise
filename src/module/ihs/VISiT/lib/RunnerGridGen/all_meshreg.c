#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "../General/include/flist.h"
#include "../General/include/points.h"
#include "../General/include/curve.h"
#include "../General/include/plane_geo.h"
#include "../General/include/profile.h"
#include "../General/include/bias.h"
#include "../General/include/curvepoly.h"
#include "../BSpline/include/bspline.h"
#include "../General/include/fatal.h"
#include "../General/include/v.h"

#include "include/rr_grid.h"
#include "include/rr_meshmisc.h"

#ifndef ABS
#define ABS(a)    ( (a) >= (0) ? (a) : -(a) )
#endif
#ifndef BSPLN_DEGREE
#define BSPLN_DEGREE 3
#endif

#ifdef DEBUG_REGIONS
#define VPRINT(x) fprintf(stderr,"%10s = [%4f, %4f, %4f]\n",#x,x[0], x[1], x[2])
#define VPRINTF(x,f) fprintf(f,"%10s = [%4f, %4f, %4f]\n",#x,x[0], x[1], x[2])
#endif

extern void DetermineCoefficients(float *x, float *y, float *a);
extern float EvaluateParameter(float x, float *a);

int CreateRR_GridRegions(int nob, struct rr_grid *grid)
{
    int i, j, ix;
    int ref;  // reference plane's index in cge list
    int ssratio = 0, psratio = 0;

    float t, t2, t3, tt, slen, a[3];
    float dphi = 2*M_PI / nob;
    float phi[3], l0, r0, arc[2];
    float p[3];
    float p1[3], p2[3], p3[3];
    float u1[3], u2[3];
    float v1[3], v3[3];
    float v1prev[3], v3prev[3];

    struct Flist *spara;
    struct Point *sline;

#ifdef DOUBLE_SPLINE14
    float q[3];
    struct Point *sspoly;
    struct Point *pspoly;
#endif

#ifdef GAP
    int itip, ishroud;
    float cllen, *clpara;
#endif

#ifdef BL_REF
    float blthick = 0.0; // bound. lay. thickness, blades
#endif
#ifdef MODIFY_LINE14
    float len = 0.0;  // length of 1.1 & 1.3 in s-l-coord. sys.
#endif

    struct Point *poly  = NULL;
    struct Flist *node  = NULL;

    struct cgrid  *cge    = NULL;
    struct region *reg    = NULL;

#ifdef DEBUG_REGIONS
    int DumpCGridElement(struct cgrid *cge, struct ge *ge, float dphi,
			 float phi1);
#ifdef GAP
    int DumpGapRegions(struct cgrid *cge, int n);
#endif
#endif


#ifdef DEBUG_REGIONS
    char fndebug[111];
    FILE *fpdebug;
#endif

#ifdef DEBUG_REGIONS
    fprintf(stderr,"\n CreateRR_GridRegions()\n");
#endif

    grid->out_part[1]   = 1.0 - grid->out_part[0];
    grid->cledis        = grid->cdis;
    grid->lowindis      = grid->lowdis;
    if(grid->ssdis <= grid->psdis) {
	fatal("illegal discretization parameters! grid->ssdis <= grid->psdis");
	exit(-1);
    }
	
    // **************************************************
    // pre-works
    // check and free memory
    if(poly) {
	FreePointStruct(poly);
	poly = NULL;
    }
    if(node) {
	FreeFlistStruct(node);
	node = NULL;
    }

    // index of reference plane
#ifdef GAP
    ref = (int)(grid->ge_num - grid->gp_num)/2;
#else
    ref = (int)grid->ge_num/2;
#endif
    if ((grid->le_dis = grid->psdis/6) < 3) grid->le_dis = 3;
    // circumferential angle for merid. length = 0, 1
    cge = grid->cge[ref];
    r0 = grid->ge[ref]->ml->p->x[grid->iinlet];
    if(grid->iinlet == 0) l0 = 0.0;
    else l0 = grid->ge[ref]->ml->len[grid->iinlet];

    phi[0] = phi[2] = cge->cl->x[0] + ( (l0 - cge->cl->y[0])/(cge->cl->y[0]-cge->cl->y[1]) )
	* (cge->cl->x[0]-cge->cl->x[1]) * grid->phi_scale[0];
    phi[1] = cge->cl->x[cge->cl->nump-1] - grid->phi_scale[1] * dphi;
    // get coefficients for skewed inlet computation
    u1[0] = grid->ge[0]->para;
    u1[1] = grid->ge[ref]->para;
    u1[2] = grid->ge[grid->ge_num-1]->para;
    v1[0] = grid->phi_skew[0];
    v1[1] = grid->phi_scale[0];
    v1[2] = grid->phi_skew[1];
    DetermineCoefficients(u1,v1,a);
#ifdef BL_REF
    arc[0]  = phi[0]*r0;
    u1[0]   = (cge->clarc->list[0] - arc[0]) * (1.0 - grid->bl_scale[0]);
    u1[1]   = (cge->cl->y[0] - l0) * (1.0 - grid->bl_scale[0]);
    u1[2]   = 0.0;
    blthick = V_Len(u1);
    t       = grid->bl_scale[0];
#endif

    spara = AllocFlistStruct(grid->cdis+1);
    sline = AllocPointStruct();

    // **************************************************
    // calc regions for each plane i
    for(i = 0; i < grid->ge_num; i++) {
	cge = grid->cge[i];
	cge->reg = GetRegionsMemory(cge->reg, grid->reg_num);
	spara->num  = 0;
	sline->nump = 0;
	r0 = grid->ge[i]->ml->p->x[grid->iinlet];
	if(grid->iinlet == 0) l0 = 0.0;
	else l0 = grid->ge[i]->ml->len[grid->iinlet];
	if(grid->skew_runin) {
		cge = grid->cge[ref];
		phi[0] = cge->cl->x[0] + ( (l0 - cge->cl->y[0])/(cge->cl->y[0]-cge->cl->y[1]) )
			* (cge->cl->x[0]-cge->cl->x[1]) * 
			EvaluateParameter(grid->ge[i]->para,a);
		cge = grid->cge[i];
	}
	arc[0] = phi[0]*r0;
	arc[1] = phi[1]*grid->ge[i]->ml->p->x[grid->ge[i]->ml->p->nump-1];
#ifdef DEBUG_REGIONS
	sprintf(fndebug,"rr_debugmesh_%02d.txt",i);
	if( (fpdebug = fopen(fndebug,"w+")) == NULL) {
	    fprintf(stderr,"Shit happened opening file '%s'!\n",fndebug);
	    exit(-1);
	}
	fprintf(fpdebug,"grid->iinlet = %d, grid->ioutlet = %d\n",
		grid->iinlet, grid->ioutlet);
	fprintf(fpdebug,"cge->reg[2]->numl = %d\n",cge->reg[2]->numl);
	fprintf(fpdebug,"ref  = %d, cge->cl->x[cge->cl->nump-1] = %f\n",ref, cge->cl->x[cge->cl->nump-1]);
	fprintf(fpdebug,"phi[0] = %f, phi[1] = %f, dphi = %f\n", phi[0], phi[1], dphi);
	fprintf(fpdebug,"arc[0] = %f, arc[1] = %f, r0 = %f, l0 = %f, ml->x[nump-1] = %f\n",
		arc[0], arc[1], r0, l0, grid->ge[i]->ml->p->x[grid->ge[i]->ml->p->nump-1]);
#endif

	// **************************************************
	// inlet region, no. 1
	reg = cge->reg[0];
	// create inlet edge line (1.1)
	reg->para[0] = CalcBladeElementBias(grid->cdis, 0.0, 1.0, grid->cbias_type, grid->cbias);
	if(grid->mesh_ext) {
	    // create line 1.1 as a spline
	    // first point & vector, ss
	    p1[0] =  arc[0];
	    p1[1] =  l0;
	    p1[2] =  0.0;
	    v1[0] =  1.0;
	    v1[1] =  tan(-grid->angle_ext[0]);
	    v1[2] =  0.0;
	    // last point & vector, ps
	    p3[0] =  arc[0] + dphi*r0;
	    p3[1] =  l0;
	    p3[2] =  0.0;
	    v3[0] = -1.0;
	    v3[1] = -tan(grid->angle_ext[1]);
	    v3[2] =  0.0;
	    t = 0.5;
	    LineIntersect(p3,v3, p1,v1, p2);
	    poly = CurvePolygon(p1,p2,p3, t, t);
	    node = BSplineKnot(poly, BSPLN_DEGREE);
	    for(j = 0; j < grid->cdis; j++) {
		BSplinePoint(BSPLN_DEGREE, poly, node, reg->para[0]->list[j], p);
		AddVPoint(reg->line[0], p);
	    }
	    FreePointStruct(poly);
	    FreeFlistStruct(node);
	    CalcPointCoords(reg->line[0], &reg->arc[0], grid->ge[i]->ml);
	}
	else {
	    p[1] = l0;              // merid. length coord.
	    p[2] = r0;  // radius
	    for(j = 0; j < reg->para[0]->num; j++) { 
		p[0] = phi[0] + reg->para[0]->list[j]*dphi;
		AddVPoint(reg->line[0], p);
	    }
	    reg->arc[0] = GetCircleArclen(reg->line[0]);
	}
	// left(suct. side, 1.2) and right(pres. side, 1.3) line
#ifdef BL_REF
	u1[0] = (cge->clarc->list[0] - arc[0]);
	u1[1] = cge->cl->y[0] - l0;
	u1[2] = 0.0;			
	t = 1.0 - (blthick/V_Len(u1));
#else
	// linear interpolation for t between hub and shroud
	t = grid->bl_scale[0] * (1.0 - grid->ge[i]->para) 
	    + grid->ge[i]->para * grid->bl_scale[1];
#endif
#
#ifdef DEBUG_REGIONS
	fprintf(stderr," line 1.2: t = %f, para = %f\n",t,grid->ge[i]->para);
	fprintf(fpdebug," line 1.2: t = %f, para = %f\n",t,grid->ge[i]->para);
#endif
	reg->para[1] = CalcBladeElementBias(grid->ssmdis, 0.0, 1.0, 
					    grid->ssmbias_type, grid->ssmbias);
	u1[0] = (cge->cl->x[0] - phi[0])*t;
	u1[1] = (cge->cl->y[0] - l0)*t;
	u1[2] = 0.0;
	u2[0] = phi[0];
	u2[1] = l0;
	u2[2] = 0.0;
#ifdef DEBUG_REGIONS
	fprintf(fpdebug,"\n line 1.2, reg->para[1]:\n");
	DumpFlist2File(reg->para[1],fpdebug);
	VPRINTF(u1,fpdebug);
	VPRINTF(u2,fpdebug);
	fprintf(fpdebug,"\n");
#endif
	CalcLinearCurve(reg->line[1], reg->para[1], u1, u2);
	GetPointRadius(reg->line[1], grid->ge[i]->ml);
	reg->arc[1] = GetCircleArclen(reg->line[1]);
		
	// 1.3
	reg->para[2] = CopyFlistStruct(reg->para[1]);
	XShiftCurve(reg->line[1], reg->line[2], dphi);
	reg->arc[2] = GetCircleArclen(reg->line[2]);

	// leading edge line (1.4) comes after blade envelope regions

	// **************************************************
	// ps-envelope region (boundary layer around blade), no. 4
	reg = cge->reg[3];
	// extension, right line, leading edge (4.1)
#ifdef DEBUG_REGIONS
	fprintf(stderr," 4.1\n");
#endif
	reg->para[0] = CalcBladeElementBias(grid->psedis, 0.0, 1.0,
					    grid->psebias_type, grid->psebias);

	u1[0] = (cge->cl->x[0]+dphi) - cge->reg[0]->line[2]->x[grid->ssmdis-1];
	u1[1] = cge->cl->y[0] - cge->reg[0]->line[2]->y[grid->ssmdis-1];
	u1[2] = 0.0;
	u2[0] = cge->reg[0]->line[2]->x[grid->ssmdis-1];
	u2[1] = cge->reg[0]->line[2]->y[grid->ssmdis-1];
	u2[2] = 0.0;
	CalcLinearCurve(reg->line[0], reg->para[0], u1, u2);
	GetPointRadius(reg->line[0], grid->ge[i]->ml);
	reg->arc[0] = GetCircleArclen(reg->line[0]);

	// pres. side, blade surface (4.3)
	t2 = grid->psle_part[0] * (1.0 - grid->ge[i]->para) + 
		grid->psle_part[1] * grid->ge[i]->para;
#ifdef DEBUG_REGIONS
	fprintf(stderr," 4.3\n");
#endif
	reg->para[2] = CalcBladeElementBias(grid->le_dis, 0.0, t2, 1, -2.0);
	reg->para[2] = Add2Bias(reg->para[2],grid->psdis-grid->le_dis+1,t2,1.0,
				grid->psbias_type, grid->psbias,1);
	u1[0] = -(cge->cl->y[1] - cge->cl->y[0]);
	u1[1] =   cge->clarc->list[1] - cge->clarc->list[0];
	InterpolBladeSpline(cge->ps->p, reg->para[2], 
			    reg->line[2], u1, dphi);
	CalcPointCoords(reg->line[2], &reg->arc[2], grid->ge[i]->ml);
		
	// pres. side, envelope curve (4.2)
#ifdef DEBUG_REGIONS
	fprintf(stderr,"pres. side, envelope curve (4.2), grid->le_dis = %d\n",grid->le_dis);
	fprintf(fpdebug,"pres. side, envelope curve (4.2), grid->le_dis = %d\n",grid->le_dis);
#endif
	reg->para[1] = CopyFlistStruct(reg->para[2]);
		
#ifdef DEBUG_REGIONS
	fprintf(fpdebug,"ps envelope curve (4.2), parameters:\n");
	DumpFlist2File(reg->para[1],fpdebug);
	fprintf(fpdebug,"ps surface curve (4.3), parameters:\n");
	DumpFlist2File(reg->para[2],fpdebug);
#endif
	CalcEnvelopeCurveSpline2(reg->line[1], reg->arc[0], reg->line[0],
				 reg->arc[2], reg->line[2], cge->cl,
				 reg->para[1], grid->le_dis, dphi, 1.0, 1);
	CalcPointCoords(reg->line[1], &reg->arc[1], grid->ge[i]->ml);

	// trailing edge (4.4)
	reg->para[3] = CopyFlistStruct(reg->para[0]);
	u1[0] = reg->arc[2]->list[grid->psdis-1] - reg->arc[1]->list[grid->psdis-1];
	u1[1] = reg->line[2]->y[grid->psdis-1] - reg->line[1]->y[grid->psdis-1];
	u1[2] = 0.0;
	u2[0] = reg->arc[1]->list[grid->psdis-1];
	u2[1] = reg->line[1]->y[grid->psdis-1];
	u2[2] = 0.0;
	CalcLinearCurve(reg->line[3], reg->para[3], u1, u2);
	CalcPointCoords(reg->line[3], &reg->arc[3], grid->ge[i]->ml);

	// **************************************************
	// ss-envelope, no. 2
	reg = cge->reg[1];
	// extension, left side, copy from right side (2.1)
	reg->para[0] = CopyFlistStruct(cge->reg[3]->para[0]);
	XShiftCurve(cge->reg[3]->line[0],reg->line[0], -dphi); 
	GetPointRadius(reg->line[0], grid->ge[i]->ml);
	reg->arc[0] = GetCircleArclen(reg->line[0]);

	// ss, blade surface (2.2)
	// section for connection point with ps-envelope
	t  = (1.0 - grid->ge[i]->para) * grid->ss_part[0] + grid->ge[i]->para * grid->ss_part[1];
	t2 = grid->ssle_part[0] * (1.0 - grid->ge[i]->para) + 
		grid->ssle_part[1] * grid->ge[i]->para;
	reg->para[1] = CalcBladeElementBias(grid->le_dis,0.0, t2, 1, -2.0);
	reg->para[1] = Add2Bias(reg->para[1], grid->psdis-grid->le_dis+1, t2,t, 
				grid->psbias_type, grid->psbias,1);
	reg->para[1] = Add2Bias(reg->para[1], (grid->ssdis - grid->psdis + 1), t, 1.0, 
				grid->ssbias_type, grid->ssbias, 1);
		
	u1[0] = -(cge->cl->y[1] - cge->cl->y[0]);
	u1[1] =   cge->clarc->list[1] - cge->clarc->list[0];
	InterpolBladeSpline(cge->ss->p, reg->para[1], 
			    reg->line[1], u1, 0.0);		
	CalcPointCoords(reg->line[1], &reg->arc[1], grid->ge[i]->ml);

	// suct. side, envelope curve (2.3)
#ifdef DEBUG_REGIONS
	fprintf(stderr,"suct. side, envelope curve (2.3)\n");
#endif
	reg->para[2] = CalcBladeElementBias(grid->le_dis,0.0, 0.06, 1, -2.0);
	reg->para[2] = Add2Bias(reg->para[2], grid->psdis-grid->le_dis+1, 0.06,t, 
				grid->psbias_type, grid->psbias,1);
	reg->para[2] = Add2Bias(reg->para[2], (grid->ssdis - grid->psdis + 1), t, 1.0, 
				grid->ssbias_type, grid->ssbias, 1);
#ifdef DEBUG_REGIONS
	fprintf(fpdebug,"ss envelope curve (2.3), parameters:\n");
	DumpFlist2File(reg->para[2],fpdebug);
	fprintf(fpdebug,"ps surface curve (2.2), parameters:\n");
	DumpFlist2File(reg->para[1],fpdebug);
#endif
	CalcEnvelopeCurveSpline2(reg->line[2], reg->arc[0], reg->line[0],
				 reg->arc[1], reg->line[1], cge->cl,
				 reg->para[2], grid->le_dis, 0.0, 1.0, -1);
	CalcPointCoords(reg->line[2], &reg->arc[2], grid->ge[i]->ml);

	// trailing edge (2.4)
	reg->para[3] = CopyFlistStruct(reg->para[0]);
	u1[0] = reg->arc[1]->list[grid->ssdis-1] - reg->arc[2]->list[grid->ssdis-1];
	u1[1] = reg->line[1]->y[grid->ssdis-1] - reg->line[2]->y[grid->ssdis-1];
	u1[2] = 0.0;
	u2[0] = reg->arc[2]->list[grid->ssdis-1];
	u2[1] = reg->line[2]->y[grid->ssdis-1];
	u2[2] = 0.0;
	CalcLinearCurve(reg->line[3], reg->para[3], u1, u2);
	CalcPointCoords(reg->line[3], &reg->arc[3], grid->ge[i]->ml);
	// **************************************************
	// leading edge line (1.4) of inlet region no. 1, spline
	reg = cge->reg[0];

	reg->para[3] = CalcBladeElementBias(grid->cdis, 0.0,1.0, grid->clebias_type, grid->clebias);
	t = 0.5; t2 = 0.4;// partition for spline polygon
	// left side, suction side, point, p1 and vector v1
	p1[0] = reg->arc[1]->list[reg->line[1]->nump-1];
	p1[1] = reg->line[1]->y[reg->line[1]->nump-1];
	p1[2] = 0.0;
	u1[0] = reg->arc[1]->list[0] - reg->arc[1]->list[reg->line[1]->nump-1];
	u1[1] = reg->line[1]->y[0] - reg->line[1]->y[reg->line[1]->nump-1];
	u1[2] = 0.0;
	V_Norm(u1);
	u2[0] = cge->reg[1]->arc[2]->list[1] - cge->reg[1]->arc[2]->list[0];
	u2[1] = cge->reg[1]->line[2]->y[1]   - cge->reg[1]->line[2]->y[0];
	u2[2] = 0.0;			
	V_Norm(u2);
	V_Add(u1, u2, v1);
	// right side (ps), point p3 and vector v3
	p3[0] = reg->arc[2]->list[reg->line[2]->nump-1];
	p3[1] = reg->line[2]->y[reg->line[2]->nump-1];
	p3[2] = 0.0;
	u1[0] = reg->arc[2]->list[0] - reg->arc[2]->list[reg->line[2]->nump-1];
	u1[1] = reg->line[2]->y[0] - reg->line[2]->y[reg->line[2]->nump-1];
	u1[2] = 0.0;
	V_Norm(u1);
	u2[0] = cge->reg[3]->arc[1]->list[1] - cge->reg[3]->arc[1]->list[0];
	u2[1] = cge->reg[3]->line[1]->y[1]   - cge->reg[3]->line[1]->y[0];
	u2[2] = 0.0;
	V_Norm(u2);
	V_Add(u1, u2, v3);
	t3 = tan(grid->v14_angle[1]);
	tt = (v3[1]/v3[0]);
#ifdef DEBUG_REGIONS
	VPRINTF(v3,fpdebug);
	fprintf(fpdebug," tt = %f, t3 = %f (tan)\n",
		tt, t3);
#endif
	if(tt < t3) {
	    if(tt < 0.0) {
		v3[0] =  0.0;
		v3[1] = -1.0;
	    }
	    else v3[1] =  v3[0] * t3;
	}
#ifdef DEBUG_REGIONS
	VPRINTF(v3,fpdebug);
#endif
	if(i == 0) {
	    v1prev[0] = v1[0];
	    v1prev[1] = v1[1];
	    v1prev[2] = 0.0;

	    v3prev[0] = v3[0];
	    v3prev[1] = v3[1];
	    v3prev[2] = 0.0;
	}
	else {
	    t3 = 0.75;
	    v1[0] = (1.0 - t) * v1[0] + t3 * v1prev[0];
	    v1[1] = (1.0 - t) * v1[1] + t3 * v1prev[1];
	    v1prev[0] = v1[0];
	    v1prev[1] = v1[1];

	    t3 = 0.5;
	    v3[0] = (1.0 - t) * v3[0] + t3 * v3prev[0];
	    v3[1] = (1.0 - t) * v3[1] + t3 * v3prev[1];
	    v3prev[0] = v3[0];
	    v3prev[1] = v3[1];
	}
	// if ss-vector points down
#ifdef DOUBLE_SPLINE14
	if(v1[1] > 0.0) {
	    q[0] = p1[0] + 0.25*(p3[0] - p1[0]);
	    q[1] = p1[1] + 0.25*(p3[1] - p1[1]);
	    q[2] = 0.0;
	    u1[0] = 1.0;
	    u1[1] = -tan(grid->v14_angle[0]);
	    LineIntersect(p1,v1, q,u1, p2);
	    sspoly = CurvePolygon(p1,p2,q,0.9,0.2);
	    LineIntersect(q,u1, p3,v3, p2);
	    pspoly = CurvePolygon(q,p2,p3,0.2,0.9);
	    poly = AllocPointStruct();
	    for(j = 0; j < sspoly->nump; j++) {
		AddPoint(poly,sspoly->x[j], sspoly->y[j], sspoly->z[j]);
	    }
	    for(j = 1; j < pspoly->nump; j++) {
		AddPoint(poly,pspoly->x[j], pspoly->y[j], pspoly->z[j]);
	    }
	    FreePointStruct(sspoly);
	    FreePointStruct(pspoly);
	}
	else { // v1 points up
	    // check v1 angle against arc-axis
	    if((-v1[1]/v1[0]) < tan(grid->v14_angle[0]))
		v1[1] = -v1[0]*tan(grid->v14_angle[0]);
	    LineIntersect(p3,v3, p1,v1, p2);
	    poly = CurvePolygon(p1,p2,p3, t, t2);
	}
#else // DOUBLE_SPLINE14
	if((-v1[1]/v1[0]) < tan(grid->v14_angle[0]))
	    v1[1] = -v1[0]*tan(grid->v14_angle[0]);
	LineIntersect(p3,v3, p1,v1, p2);
	poly = CurvePolygon(p1,p2,p3, t, t2);
#endif // DOUBLE_SPLINE14
#ifdef DEBUG_REGIONS
	fprintf(fpdebug," Tangent vectors 1.4, %f, %f\n",
		grid->v14_angle[0]*180/M_PI,grid->v14_angle[1]*180/M_PI);
	VPRINTF(v1,fpdebug);
	VPRINTF(v3,fpdebug);
#endif
	node = BSplineKnot(poly, BSPLN_DEGREE);
	// create spline, parameters may be wrong!
	slen = 0.0;
	for(j = 0; j < grid->cdis; j++) {
	    BSplinePoint(BSPLN_DEGREE, poly, node, reg->para[3]->list[j], p);
	    if(j) slen += sqrt(pow(sline->x[j-1]-p[0],2) + pow(sline->y[j-1]-p[1],2));			
	    AddVPoint(sline,p);
	    Add2Flist(spara, slen);
	}
	for(j = 0; j < grid->cdis; j++) {
	    spara->list[j] /= slen;
	}

	// correct points and add them to line[3]
	ix = 0;
	for(j = 0; j < grid->cdis; j++) {
	    ix = GetPointIndex(grid->cdis, spara->list, reg->para[3]->list[j], ix);
	    p[0] = sline->x[ix] + (reg->para[3]->list[j] - spara->list[ix])*(sline->x[ix+1]-sline->x[ix])/
		(spara->list[ix+1] - spara->list[ix]);
	    p[1] = sline->y[ix] + (reg->para[3]->list[j] - spara->list[ix])*(sline->y[ix+1]-sline->y[ix])/
		(spara->list[ix+1] - spara->list[ix]);
	    AddVPoint(reg->line[3], p);
	}
	FreePointStruct(poly);
	FreeFlistStruct(node);
#ifdef MODIFY_LINE14
	t = 0.3;
	if(grid->iinlet == 0) {
	    len = t * (reg->line[1]->y[reg->line[1]->nump-1] - reg->line[1]->y[0]) 
		+ reg->line[1]->y[0];
	}
	for(j = 1; j < reg->line[3]->nump-1; j++) {
	    if(grid->iinlet != 0) {
		len = t * (reg->line[1]->y[reg->line[1]->nump-1] - reg->line[0]->y[j]) 
		    + reg->line[0]->y[j];
	    }
	    if(reg->line[3]->y[j] < len) {
#ifdef DEBUG_REGIONS
		fprintf(fpdebug,"j: %2d, reg->line[3]->y[j] < len, %f < %f\n", 
			j, reg->line[3]->y[j], len);
		fprintf(stderr,"j: %2d, reg->line[3]->y[j] < len, %f < %f\n", 
			j, reg->line[3]->y[j], len);
#endif
		reg->line[3]->y[j] = len;
	    }
	}
#endif
	CalcPointCoords(reg->line[3], &reg->arc[3], grid->ge[i]->ml);

	// **************************************************
#ifdef DEBUG_REGIONS
	fprintf(stderr,"upper core region, no. 3\n");
	fprintf(fpdebug,"upper core region, no. 3\n");
#endif
	// upper core region, no. 3
	reg = cge->reg[2];
	// leading edge line (3.1 = 1.4)
	FreePointStruct(reg->line[0]);
	reg->line[0] = CopyPointStruct(cge->reg[0]->line[3]);
	reg->arc[0]  = CopyFlistStruct(cge->reg[0]->arc[3]);
	reg->para[0] = CopyFlistStruct(cge->reg[0]->para[3]);
	// ps-envelope (3.3 = 4.2)
	FreePointStruct(reg->line[2]);
	reg->line[2] = CopyPointStruct(cge->reg[3]->line[1]);
	reg->arc[2]  = CopyFlistStruct(cge->reg[3]->arc[1]);
	reg->para[2] = CopyFlistStruct(cge->reg[3]->para[1]);
	// ss-upper part, 0..t (3.2 = 2.3)
	FreePointStruct(reg->line[1]);
	reg->para[1] = CopyFlistStruct(reg->para[2]);
	reg->line[1] = nCopyPointStruct(cge->reg[1]->line[2],reg->para[1]->num);
	reg->arc[1]  = nCopyFlistStruct(cge->reg[1]->arc[2],reg->para[1]->num);
#ifdef DEBUG_REGIONS
	fprintf(fpdebug,"upper core\n");
	fprintf(fpdebug,"reg->arc[1]\n");
	DumpFlist2File(reg->arc[1], fpdebug);
#endif //DEBUG_REGIONS
	// trailing edge to ss-envelope (3.4)
	reg->para[3] = CalcBladeElementBias(grid->cdis, 0.0, 1.0, grid->midbias_type, grid->midbias);
	u1[0] = reg->arc[2]->list[reg->arc[2]->num-1] - reg->arc[1]->list[reg->arc[1]->num-1];
	u1[1] = reg->line[2]->y[reg->line[2]->nump-1] - reg->line[1]->y[reg->line[1]->nump-1];
	u1[2] = 0.0;
	u2[0] = reg->arc[1]->list[reg->arc[1]->num-1];
	u2[1] = reg->line[1]->y[reg->line[1]->nump-1];
	u2[2] = 0.0;
	CalcLinearCurve(reg->line[3], reg->para[3], u1, u2);
	CalcPointCoords(reg->line[3], &reg->arc[3], grid->ge[i]->ml);
		
	// **************************************************
#ifdef DEBUG_REGIONS
	fprintf(stderr,"lower left region, no. 5\n");
	fprintf(fpdebug,"lower left region, no. 5\n");
#endif
	// lower left region, no. 5
	reg = cge->reg[4];
	if(i == 0) {
	    ssratio = CalcTERatio(grid->cge[0]->reg[1], grid->cge[0]->clarc, 
				  grid->cge[0]->cl, 1, 2, 0.0);
	}

	// t has to be calculated for each plane
	t = CalcTEParameter(cge->reg[1], cge->clarc, cge->cl, 1, 2, 0.0);

#ifdef DEBUG_REGIONS
	fprintf(stderr,"line at trailing edge of ss (5.1), t = %f, ssratio = %d\n", t, ssratio);
	fprintf(fpdebug,"line at trailing edge of ss (5.1)\n");
#endif
	// now the region, no. 5, line at trailing edge of ss (5.1)
	if(t != 0.0) {
	    reg->para[0] = CalcBladeElementBias(ssratio, 0.0, t, 0, 1.0);
#ifdef DEBUG_REGIONS
	    fprintf(stderr, "0...t\n");
	    fprintf(fpdebug, "0...t\n");
	    DumpFlist2File(reg->para[0], fpdebug);
#endif
			
	    u1[0] = cge->reg[1]->arc[2]->list[grid->ssdis-1] - cge->clarc->list[cge->clarc->num-1];
	    u1[1] = cge->reg[1]->line[2]->y[grid->ssdis-1] - cge->cl->y[cge->cl->nump-1];
	    u1[2] = 0.0;
	    u2[0] = cge->clarc->list[cge->clarc->num-1];
	    u2[1] = cge->cl->y[cge->cl->nump-1];
	    u2[2] = 0.0;
	    CalcLinearCurve(reg->line[0], reg->para[0], u1, u2);
	    reg->line[0]->nump--; // eliminate last point, double with line (2.4 first point)
	    CalcPointCoords(reg->line[0], &reg->arc[0], grid->ge[i]->ml);
#ifdef DEBUG_REGIONS
	    fprintf(stderr, "arc 0...t\n");
	    fprintf(fpdebug, "arc 0...t\n");
	    DumpFlist2File(reg->arc[0], fpdebug);
#endif
		
	    reg->para[0] = Add2Bias(reg->para[0], cge->reg[1]->para[3]->num, t, 1.0, 
				    grid->psebias_type, -grid->psebias, 1);
	} // t != 0, te exists
	else {
	    reg->para[0] = AllocFlistStruct(cge->reg[1]->para[3]->num+1);
	    for(j = cge->reg[1]->para[3]->num-1; j >= 0; j--) 
		Add2Flist(reg->para[0], 1.0-cge->reg[1]->para[3]->list[j]);
	    reg->arc[0]  = AllocFlistStruct(cge->reg[1]->para[3]->num+1);
	}
	for(j = (cge->reg[1]->line[3]->nump-1); j >= 0; j--) {
	    AddPoint(reg->line[0], cge->reg[1]->line[3]->x[j], cge->reg[1]->line[3]->y[j],
		     cge->reg[1]->line[3]->z[j]);
	    Add2Flist(reg->arc[0], cge->reg[1]->arc[3]->list[j]);
#ifdef DEBUG_REGIONS
	    fprintf(fpdebug," j = %d, cge->reg[1]->arc[3]->list[j] = %f\n",j, cge->reg[1]->arc[3]->list[j]);
#endif
	}
#ifdef DEBUG_REGIONS
	fprintf(fpdebug," cge->reg[1]->line[3]->nump-1 = %d\n", cge->reg[1]->line[3]->nump-1);
	fprintf(fpdebug,"cge->reg[1]->para[3]\n");
	DumpFlist2File(cge->reg[1]->para[3], fpdebug);
	fprintf(fpdebug,"reg->para[0]\n");
	DumpFlist2File(reg->para[0], fpdebug);
	fprintf(fpdebug,"cge->arc[1]->list[3]\n");
	DumpFlist2File(cge->reg[1]->arc[3], fpdebug);
	fprintf(fpdebug,"reg->arc[0]\n");
	DumpFlist2File(reg->arc[0], fpdebug);
#endif
		
#ifdef DEBUG_REGIONS
	fprintf(fpdebug,"outer spline, te -> ml end (5.2)\n");
	fprintf(stderr,"outer spline, te -> ml end (5.2)\n");
#endif
	// outer spline, te -> ml end (5.2)
	reg->para[1] = CalcBladeElementBias(grid->lowdis, 0.0, 1.0, 
					    grid->lowbias_type, grid->lowbias);
	// first point + vector, trailing edge, center line
	v1[0] = cge->clarc->list[cge->clarc->num-1] 
	    - cge->clarc->list[cge->clarc->num-2];
	v1[1] = cge->cl->y[cge->cl->nump-1]
	    - cge->cl->y[cge->cl->nump-2];
	v1[2] = 0.0;
	p1[0] = cge->clarc->list[cge->clarc->num-1];
	p1[1] = cge->cl->y[cge->cl->nump-1];
	p1[2] = 0.0;
	// last point + vector, outlet
		
	v3[0] =  phi[1]*(grid->ge[i]->ml->p->x[grid->ge[i]->ml->p->nump-2] 
			 - grid->ge[i]->ml->p->x[grid->ge[i]->ml->p->nump-1]);
	v3[1] =  grid->ge[i]->ml->len[grid->ge[i]->ml->p->nump-2] 
	    - grid->ge[i]->ml->len[grid->ge[i]->ml->p->nump-1];
	v3[2] =  0.0;
	p3[0] =  arc[1];
	p3[1] =  grid->ge[i]->ml->len[grid->ge[i]->ml->p->nump-1];
	p3[2] =  0.0;
	LineIntersect(p3,v3, p1,v1, p2);
	if(p2[1] > grid->ge[i]->ml->len[grid->ge[i]->ml->p->nump-1]) {
	    fprintf(stderr," 5.2: CreateRR_GridRegions: %d: ill intersection point, point shifted!\n", i+1);
	    p2[1] = grid->ge[i]->ml->len[grid->ge[i]->ml->p->nump-1] * 0.95;
	}
	t  = 0.6;
	t2 = 0.6;
	poly = CurvePolygon(p1,p2,p3, t, t2);
	node = BSplineKnot(poly, BSPLN_DEGREE);
	for(j = 0; j < reg->para[1]->num; j++) {
	    BSplinePoint(BSPLN_DEGREE, poly, node, reg->para[1]->list[j], p);
	    AddVPoint(reg->line[1], p);
	}
	FreePointStruct(poly);
	FreeFlistStruct(node);
	CalcPointCoords(reg->line[1], &reg->arc[1], grid->ge[i]->ml);
		
	// inner spline, te-envelope -> ml end, 5.3
	reg = cge->reg[4];
	reg->para[2] = CalcBladeElementBias(grid->lowindis, 0.0, 1.0,
					    grid->lowin_type, grid->lowinbias);
		
	// first point + vector
#ifdef TE_VECTOR
	v1[0] = cge->reg[1]->arc[2]->list[cge->reg[1]->arc[2]->num-1] 
	    - cge->reg[1]->arc[2]->list[cge->reg[1]->arc[2]->num-2]; 
	v1[1] = cge->reg[1]->line[2]->y[cge->reg[1]->line[2]->nump-1] 
	    - cge->reg[1]->line[2]->y[cge->reg[1]->line[2]->nump-2];

#else
	v1[0] = cge->clarc->list[cge->clarc->num-1] 
	    - cge->clarc->list[cge->clarc->num-2];
	v1[1] = cge->cl->y[cge->cl->nump-1]
	    - cge->cl->y[cge->cl->nump-2];
#endif
	v1[2] = 0.0; 		
	p1[0] = cge->reg[1]->arc[2]->list[cge->reg[1]->arc[2]->num-1];
	p1[1] = cge->reg[1]->line[2]->y[cge->reg[1]->line[2]->nump-1];
	p1[2] = 0.0;
	// last point + vector, outlet
	t = grid->out_part[0];
	v3[0] =  (phi[1] + t*dphi)*(grid->ge[i]->ml->p->x[grid->ge[i]->ml->p->nump-2] 
				    - grid->ge[i]->ml->p->x[grid->ge[i]->ml->p->nump-1]);
	v3[1] =  grid->ge[i]->ml->len[grid->ge[i]->ml->p->nump-2] 
	    - grid->ge[i]->ml->len[grid->ge[i]->ml->p->nump-1];
	v3[2] =  0.0;
	p3[0] =  (phi[1] + t*dphi)*grid->ge[i]->ml->p->x[grid->ge[i]->ml->p->nump-1];
	p3[1] =  grid->ge[i]->ml->len[grid->ge[i]->ml->p->nump-1];
	p3[2] =  0.0;
	LineIntersect(p3,v3, p1,v1, p2);
	if(p2[1] > grid->ge[i]->ml->len[grid->ge[i]->ml->p->nump-1]) {
	    fprintf(stderr," 5.3: CreateRR_GridRegions: %d: ill intersection point, point shifted!\n", i+1);
	    p2[1] = grid->ge[i]->ml->len[grid->ge[i]->ml->p->nump-1] * 0.95;
	}
	t = 0.6; t2 = 0.6;
	poly = CurvePolygon(p1,p2,p3, t, t2);
	node = BSplineKnot(poly, BSPLN_DEGREE);
	for(j = 0; j < reg->para[2]->num; j++) {
	    BSplinePoint(BSPLN_DEGREE, poly, node, reg->para[2]->list[j], p);
	    AddVPoint(reg->line[2], p);
	}
	FreePointStruct(poly);
	FreeFlistStruct(node);
	CalcPointCoords(reg->line[2], &reg->arc[2], grid->ge[i]->ml);

	// outlet line, 5.4
	reg->para[3] = CalcBladeElementBias(reg->para[0]->num, 0.0, 1, grid->ssxbias_type, grid->ssxbias);
	u1[0] = reg->arc[2]->list[reg->arc[2]->num-1] - reg->arc[1]->list[reg->arc[1]->num-1];
	u1[1] = 0.0;
	u1[2] = 0.0;
	u2[0] = arc[1];
	u2[1] = grid->ge[i]->ml->len[grid->ge[i]->ml->p->nump-1];
	u2[2] = 0.0;
	CalcLinearCurve(reg->line[3], reg->para[3], u1, u2);
	CalcPointCoords(reg->line[3], &reg->arc[3], grid->ge[i]->ml);

	// **************************************************
#ifdef DEBUG_REGIONS
	fprintf(fpdebug,"ps extension, no. 7\n");
#endif
	// ps extension, no. 7
	reg = cge->reg[6];
	// trailing edge of ps (7.1)
	if(i == 0) {
	    psratio = CalcTERatio(cge->reg[3], cge->clarc, cge->cl,2,1, dphi);
	}

	// t has to be calculated for each plane
	t = 1.0 - CalcTEParameter(cge->reg[3], cge->clarc, cge->cl, 2, 1, dphi);

	reg->para[0] = CalcBladeElementBias(grid->psedis, 0.0, t, 
					    grid->psebias_type, grid->psebias);
	for(j = 0; j < cge->reg[3]->line[3]->nump; j++) {
	    AddPoint(reg->line[0], cge->reg[3]->arc[3]->list[j], 
		     cge->reg[3]->line[3]->y[j], cge->reg[3]->line[3]->z[j]);
	}

#ifdef DEBUG_REGIONS
	fprintf(stderr,"Reg. no. 7\n");
	fprintf(stderr,"psratio = %d, t = %f\n", psratio, t);
	fprintf(fpdebug,"Reg. no. 7\n");
	fprintf(fpdebug,"psratio = %d, t = %f\n", psratio, t);
	DumpFlist2File(reg->para[0], fpdebug);
#endif
	if(t < 1.0) {
	    reg->para[0] = Add2Bias(reg->para[0], psratio, t, 1.0, 
				    0, 1.0, 1);
#ifdef DEBUG_REGIONS
	    DumpFlist2File(reg->para[0], fpdebug);
#endif
	    u1[0] = cge->clarc->list[cge->clarc->num-1]
		+ dphi*cge->cl->z[cge->cl->nump-1]
		- cge->reg[3]->arc[1]->list[cge->reg[3]->arc[1]->num-1];
	    u1[1] = cge->cl->y[cge->cl->nump-1]
		- cge->reg[3]->line[1]->y[cge->reg[3]->line[1]->nump-1];
	    u1[2] = 0.0;
	    u2[0] = cge->reg[3]->arc[1]->list[cge->reg[3]->arc[1]->num-1];
	    u2[1] = cge->reg[3]->line[1]->y[cge->reg[3]->line[1]->nump-1];
	    u2[2] = 0.0;
	    for(j = reg->line[0]->nump; j < psratio + cge->reg[3]->line[3]->nump-1; j++) {
		p[0] = u1[0]*reg->para[0]->list[j] + u2[0];
		p[1] = u1[1]*reg->para[0]->list[j] + u2[1];
		AddVPoint(reg->line[0], p);
	    }
	} // t < 1.0
	CalcPointCoords(reg->line[0], &reg->arc[0], grid->ge[i]->ml);
	// ps-envelope extension, 7.2
	t = (1.0 - grid->ge[i]->para) * grid->ps_part[0] 
	    + grid->ge[i]->para * grid->ps_part[1];
	reg->para[1] = CalcBladeElementBias( (grid->ssdis - grid->psdis + 1), 0.0,t,
					     grid->ssbias_type, grid->ssbias);
#ifdef DEBUG_REGIONS
	fprintf(fpdebug,"\t** 7.2 **\n");
	DumpFlist2File(reg->para[1], fpdebug);
#endif
	reg->para[1] = Add2Bias(reg->para[1], grid->lowindis, t, 1.0, 
				grid->lowin_type, grid->lowinbias, 1);	
#ifdef DEBUG_REGIONS
	DumpFlist2File(reg->para[1], fpdebug);
#endif

#ifdef PSENVEXT_SPLINE
	// first point + vector, trailing edge
#ifdef TE_VECTOR
	v1[0] = cge->reg[3]->arc[1]->list[cge->reg[3]->arc[1]->num-1]
	    - cge->reg[3]->arc[1]->list[cge->reg[3]->arc[1]->num-2];
	v1[1] = cge->reg[3]->line[1]->y[cge->reg[3]->line[1]->nump-1]
	    - cge->reg[3]->line[1]->y[cge->reg[3]->line[1]->nump-2];
#elif CL_VECTOR
	v1[0] = cge->clarc->list[cge->clarc->num-1]
	    - cge->clarc->list[cge->clarc->num-2];
	v1[1] = cge->cl->y[cge->cl->nump-1]
	    - cge->cl->y[cge->cl->nump-2];
#else
	v1[0] = (1.0 - grid->ge[i]->para) * (cge->reg[3]->arc[1]->list[cge->reg[3]->arc[1]->num-1]
					     - cge->reg[3]->arc[1]->list[cge->reg[3]->arc[1]->num-2])
	    + grid->ge[i]->para * (cge->clarc->list[cge->clarc->num-1]
				   - cge->clarc->list[cge->clarc->num-2]);
	v1[1] = (1.0 - grid->ge[i]->para) * (cge->reg[3]->line[1]->y[cge->reg[3]->line[1]->nump-1]
					     - cge->reg[3]->line[1]->y[cge->reg[3]->line[1]->nump-2])
	    + grid->ge[i]->para * (cge->cl->y[cge->cl->nump-1]
				   - cge->cl->y[cge->cl->nump-2]);
#ifdef DEBUG_REGIONS
	fprintf(stderr,"Using variable vector\n");
#endif
#endif // end TE/CL_VECTOR
	v1[2] = 0.0;
	p1[0] = cge->reg[3]->arc[1]->list[cge->reg[3]->arc[1]->num-1];
	p1[1] = cge->reg[3]->line[1]->y[cge->reg[3]->line[1]->nump-1];
	p1[2] = 0.0;
	// last point + vector
	t = grid->out_part[1];
	v3[0] = (phi[1] + t*dphi)*(grid->ge[i]->ml->p->x[grid->ge[i]->ml->p->nump-2] 
				   - grid->ge[i]->ml->p->x[grid->ge[i]->ml->p->nump-1]);
	v3[1] =  grid->ge[i]->ml->len[grid->ge[i]->ml->p->nump-2] 
	    - grid->ge[i]->ml->len[grid->ge[i]->ml->p->nump-1];
	v3[2] = 0.0;
	p3[0] = (phi[1] + t*dphi)*grid->ge[i]->ml->p->x[grid->ge[i]->ml->p->nump-1]; 
	p3[1] = grid->ge[i]->ml->len[grid->ge[i]->ml->p->nump-1];
	p3[2] = 0.0; 
	LineIntersect(p3,v3, p1,v1, p2);
	t  = 0.3;
	t2 = 0.6;
	poly = CurvePolygon(p1,p2,p3, t, t2);
	node = BSplineKnot(poly, BSPLN_DEGREE);
	for(j = 0; j < reg->para[1]->num; j++) {
	    BSplinePoint(BSPLN_DEGREE, poly, node, reg->para[1]->list[j], p);
	    AddVPoint(reg->line[1], p);
	}
	FreePointStruct(poly);
	FreeFlistStruct(node);
	CalcPointCoords(reg->line[1], &reg->arc[1], grid->ge[i]->ml);
#endif // end PSENVEXT_SPLINE

	// ps center line extension, 7.3
	reg->para[2] = CopyFlistStruct(reg->para[1]);
	InterpolBlade(cge->reg[4]->para[1]->list, cge->reg[4]->line[1], 
		      reg->para[2], reg->line[2], dphi);
	CalcPointCoords(reg->line[2], &reg->arc[2], grid->ge[i]->ml);

#ifndef PSENVEXT_SPLINE
	// ps envelope extension, 7.2
	t = grid->out_part[1];
	u1[0] = (phi[1] + t*dphi)*grid->ge[i]->ml->p->x[grid->ge[i]->ml->p->nump-1]
	    - reg->arc[2]->list[reg->arc[2]->num-1];
	u1[1] = grid->ge[i]->ml->len[grid->ge[i]->ml->p->nump-1]
	    - reg->line[2]->y[reg->line[2]->nump-1];
	u1[2] = 0.0;
	u2[0] = reg->arc[0]->list[0] - reg->arc[0]->list[reg->arc[0]->num-1];
	u2[1] = reg->line[0]->y[0]   - reg->line[0]->y[reg->line[0]->nump-1];
	u2[2] = 0.0;

	CalcEnvelopeCurve(reg->line[1], reg->arc[0], reg->line[0],
			  reg->arc[2], reg->line[2], reg->line[2],
			  reg->para[1], dphi, (V_Len(u1)/V_Len(u2)), 1);
	CalcPointCoords(reg->line[1], &reg->arc[1], grid->ge[i]->ml);
#endif
	
	// outlet line, 7.4
	reg->para[3] = CalcBladeElementBias(reg->para[0]->num, 0.0, 1.0, 
					    grid->psxbias_type, grid->psxbias);
	u1[0] = reg->arc[2]->list[reg->arc[2]->num-1] - reg->arc[1]->list[reg->arc[1]->num-1];
	u1[1] = 0.0;
	u1[2] = 0.0;
	u2[0] = reg->arc[1]->list[reg->arc[1]->num-1];
	u2[1] = grid->ge[i]->ml->len[grid->ge[i]->ml->p->nump-1];
	u2[2] = 0.0;
	CalcLinearCurve(reg->line[3], reg->para[3], u1, u2);
	CalcPointCoords(reg->line[3], &reg->arc[3], grid->ge[i]->ml);

	// **************************************************
	// lower core region, no. 6
	reg = cge->reg[5];
	// inlet line, 6.1 = 3.4
	FreePointStruct(reg->line[0]);
	reg->line[0] = CopyPointStruct(cge->reg[2]->line[3]);
	reg->arc[0]  = CopyFlistStruct(cge->reg[2]->arc[3]);
	reg->para[0] = CopyFlistStruct(cge->reg[2]->para[3]);
	// right, ps, line, 6.3 = 7.2
	FreePointStruct(reg->line[2]);
	reg->line[2] = CopyPointStruct(cge->reg[6]->line[1]);
	reg->arc[2]  = CopyFlistStruct(cge->reg[6]->arc[1]);
	reg->para[2] = CopyFlistStruct(cge->reg[6]->para[1]);

	// left, ss, line, 6.2 = 5.3 + 2.3
	reg->para[1] = CopyFlistStruct(cge->reg[6]->para[1]);
	for(j = grid->psdis-1; j < grid->ssdis; j++) {
	    AddPoint(reg->line[1], cge->reg[1]->arc[2]->list[j], cge->reg[1]->line[2]->y[j], 0.0);
	}
	for(j = 1; j < cge->reg[4]->line[2]->nump; j++) {
	    AddPoint(reg->line[1], cge->reg[4]->arc[2]->list[j], cge->reg[4]->line[2]->y[j], 0.0);
	}
	CalcPointCoords(reg->line[1], &reg->arc[1], grid->ge[i]->ml);

	// outlet line, 6.4
	reg->para[3] = CalcBladeElementBias(grid->cdis, 0.0, 1.0, grid->cxbias_type, grid->cxbias);
	u1[0] = reg->arc[2]->list[reg->arc[2]->num-1] - reg->arc[1]->list[reg->arc[1]->num-1];
	u1[1] = reg->line[2]->y[reg->line[2]->nump-1] - reg->line[1]->y[reg->line[1]->nump-1];
	u1[2] = 0.0;
	u2[0] = reg->arc[1]->list[reg->arc[1]->num-1];
	u2[1] = reg->line[1]->y[reg->line[1]->nump-1];
	u2[2] = 0.0;
	CalcLinearCurve(reg->line[3], reg->para[3], u1, u2);
	CalcPointCoords(reg->line[3], &reg->arc[3], grid->ge[i]->ml);

	// **************************************************
	// extension of inlet region, no. 8
	if(grid->iinlet != 0) {
	    reg = cge->reg[7];
	    // copy last line, 8.4 = 1.1
	    FreePointStruct(reg->line[3]);
	    reg->line[3] = CopyPointStruct(cge->reg[0]->line[0]);
	    reg->arc[3]  = CopyFlistStruct(cge->reg[0]->arc[0]);
	    reg->para[3] = CopyFlistStruct(cge->reg[0]->para[0]);
	    // left (ss) line, 8.2
#ifdef DEBUG_REGIONS
	    fprintf(stderr," 8.2\n");
#endif
	    reg->para[1] = CalcBladeElementBias(grid->extdis, 0.0, 1.0, 
						grid->extbias_type, grid->extbias);
#ifdef DEBUG_REGIONS
	    fprintf(fpdebug,"\n para, 8.2:\n");
	    DumpFlist2File(reg->para[1],fpdebug);
#endif
	    u2[0] = grid->phi0_ext * phi[2];
	    u2[1] = 0.0;
	    u1[0] = reg->line[3]->x[0] - u2[0];
	    u1[1] = reg->line[3]->y[0];
	    CalcLinearCurve(reg->line[1], reg->para[1], u1, u2);
#ifdef DEBUG_REGIONS
	    VPRINTF(u1,fpdebug);
	    VPRINTF(u2,fpdebug);
	    fprintf(fpdebug," phi[2] = %f\n",phi[2]);
	    for(j = 0; j < reg->line[1]->nump; j++) {
		fprintf(fpdebug,"8.2: j = %3d: arc = %f\n", j, reg->line[1]->x[j]);
	    }
#endif
	    GetPointRadius(reg->line[1], grid->ge[i]->ml);
	    reg->arc[1] = GetCircleArclen(reg->line[1]);
#ifdef DEBUG_REGIONS
	    for(j = 0; j < reg->line[1]->nump; j++) {
		fprintf(fpdebug,"8.2: phi = %f, arc = %f\n", 
			reg->line[1]->x[j], reg->arc[1]->list[j]);
	    }
#endif
	    // right line, 8.3
	    reg->para[2] = CopyFlistStruct(reg->para[1]);
#ifdef DEBUG_REGIONS
	    fprintf(fpdebug,"8.3: i: %d, dphi = %f\n", i, dphi);
#endif
	    XShiftCurve(reg->line[1], reg->line[2], dphi);
	    reg->arc[2] = GetCircleArclen(reg->line[2]);
#ifdef DEBUG_REGIONS
	    for(j = 0; j < reg->line[1]->nump; j++) {
		fprintf(fpdebug," phi(8.3 - 8.2): %f\n", 
			180/M_PI*(reg->line[2]->x[j] - reg->line[1]->x[j]));
	    }
#endif
	    // inlet line, 8.1
	    reg->para[0] = CalcBladeElementBias(grid->cdis, 0.0, 1.0, 0, 1.0);
	    u1[0] = dphi * grid->ge[i]->ml->p->x[0];
	    u1[1] = 0.0;
	    u2[0] = reg->arc[1]->list[0];
	    u2[1] = 0.0;
	    CalcLinearCurve(reg->line[0], reg->para[0], u1, u2);
	    CalcPointCoords(reg->line[0], &reg->arc[0], grid->ge[i]->ml);
	}// iinlet != 0

#ifdef DEBUG_REGIONS
	fclose(fpdebug);
#endif

    } // end i
    FreeFlistStruct(spara);
    FreePointStruct(sline);

    // **************************************************
#ifdef GAP
    ishroud = grid->ge_num - 1;
    itip    = grid->ge_num - (grid->gp_num);
    if( (clpara = (float*)calloc(grid->cge[0]->cl->nump, sizeof(float))) == NULL) {
	fatal("memory for (float)!");
	exit(-1);
    }
    for(i = itip; i <= ishroud; i++) {
#ifdef DEBUG_REGIONS
	sprintf(fndebug,"rr_debugapmesh_%02d.txt",i);
	if( (fpdebug = fopen(fndebug,"w+")) == NULL) {
	    fprintf(stderr,"Shit happened opening file '%s'!\n",fndebug);
	    exit(-1);
	}
	fprintf(fpdebug," i: %d, itip = %d, ishroud = %d\n",i, itip, ishroud);
#endif
	cge = grid->cge[i];
	// **************************************************
	// suction side tip region, no. 9
#ifdef DEBUG_REGIONS
	fprintf(fpdebug,"region no. 9\n");
	fprintf(stderr,"region no. 9\n");
#endif
	reg = cge->reg[grid->reg_num];
	// center line, 9.2
#ifdef DEBUG_REGIONS
	fprintf(fpdebug,"9.2\n");
	fprintf(stderr,"9.2\n");
#endif
	reg->line[1] = AllocPointStruct();
	cllen = 0;
	for(j = 1; j < cge->cl->nump; j++) {
	    cllen += sqrt(pow((cge->cl->y[j] - cge->cl->y[j-1]), 2)
			  + pow((cge->clarc->list[j] - cge->clarc->list[j-1]), 2));
	    clpara[j] = cllen;
	}
	for(j = 1; j < cge->cl->nump; j++) {
	    clpara[j] /= cllen;
#ifdef DEBUG_REGIONS
	    fprintf(fpdebug," clpara[%02d] = %f\n", j, clpara[j]);
#endif
	}
	reg->para[1] = CopyFlistStruct(cge->reg[1]->para[1]);
	InterpolBlade(clpara, cge->cl, 
		      reg->para[1], reg->line[1], 0.0);
	CalcPointCoords(reg->line[1], &reg->arc[1], grid->ge[i]->ml);

	// blade surface (ss), 9.3
#ifdef DEBUG_REGIONS
	fprintf(fpdebug,"9.3\n");
	fprintf(stderr,"9.3\n");
#endif
	FreePointStruct(reg->line[2]);
	reg->line[2] = CopyPointStruct(cge->reg[1]->line[1]);
	reg->para[2] = CopyFlistStruct(cge->reg[1]->para[1]);
	reg->arc[2]  = CopyFlistStruct(cge->reg[1]->arc[1]);

	// trailing edge, 9.4
#ifdef DEBUG_REGIONS
	fprintf(fpdebug,"9.4\n");
	fprintf(stderr,"9.4\n");
#endif
	FreePointStruct(reg->line[3]);
	reg->line[3] = nCopyPointStruct(cge->reg[4]->line[0], ssratio);
	reg->para[3] = CalcBladeElementBias(ssratio, 0.0, 1.0, 0, 1.0);
	reg->arc[3]  = nCopyFlistStruct(cge->reg[4]->arc[0], ssratio);
#ifdef DEBUG_REGIONS
	DumpFlist2File(reg->para[3], fpdebug);
#endif
	// **************************************************
	// pressure side tip region, no. 10
	reg = cge->reg[grid->reg_num+1];
	// center line, 10.3
#ifdef DEBUG_REGIONS
	fprintf(fpdebug,"10.3\n");
	fprintf(stderr,"10.3\n");
#endif
	reg->line[2] = AllocPointStruct();
	reg->para[2] = CopyFlistStruct(cge->reg[3]->para[2]);
	InterpolBlade(clpara, cge->cl, 
		      reg->para[2], reg->line[2], dphi);
	CalcPointCoords(reg->line[2], &reg->arc[2], grid->ge[i]->ml);

	// blade surface (ps), 10.2
	FreePointStruct(reg->line[1]);
	reg->line[1] = CopyPointStruct(cge->reg[3]->line[2]);
	reg->para[1] = CopyFlistStruct(cge->reg[3]->para[2]);
	reg->arc[1]  = CopyFlistStruct(cge->reg[3]->arc[2]);

	// trailing edge, 10.4
#ifdef DEBUG_REGIONS
	fprintf(fpdebug,"10.4\n");
	fprintf(stderr,"10.4\n");
#endif
	reg->line[3] = AllocPointStruct();
	reg->para[3] = CalcBladeElementBias(psratio, 0.0, 1.0, 0, 1.0);
	reg->arc[3]  = AllocFlistStruct(psratio+1);
	for(j = cge->reg[6]->line[0]->nump - psratio; 
	    j < cge->reg[6]->line[0]->nump; j++) {
	    AddPoint(reg->line[3], cge->reg[6]->line[0]->x[j],
		     cge->reg[6]->line[0]->y[j], cge->reg[6]->line[0]->z[j]);
	    Add2Flist(reg->arc[3], cge->reg[6]->arc[0]->list[j]);
	}
		

	// **************************************************

		
#ifdef DEBUG_REGIONS
	fclose(fpdebug);
#endif
    } // end i

    free(clpara);
#endif // GAP		
    // **************************************************

#ifdef DEBUG_REGIONS
    for(i = 0; i < grid->ge_num; i++) {
	DumpCGridElement(grid->cge[i], grid->ge[i], dphi, phi[1]);
    } // end i, grid->ge_num, inlet region
#ifdef GAP
    for(i = itip; i <= ishroud; i++) {
	DumpGapRegions(grid->cge[i], i);
    } // end i, grid->ge_num, inlet region
#endif
#endif // DEBUG_REGIONS


    return 0;
}















