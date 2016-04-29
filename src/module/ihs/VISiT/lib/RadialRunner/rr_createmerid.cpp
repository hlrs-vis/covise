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
#ifdef DIAGONAL_RUNNER
#include "include/diagonal.h"
#endif
#include "../General/include/points.h"
#include "../General/include/curve.h"
#include "../General/include/flist.h"
#include "../General/include/plane_geo.h"
#include "../General/include/parameter.h"
#include "../General/include/bias.h"
#include "../General/include/curvepoly.h"
#include "../BSpline/include/bspline.h"
#include "../General/include/common.h"
#include "../General/include/log.h"
#include "../General/include/fatal.h"
#include "../General/include/v.h"

#define SHROUD_EXT	0.75			  // extension factor of shroud height
#define HUB_EXT_RAD	0.1			  // radius at hub extension end
#define IN_EXT_H	0.05			  // height factor for inlet ext.
#define IN_EXT_R	0.2			  // radius factor for inlet ext.
#define BSPLN_DEGREE 3				  // bspline degree
#define SMALL  1.0E-04

extern int getContourData(struct radial *rr);

int CreateRR_MeridianContours(struct radial *rr)
{
	int i, j;
	int be_max, p_max;
	float t, p1[3], p3[3], p2[3], m[3], p0[3], p4[3];
	float sec, vec[3], s_end[3], s_ext[3], h_ext[3];
	float norm[3], u[3], v[3], u1[3], u2[3];
	float z_trans;
	struct Point *h_poly = NULL;
	struct Flist *h_knot = NULL;
	struct Point *s_poly = NULL;
	struct Point *poly2	 = NULL;
	struct Flist *s_knot = NULL;

	char fname[256];
	FILE *fcrv;

	// get contour data from database
	//if(rr->db2merid) getContourData(rr);
	getContourData(rr); // just an info at the moment!

	// to provide backwards compatibilty of old
	// cfg-files, set values if not set in cfg-file
	if((!rr->hspara[0] && !rr->hspara[1]) &&
	   (!rr->sspara[0] && !rr->sspara[1])) {
		rr->hspara[0] = rr->hspara[1] = 0.6f;
		rr->sspara[0] = rr->sspara[1] = 0.6f;
	}

	// blade edge contours
	// free + allocate memory
	if (rr->le->c) {
		FreeCurveStruct(rr->le->c);
		rr->le->c = NULL;
	}
	rr->le->c = AllocCurveStruct();
	if (rr->te->c) {
		FreeCurveStruct(rr->te->c);
		rr->te->c = NULL;
	}
	rr->te->c = AllocCurveStruct();

	// **************************************************
	// shroud contour
	// free + allocate memory
	if (s_poly) {
		FreePointStruct(s_poly);
		s_poly = NULL;
		s_knot = NULL;
	}
	p1[1] = 0.0;
	p3[1] = p3[2] = 0.0;
	p2[0] = 0.0;
	p2[1] = 0.0;
	p2[2] = 0.0;
	u[1]  = 0.0;
	v[1]  = 0.0;
	norm[1] = 0.0;
	// if straight contour at inlet
	// inlet point and vector
	p1[0] = rr->ref * rr->diam[0] * 0.5;
	p1[2] = rr->ref * rr->height;
	v[0]  = -sin(rr->angle[0]);
	v[2]  =	 cos(rr->angle[0]);
	// outlet point and vector
	p3[0] = rr->ref * rr->diam[1] * 0.5;
	u[0]  = -sin(rr->angle[1]);
	u[2]  =	 cos(rr->angle[1]);
	// shroud spline polygon

	dprintf(4,"\n CreateRR_MeridianContours(): s_poly-points\n");
	dprintf(4," p1 = [%16.8f  %16.8f  %16.8f]\n",p1[0],p1[1],p1[2]);
	dprintf(4," p2 = [%16.8f  %16.8f  %16.8f]\n",p2[0],p2[1],p2[2]);
	dprintf(4," p3 = [%16.8f  %16.8f  %16.8f]\n",p3[0],p3[1],p3[2]);
	dprintf(4," u	= [%16.8f  %16.8f  %16.8f]\n",u[0],u[1],u[2]);
	dprintf(4," v	= [%16.8f  %16.8f  %16.8f]\n",v[0],v[1],v[2]);

	LineIntersectXZ(p3, u, p1, v, p2);

	dprintf(4," p2 = [%16.8f  %16.8f  %16.8f]\n",p2[0],p2[1],p2[2]);
	if(rr->straight_cont[1]) {
		// move spline start point, poly2 = help-polygon
		for(i = 0; i < 3; i++) {
			u1[i] = p2[i] - p1[i];
			u2[i] = p2[i] - p3[i];
			p0[i] = u1[i]*rr->sstparam[0]+p1[i];
			p4[i] = u2[i]*rr->sstparam[1]+p3[i];
		}
		poly2 = CurvePolygon(p0, p2, p4, rr->sspara[0],rr->sspara[1]);
		s_poly = AllocPointStruct();
		AddVPoint(s_poly,p1);
		for(i = 0; i < poly2->nump; i++) {
			AddPoint(s_poly,poly2->x[i],poly2->y[i],poly2->z[i]);
		}
		AddVPoint(s_poly,p3);
		FreePointStruct(poly2);

		dprintf(4,"straight shroud, %f, %f\n",
				rr->sstparam[0],rr->sstparam[1]);
		for(i = 0; i < s_poly->nump; i++) {
			dprintf(4,"%d %f %f %f\n",
					i,s_poly->x[i],s_poly->y[i],s_poly->z[i]);
		}
		dprintf(4," p0 = [%16.8f  %16.8f  %16.8f]\n",
				p0[0],p0[1],p0[2]);
		dprintf(4," p1 = [%16.8f  %16.8f  %16.8f]\n",
				p1[0],p1[1],p1[2]);
		dprintf(4," p3 = [%16.8f  %16.8f  %16.8f]\n",
				p3[0],p3[1],p3[2]);
		dprintf(4," p4 = [%16.8f  %16.8f  %16.8f]\n",
				p4[0],p4[1],p4[2]);
	}
	else {
		s_poly = CurvePolygon(p1, p2, p3,rr->sspara[0], rr->sspara[1]);
	}
	dprintf(4,"\n CreateRR_MeridianContours(): s_poly-points\n");
	dprintf(4," p1 = [%16.8f  %16.8f  %16.8f]\n",p1[0],p1[1],p1[2]);
	dprintf(4," p2 = [%16.8f  %16.8f  %16.8f]\n",p2[0],p2[1],p2[2]);
	dprintf(4," p3 = [%16.8f  %16.8f  %16.8f]\n",p3[0],p3[1],p3[2]);

	// **************************************************
	// hub contour
	if (h_poly) {
		FreePointStruct(h_poly);
		h_poly = NULL;
		FreeFlistStruct(h_knot);
		h_knot= NULL;
	}
	h_poly = AllocPointStruct();
	// inlet point and vector
	norm[0] = -sin(rr->angle[0] - rr->iop_angle[1] - 0.5 * M_PI);
	norm[2] =  cos(rr->angle[0] - rr->iop_angle[1] - 0.5 * M_PI);
	p1[0]  += rr->ref * rr->cond[0] * norm[0];
	p1[2]  += rr->ref * rr->cond[0] * norm[2];
	v[0]	= -sin(rr->angle[0] - (rr->iop_angle[0] + rr->iop_angle[1]));
	v[2]	=  cos(rr->angle[0] - (rr->iop_angle[0] + rr->iop_angle[1]));
	// outlet point and vector
	norm[0] = -sin(rr->angle[1] - rr->oop_angle[1] + 0.5 * M_PI);
	norm[2] =  cos(rr->angle[1] - rr->oop_angle[1] + 0.5 * M_PI);
	p3[0]  +=  rr->ref * rr->cond[1] * norm[0];
	p3[2]  +=  rr->ref * rr->cond[1] * norm[2];
	u[0]	= -sin(rr->angle[1] - (rr->oop_angle[0] + rr->oop_angle[1]));
	u[2]	=  cos(rr->angle[1] - (rr->oop_angle[0] + rr->oop_angle[1]));
	// hub spline polygon
	dprintf(4,"\n CreateRR_MeridianContours(): h_poly-points\n");
	dprintf(4," p1 = [%16.8f  %16.8f  %16.8f]\n",p1[0],p1[1],p1[2]);
	dprintf(4," p2 = [%16.8f  %16.8f  %16.8f]\n",p2[0],p2[1],p2[2]);
	dprintf(4," p3 = [%16.8f  %16.8f  %16.8f]\n",p3[0],p3[1],p3[2]);
	LineIntersectXZ(p3, u, p1, v, p2);
	if(rr->straight_cont[0]) {
		// move spline start point, poly2 = help-polygon
		for(i = 0; i < 3; i++) {
			u1[i] = p2[i] - p1[i];
			u2[i] = p2[i] - p3[i];
			p0[i] = u1[i]*rr->hstparam[0]+p1[i];
			p4[i] = u2[i]*rr->hstparam[1]+p3[i];
		}
		poly2 = CurvePolygon(p0, p2, p4, rr->hspara[0],rr->hspara[1]);
		h_poly = AllocPointStruct();
		AddVPoint(h_poly,p1);
		for(i = 0; i < poly2->nump; i++) {
			AddPoint(h_poly,poly2->x[i],poly2->y[i],poly2->z[i]);
		}
		AddVPoint(h_poly,p3);
		FreePointStruct(poly2);
		dprintf(4,"straight hub, %f, %f\n",
				rr->hstparam[0],rr->hstparam[1]);
		for(i = 0; i < h_poly->nump; i++) {
			dprintf(4,"%d %f %f %f\n",
					i,h_poly->x[i],h_poly->y[i],h_poly->z[i]);
		}
		dprintf(4," p0 = [%16.8f  %16.8f  %16.8f]\n",
				p0[0],p0[1],p0[2]);
		dprintf(4," p1 = [%16.8f  %16.8f  %16.8f]\n",
				p1[0],p1[1],p1[2]);
		dprintf(4," p3 = [%16.8f  %16.8f  %16.8f]\n",
				p3[0],p3[1],p3[2]);
		dprintf(4," p4 = [%16.8f  %16.8f  %16.8f]\n",
				p4[0],p4[1],p4[2]);
	}
	else {
		h_poly = CurvePolygon(p1, p2, p3,rr->hspara[0], rr->hspara[1]);
	}
	dprintf(4,"\n CreateRR_MeridianContours(): h_poly-points\n");
	dprintf(4," p1 = [%16.8f  %16.8f  %16.8f]\n",p1[0],p1[1],p1[2]);
	dprintf(4," p2 = [%16.8f  %16.8f  %16.8f]\n",p2[0],p2[1],p2[2]);
	dprintf(4," p3 = [%16.8f  %16.8f  %16.8f]\n",p3[0],p3[1],p3[2]);
	// translate inlet mid plane to z = 0
	z_trans = 0.5 * (h_poly->z[0] + s_poly->z[0]);
	for (i = 0; i < s_poly->nump; i++)
		s_poly->z[i] -= z_trans;
	for (i = 0; i < h_poly->nump; i++)
		h_poly->z[i] -= z_trans;

	// interpolate remaining meridian curves from hub/shroud contour
	// spline knot vectors
	s_knot = BSplineKnot(s_poly, BSPLN_DEGREE);
	h_knot = BSplineKnot(h_poly, BSPLN_DEGREE);

	// leading edge start/end point and gradients
	BSplinePoint(BSPLN_DEGREE, h_poly, h_knot, rr->le->para[0], p1);
	AddCurvePoint(rr->le->c, p1[0], p1[1], p1[2], 0.0, 0.0);
	BSplineNormalXZ(BSPLN_DEGREE, h_poly, h_knot, rr->le->para[0], rr->le->h_norm);
	BSplinePoint(BSPLN_DEGREE, s_poly, s_knot, rr->le->para[1], p1);
	AddCurvePoint(rr->le->c, p1[0], p1[1], p1[2], 0.0, 0.0);
	BSplineNormalXZ(BSPLN_DEGREE, s_poly, s_knot, rr->le->para[1], rr->le->s_norm);

	// trailing edge start/end points and gradients
	BSplinePoint(BSPLN_DEGREE, h_poly, h_knot, rr->te->para[0], p1);
	AddCurvePoint(rr->te->c, p1[0], p1[1], p1[2], 0.0, 0.0);
	BSplineNormalXZ(BSPLN_DEGREE, h_poly, h_knot, rr->te->para[0], rr->te->h_norm);
	BSplinePoint(BSPLN_DEGREE, s_poly, s_knot, rr->te->para[1], p1);
	AddCurvePoint(rr->te->c, p1[0], p1[1], p1[2], 0.0, 0.0);
	BSplineNormalXZ(BSPLN_DEGREE, s_poly, s_knot, rr->te->para[1], rr->te->s_norm);

	for (i = 0; i < NPOIN_MERIDIAN; i++) {
		sec = (float)i/(float)(NPOIN_MERIDIAN - 1);
		dprintf(4,"\n CreateRR_MeridianContours(): i = %3d\n",i);
#ifdef DEBUG_MERIDIANS
		DumpPoints(h_poly,NULL);
#endif
		BSplinePoint(BSPLN_DEGREE, h_poly, h_knot, sec, p1);
		BSplinePoint(BSPLN_DEGREE, s_poly, s_knot, sec, p2);
		vec[0] = p2[0] - p1[0];
		vec[1] = p2[1] - p1[1];
		vec[2] = p2[2] - p1[2];
		dprintf(4," BSplinePoint ... done!\n");
		sprintf(fname, "rr_calcmeridian_%02d.txt", i);
#ifdef DEBUG_MERIDIANS
		if ((fcrv = fopen(fname, "w")) == NULL) {
			dprintf(0, "error opening meridian file '%s' *yakk*\n", fname);
			return 0;
		}
		fprintf(fcrv," vec = [%16.8f  %16.8f  %16.8f]\n",vec[0],vec[1],vec[2]);
		fprintf(fcrv," p1  = [%16.8f  %16.8f  %16.8f]\n",p1[0],p1[1],p1[2]);
#endif
		for (j = 0; j < rr->be_num; j++) {
			if (i == 0)	{				 // delete previous, allocate new
				if (rr->be[j]->ml) {
					FreeCurveStruct(rr->be[j]->ml);
					rr->be[j]->ml = NULL;
				}
				rr->be[j]->ml = AllocCurveStruct();
#ifndef NO_INLET_EXT
				if(rr->be[j]->ml->p->portion < NPOIN_EXT) {
					rr->be[j]->ml->p->portion = NPOIN_EXT + 10;
				}
				// init point mem. and leave space for pre-ext.
				AddCurvePoint(rr->be[j]->ml, 0.0, 0.0, 0.0, 0.0, 0.0);
				rr->be[j]->ml->p->nump = NPOIN_EXT-1;
#endif
			}
			m[0] = p1[0] + rr->be[j]->para * vec[0];
			m[1] = p1[1] + rr->be[j]->para * vec[1];
			m[2] = p1[2] + rr->be[j]->para * vec[2];
#ifdef DEBUG_MERIDIANS
			fprintf(fcrv," m   = [%16.8f  %16.8f  %16.8f]\n",m[0],m[1],m[2]);
#endif
			AddCurvePoint(rr->be[j]->ml, m[0], m[1], m[2], 0.0, rr->be[j]->para);
		}						// end j
#ifdef GAP
		if(i == 0) {
			if(rr->gp->ml) {
				FreeCurveStruct(rr->gp->ml);
				rr->gp->ml = NULL;
			}
			rr->gp->ml = AllocCurveStruct();
#ifndef NO_INLET_EXT
			if(rr->gp->ml->p->portion < NPOIN_EXT) {
				rr->gp->ml->p->portion = NPOIN_EXT + 1;
			}
			// init point mem. and leave space for inlet-ext.
			AddCurvePoint(rr->gp->ml, 0.0, 0.0, 0.0, 0.0, 0.0);
			rr->gp->ml->p->nump = NPOIN_EXT-1;
#endif					 // NO_INLET_EXT
		}
		m[0] = p1[0] + rr->gp->para * vec[0];
		m[1] = p1[1] + rr->gp->para * vec[1];
		m[2] = p1[2] + rr->gp->para * vec[2];
		AddCurvePoint(rr->gp->ml, m[0], m[1], m[2], 0.0, rr->gp->para);
#endif						// GAP
#ifdef DEBUG_MERIDIANS
		fclose(fcrv);
#endif
	}						   // end i

	FreePointStruct(s_poly);
	FreePointStruct(h_poly);
	FreeFlistStruct(s_knot);
	FreeFlistStruct(h_knot);

	// shroud contour end point and extension vector
	be_max = rr->be_num - 1;
	p_max  = rr->be[be_max]->ml->p->nump - 1;
	p3[0] = rr->ext_diam[1] * rr->ref * 0.5;
	p3[2] = rr->ext_height[1]*rr->ref;
	s_ext[0]  = s_end[0] = rr->be[be_max]->ml->p->x[p_max];
	s_ext[1]  = s_end[1] = rr->be[be_max]->ml->p->y[p_max];
	s_ext[1] -= rr->be[be_max]->ml->p->y[p_max-1];
	s_ext[2]  = s_end[2] = rr->be[be_max]->ml->p->z[p_max];
	if(p3[0]) {
		s_ext[0] -=  p3[0];
		s_ext[0] *= -1; // invert direction
		t = 1.0;
		s_ext[2]  = -p3[2];
	}
	else {
		s_ext[0] -= rr->be[be_max]->ml->p->x[p_max-1];
		s_ext[2] -= rr->be[be_max]->ml->p->z[p_max-1];
		t = (rr->ext_height[1] * rr->ref)/fabs(s_ext[2]);
	}

	dprintf(4," rr->ext_diam[1] = %f, p3[0] = %f\n", rr->ext_diam[1], p3[0]);
	dprintf(4," s_ext = [%f  %f  %f]\n",s_ext[0], s_ext[1], s_ext[2]);

	p3[1]  = s_end[1] + t * s_ext[1];
	p3[2]  = s_end[2] + t * s_ext[2];
	// hub contour end point and extension vector
	p_max  = rr->be[0]->ml->p->nump - 1;
	h_ext[0]  = p1[0] = rr->be[0]->ml->p->x[p_max];
	h_ext[0] -= rr->be[0]->ml->p->x[p_max-1];
	h_ext[1]  = p1[1] = rr->be[0]->ml->p->y[p_max];
	h_ext[1] -= rr->be[0]->ml->p->y[p_max-1];
	h_ext[2]  = p1[2] = rr->be[0]->ml->p->z[p_max];
	h_ext[2] -= rr->be[0]->ml->p->z[p_max-1];
	// hub extension end point and end vector
	p3[0] -= rr->ref * rr->ext_cond[1];
	// keep p3[2], z-coord.
	p3[1] = 0.0;
	v[0]  = 0.0;
	v[1]  = 0.0;
	v[2]  = 1.0;
	if((fabs(h_ext[0]) < SMALL) && (fabs(p1[0]-p3[0]) > SMALL)) v[0] = 1.0;
	LineIntersectXZ(p1, h_ext,p3, v, p2);
	if(p2[2] <= p3[2] || p2[2] >= p1[2]) {
		p2[2] = 0.5*(p1[2]+p3[2]);
		p2[0] = p1[0] + (p2[2]-p1[2])/h_ext[2]*h_ext[0];
	}
	h_poly = CurvePolygon(p1, p2, p3, 0.5, 0.5);
	h_knot = BSplineKnot(h_poly, BSPLN_DEGREE);
	// runner meridian contours extension
	for (i = 1; i < NPOIN_EXT; i++) {
		sec = (float)i/(float)(NPOIN_EXT - 1);
		BSplinePoint(BSPLN_DEGREE, h_poly, h_knot, sec, p1);
		p2[0]  = s_end[0] + t * sec * s_ext[0];
		p2[1]  = s_end[1] + t * sec * s_ext[1];
		p2[2]  = s_end[2] + t * sec * s_ext[2];
		vec[0] = p2[0] - p1[0];
		vec[1] = p2[1] - p1[1];
		vec[2] = p2[2] - p1[2];
		for(j = 0; j < rr->be_num; j++) {
			m[0] = p1[0] + rr->be[j]->para * vec[0];
			m[1] = p1[1] + rr->be[j]->para * vec[1];
			m[2] = p1[2] + rr->be[j]->para * vec[2];
			AddCurvePoint(rr->be[j]->ml, m[0], m[1], m[2], 0.0, rr->be[j]->para);
		}
#ifdef GAP
		m[0] = p1[0] + rr->gp->para * vec[0];
		m[1] = p1[1] + rr->gp->para * vec[1];
		m[2] = p1[2] + rr->gp->para * vec[2];
		AddCurvePoint(rr->gp->ml, m[0], m[1], m[2], 0.0, rr->gp->para);
#endif
	}
	FreePointStruct(h_poly);
	FreeFlistStruct(h_knot);
	//**************************************************

	// calculate curve arc lengths
	for (j = 0; j < rr->be_num; j++)
		CalcCurveArclen(rr->be[j]->ml);
#ifdef GAP
	CalcCurveArclen(rr->gp->ml);
#endif

#ifndef NO_INLET_EXT
	// create inlet extension, shroud
	// end point, beginning of runner part
	p_max = NPOIN_EXT-1;
	p3[0] = rr->be[be_max]->ml->p->x[p_max];
	p3[2] = rr->be[be_max]->ml->p->z[p_max];
	v[0]  =	 sin(rr->angle[0]);
	v[2]  = -cos(rr->angle[0]);
	// start point
	p1[0] = rr->ext_diam[0] * rr->ref * 0.5;
	p1[2] = p3[2] + rr->ext_height[0] * rr->ref;
	u[0] = -sin(rr->ext_iangle);
	u[2] = -cos(rr->ext_iangle);
	LineIntersectXZ(p3,v, p1,u, p2);
	dprintf(4,"p_max = %d\n",p_max);
	dprintf(4,"rr->be[be_max]->ml->p->nump = %d\n",rr->be[be_max]->ml->p->nump);
	dprintf(4,"rr->be[be_max]->ml->p->x[p_max] = %f\n",rr->be[be_max]->ml->p->x[p_max]);
	dprintf(4,"rr->be[be_max]->ml->p->z[p_max] = %f\n",rr->be[be_max]->ml->p->z[p_max]);
	dprintf(4,"p1[0]/v[0] * v[2] * t = %f\n",p1[0]/v[0] * v[2] * t);
	dprintf(4,"p1 = %f	 %f	 %f\n",p1[0], p1[1], p1[2]);
	dprintf(4,"p2 = %f	 %f	 %f\n",p2[0], p2[1], p2[2]);
	dprintf(4,"p3 = %f	 %f	 %f\n",p3[0], p3[1], p3[2]);
	// for backward compatibility, see above
	if(!rr->sspara_inext[0] && !rr->sspara_inext[1]
	   && !rr->hspara_inext[0] && !rr->hspara_inext[1]) {
		rr->sspara_inext[0] = rr->sspara_inext[1] = 0.5;
		rr->hspara_inext[0] = rr->hspara_inext[1] = 0.5;
	}
	s_poly = CurvePolygon(p1,p2,p3, rr->sspara_inext[0], rr->sspara_inext[1]);
	s_knot = BSplineKnot(s_poly, BSPLN_DEGREE);

	// hub
	// last point, beginning runner part
	p3[0] = rr->be[0]->ml->p->x[p_max];
	p3[2] = rr->be[0]->ml->p->z[p_max];
	v[0]	=  sin(rr->angle[0] - (rr->iop_angle[0] + rr->iop_angle[1]));
	v[2]	= -cos(rr->angle[0] - (rr->iop_angle[0] + rr->iop_angle[1]));
	// starting point.
	p1[0] -= rr->ref * rr->ext_cond[0] * cos(rr->ext_iangle);
	p1[2] += rr->ref * rr->ext_cond[0] * sin(rr->ext_iangle);
	LineIntersectXZ(p3,v, p1,u, p2);
	h_poly = CurvePolygon(p1,p2,p3, rr->hspara_inext[0], rr->hspara_inext[1]);
	h_knot = BSplineKnot(h_poly, BSPLN_DEGREE);
	p_max = rr->be[0]->ml->p->nump;

	dprintf(4,"p_max = %d\n",p_max);
	for(i = 0; i < NPOIN_EXT-1; i++) {
		sec = (float)i/(float)(NPOIN_EXT - 1);
		BSplinePoint(BSPLN_DEGREE, h_poly, h_knot, sec, p1);
		BSplinePoint(BSPLN_DEGREE, s_poly, s_knot, sec, p2);
		vec[0] = p2[0] - p1[0];
		vec[1] = p2[1] - p1[1];
		vec[2] = p2[2] - p1[2];
		for(j = 0; j < rr->be_num; j++) {
			// shift back pointers
			if(i == 0)
			{
				rr->be[j]->ml->p->nump = 0;
			}
			m[0] = p1[0] + rr->be[j]->para * vec[0];
			m[1] = p1[1] + rr->be[j]->para * vec[1];
			m[2] = p1[2] + rr->be[j]->para * vec[2];
			AddCurvePoint(rr->be[j]->ml, m[0], m[1], m[2], 0.0, rr->be[j]->para);
		}
#ifdef GAP
		if(i == 0) {
			rr->gp->ml->p->nump = 0;
		}
		m[0] = p1[0] + rr->gp->para * vec[0];
		m[1] = p1[1] + rr->gp->para * vec[1];
		m[2] = p1[2] + rr->gp->para * vec[2];
		AddCurvePoint(rr->gp->ml, m[0], m[1], m[2], 0.0, rr->gp->para);
#endif
	}						   // end i

	FreePointStruct(s_poly);
	FreePointStruct(h_poly);
	FreeFlistStruct(s_knot);
	FreeFlistStruct(h_knot);
	// calculate curve arc lengths
	for (j = 0; j < rr->be_num; j++) {
		rr->be[j]->ml->p->nump = p_max;
		CalcCurveArclen(rr->be[j]->ml);
	}
#ifdef GAP
	rr->gp->ml->p->nump = p_max;
	CalcCurveArclen(rr->gp->ml);
#endif
#endif						   // !NO_INLET_EXT

	for (j = 0; j < rr->be_num; j++) {
		sprintf(fname, "rr_meridian_%02d.txt", j);
		if ((fcrv = fopen(fname, "w")) == NULL) {
			dprintf(0, "error writing meridian file '%s' *yakk*\n", fname);
			return 0;
		}
		DumpCurve(rr->be[j]->ml, fcrv);
		fclose(fcrv);
	}
#ifdef GAP
	sprintf(fname, "rr_meridian_gap.txt");
	if ((fcrv = fopen(fname, "w")) == NULL) {
		fprintf(stdout, "error writing meridian file '%s' *yakk*\n", fname);
		return 0;
	}
	DumpCurve(rr->gp->ml, fcrv);
	fclose(fcrv);
#endif						   // DEBUG_MERIDIANS
	return 0;
}
