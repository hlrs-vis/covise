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
#include "../General/include/flist.h"
#include "../General/include/plane_geo.h"
#include "../General/include/profile.h"
#include "../General/include/bias.h"
#include "../General/include/curvepoly.h"
#include "../BSpline/include/bspline.h"
#include "../General/include/common.h"
#include "../General/include/log.h"
#include "../General/include/fatal.h"
#include "../General/include/v.h"

#define BSPLN_DEGREE 3				  // bspline degree

int CreateRR_BladeEdge(struct edge *e)
{
	int i;
	float angle, roma[2][2];
	float hub[3], shroud[3], inter[3], mid1[3], mid2[3];
	float h_vec[3], s_vec[3];
	float sec, p[3];
	struct Point *poly;				   // edge spline polygon
	struct Flist *knot;				   // edge spline knot vector

	// for backward compatibility
	if(!e->spara[0] && !e->spara[1]) {
		e->spara[0] = e->spara[1] = 0.6f;
	}

	mid1[1] = 0.0;
	mid2[1] = 0.0;
	hub[1]	= 0.0;
	shroud[1] = 0.0;
	inter[1]  = 0.0;
	h_vec[1]  = 0.0;
	s_vec[1]  = 0.0;
	// point and gradient hub side
	hub[0]	   = e->c->p->x[0];
	hub[2]	   = e->c->p->z[0];
	angle	   = e->angle[0] - M_PI_2;
	roma[0][0] =  cos(angle);
	roma[0][1] = -sin(angle);
	roma[1][0] =  sin(angle);
	roma[1][1] =  cos(angle);
	h_vec[0]   = e->h_norm[0] * roma[0][0] + e->h_norm[2] * roma[0][1];
	h_vec[2]   = e->h_norm[0] * roma[1][0] + e->h_norm[2] * roma[1][1];
	// move hub point to properly intersect with meridian contour
	hub[0] -= (0.01 * h_vec[0]);
	hub[2] -= (0.01 * h_vec[2]);
	// point and gradient shroud side
	shroud[0]  = e->c->p->x[e->c->p->nump-1];
	shroud[2]  = e->c->p->z[e->c->p->nump-1];
	angle	   = e->angle[1] - M_PI_2;
	roma[0][0] =  cos(angle);
	roma[0][1] = -sin(angle);
	roma[1][0] =  sin(angle);
	roma[1][1] =  cos(angle);
	s_vec[0]   = e->s_norm[0] * roma[0][0] + e->s_norm[2] * roma[0][1];
	s_vec[2]   = e->s_norm[0] * roma[1][0] + e->s_norm[2] * roma[1][1];
	// move shroud point to properly intersect with meridian contour
#ifdef GAP
	shroud[0] += (0.05 * s_vec[0]);
	shroud[2] += (0.05 * s_vec[2]);
#else
	shroud[0] += (0.01 * s_vec[0]);
	shroud[2] += (0.01 * s_vec[2]);
#endif
	// intersection and partition points
	LineIntersectXZ(hub, h_vec, shroud, s_vec, inter);
	mid1[0]	 = hub[0]	 + e->spara[0] * (inter[0] - hub[0]);
	mid1[2]	 = hub[2]	 + e->spara[0] * (inter[2] - hub[2]);
	mid2[0]	 = shroud[0] + e->spara[1] * (inter[0] - shroud[0]);
	mid2[2]	 = shroud[2] + e->spara[1] * (inter[2] - shroud[2]);
	poly = AllocPointStruct();
	AddVPoint(poly, hub);
	AddVPoint(poly, mid1);
	AddVPoint(poly, mid2);
	AddVPoint(poly, shroud);
	// calculate spline and points for curve (overwrite)
	knot = AllocFlistStruct(0);
	knot = BSplineKnot(poly, BSPLN_DEGREE);
	e->c->p->nump = 0;
	for (i = 0; i <= NPOIN_EDGE; i++) {
		sec = (float)i/(float)NPOIN_EDGE;
		BSplinePoint(BSPLN_DEGREE, poly, knot, sec, p);
		AddCurvePoint(e->c, p[0], p[1], p[2], 0.0, sec);
	}
	CalcCurveArclen(e->c);

	return 0;
}
