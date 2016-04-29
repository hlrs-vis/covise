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
#include "../General/include/profile.h"
#include "../General/include/bias.h"
#include "../General/include/curvepoly.h"
#include "../BSpline/include/bspline.h"
#include "../General/include/common.h"
#include "../General/include/log.h"
#include "../General/include/fatal.h"
#include "../General/include/v.h"

#define BSPLN_DEGREE 3				  // bspline degree
#define SMALL  1.0E-04

#define POLY_POINTS 15
#ifdef DEBUG_SURFACES
#define VPRINT(x) fprintf(stderr,"%10s = [%4f, %4f, %4f]\n",#x,x[0],x[1],x[2])
#define VPRINTF(x,f) fprintf(f,"%10s = [%4f, %4f, %4f]\n",#x,x[0], x[1], x[2])
static int DumpConformProjection(struct Point *p, float l0,	 char *name, int n);
#endif // DEBUG_SURFACES

static struct Point *GetMeridionalView(struct Point *cl, struct curve *ml,
									   float rref, float rle, float hle,
									   float lelen, float rte, float hte,
									   float telen, int ile, int ite,
									   int maxiter);
struct Point *GetCartesianCoord(struct Point *src);

// use conform projection, ref. radius for cylinder = 1.0 (left out)
int MSurfacesRR_BladeElement(struct be *be, float lepar, float tepar,
							 float rle, float rte, float hle, float hte,
							 int clock, int camb_flag)
{
	int i;
	int ite, ile;

	float bllen, telen, lelen, t1, t2;
	float sec, rref, scale, htesurf, rtesurf, telensurf;
	float p[3], p1[3], p2[3], p3[3], v1[3], v3[3];

	int warned;
	float delta, sdel, cdel, ratio;
	float q[3], q1[3], q2[3], q3[3], q4[3];
#ifdef LIN_WRAPOPT
	int j;
	float p1opt[3], p2opt[3], wrapopt, wrapopt_deg, dqx, ddqx;
	float ddqx_prev, wrapopt_prev, delta2;
	float q1opt[3], q2opt[3], q3opt[3], x4opt;
#endif

	struct Point *poly	= NULL;
	struct Point *poly2 = NULL;
	struct Flist *knot	= NULL;

#ifdef DEBUG_SURFACES
	float len = 0.0;
	char fn[123];
	char fndebug[123];
	FILE *fp;
	FILE *fpdebug;
	static int ncall = 0;
	static int ndebug = 0;

	sprintf(fndebug,"rr_debugsurface_%02d.txt",ndebug++);

	if( (fpdebug = fopen(fndebug,"w+")) == NULL) {
		fprintf(stderr,"Shit happened opening file '%s'\n",fndebug);
		exit(-1);
	}
	fprintf(stderr,"MSurfaceRR_BladeElement: %d\n",ndebug);
	fprintf(fpdebug,"MSurfaceRR_BladeElement: %d\n",ndebug);
	fflush(fpdebug);
#endif

	// ref. radius!
	rref = rte;

	// free old data, get new memory
	if (poly) {
		FreePointStruct(poly);
		poly = NULL;
	}
	if (poly2) {
		FreePointStruct(poly2);
		poly2 = NULL;
	}
	if (knot) {
		FreeFlistStruct(knot);
		knot = NULL;
	}

	// get index of prev. meridian point before blade edge intersection
	if( (ile = GetPointIndex(be->ml->p->nump, be->ml->par, lepar, 0)) == -1) {
		fatal("point not found!");
	}
	if( (ite = GetPointIndex(be->ml->p->nump, be->ml->par, tepar, ile)) == -1) {
		fatal("point not found!");
	}

#ifdef DEBUG_SURFACES
	fprintf(fpdebug, "ile = %d, rle = %f, hle = %f, lepar = %f, be->ml->para[ile] = %f\n",
			ile, rle, hle, lepar, be->ml->par[ile]);
	fprintf(fpdebug, "ite = %d, rte = %f, hte = %f, tepar = %f, be->ml->para[ite] = %f\n",
			ite, rte, hte, tepar, be->ml->par[ite]);
	fprintf(fpdebug, "be->ml->p->x[ite] = %f, be->ml->p->y[ite] = %f\n",
			be->ml->p->x[ite], be->ml->p->y[ite]);
#endif

	// get real blade length from meridian points
	bllen = 0;
	bllen = sqrt(pow( (float)(rte - be->ml->p->x[ite]), 2) + pow( (float)(hte - be->ml->p->z[ite]), 2) );
	// length coord. of leading & trailing edge
	lelen = be->ml->len[ile] + sqrt(pow( (float)(rle - be->ml->p->x[ile]), 2)
									+ pow( (float)(hle - be->ml->p->z[ile]), 2) );
	telen = be->ml->len[ite] + bllen;
	bllen *= 0.5*(1/rte + 1/be->ml->p->x[ite]);
#ifdef DEBUG_SURFACES
	fprintf(fpdebug,"bllen = %f\n",bllen);
#endif
	for(i = ite; i > ile+1; i--) {
		bllen += (be->ml->len[i] - be->ml->len[i-1]) * 0.5*(1/be->ml->p->x[i] + 1/be->ml->p->x[i-1]);
#ifdef DEBUG_SURFACES
		fprintf(fpdebug," i = %d, bllen = %f, ml->len[i] = %f, ml->p->x[i] = %f\n",
				i, bllen, be->ml->len[i], be->ml->p->x[i]);
#endif
	}
	bllen += sqrt(pow( (float)(rle - be->ml->p->x[ile+1]), 2) + pow( (float)(hle - be->ml->p->z[ile+1]), 2) )
		* 0.5*(1/rle + 1/be->ml->p->x[ile+1]);
	bllen *= rref;

#ifdef DEBUG_SURFACES
	fprintf(fpdebug,"lelen = %f, telen = %f, be->ml->len[ite] = %f\n",lelen, telen, be->ml->len[ite]);
	fprintf(fpdebug,"bllen = %f\n",bllen);
#endif

	// polygon for centre line spline
	p[0]   = p[1]  = p[2]  = 0.0;
	p1[2]  = p2[2] = p3[2] = 0.0;
	v1[2]  = v3[2] = 0.0;

	p1[0]  = (be->bl_wrap + be->te_wrap) * rref;
	p1[1]  = bllen;
	p3[0]  = be->te_wrap * rref;
	p3[1]  = 0.0;

	v1[0]  = -cos(be->angle[0]);
	v1[1]  = -sin(be->angle[0]);
	v3[0]  =  cos(be->angle[1]);
	v3[1]  =  sin(be->angle[1]);

	LineIntersect(p3,v3, p1,v1, p2);
	if( (p2[1] > p1[1])	 || (p2[0] < p3[0]) ) {
		fprintf(stderr,"MSurfaceRR_BladeElement: WARNING: ill blade geometry, may cause difficulties! (para = %f)\n", be->para);
#ifdef DEBUG_SURFACES
		fprintf(fpdebug," MSurfaceRR_BladeElement: WARNING: ill blade geometry, may cause difficulties!\n" );
	fflush(fpdebug);
#endif
	}
	// **************************************************
	if(camb_flag) {
		// force max. camber to given position, using polygon
		// transform points for simplicity
		delta = M_PI   - atan( (p1[1] - p3[1])/(p1[0] - p3[0]));
		cdel  = cos(delta);
		sdel  = sin(delta);
		q[2] = q1[2] = q2[2] = q3[2] = q4[2] = 0.0;
		q1[0] = cdel * p1[0] - sdel * p1[1];
		q1[1] = sdel * p1[0] + cdel * p1[1];
		q2[0] = cdel * p2[0] - sdel * p2[1];
		q2[1] = sdel * p2[0] + cdel * p2[1];
		q3[0] = cdel * p3[0] - sdel * p3[1];
		q3[1] = sdel * p3[0] + cdel * p3[1];
		q4[0] = q1[0] + (q3[0] - q1[0]) * be->camb_pos;
#ifdef LIN_WRAPOPT
		// find optimum wrap angle
		p2opt[2] = p1opt[2] = 0.0;
		p1opt[1] = p1[1];
		wrapopt = be->bl_wrap;
		ddqx = ddqx_prev = dqx = 0;
		dqx	 = (q4[0] - q2[0]);
		ddqx = dqx / (q1[0]-q3[0]);
		j = 50;
		i = 0;
		while(ABS(dqx) > SMALL && i++ < j) {
			wrapopt_prev = wrapopt;
			wrapopt	 +=	 ddqx * M_PI/180.0;
			if(i > 1) wrapopt  += -ddqx * ( (wrapopt_prev-wrapopt)/(ddqx_prev-ddqx));
			p1opt[0]  = (wrapopt + be->te_wrap) * rref;
			LineIntersect(p3,v3, p1opt,v1, p2opt);
			delta2 = M_PI	- atan( (p1[1] - p3[1])/(p1[0] - p3[0]));
			cdel  = cos(delta2);
			sdel  = sin(delta2);
			q1opt[0] = cdel * p1opt[0] - sdel * p1opt[1];
			q1opt[1] = sdel * p1opt[0] + cdel * p1opt[1];
			q2opt[0] = cdel * p2opt[0] - sdel * p2opt[1];
			q2opt[1] = sdel * p2opt[0] + cdel * p2opt[1];
			q3opt[0] = cdel * p3[0] - sdel * p3[1];
			q3opt[1] = sdel * p3[0] + cdel * p3[1];
			x4opt = q1opt[0] + (q3opt[0] - q1opt[0]) * be->camb_pos;
			dqx	 = (x4opt - q2opt[0]);
			ddqx_prev = ddqx;
			ddqx = dqx / (q1opt[0]-q3opt[0]);
#ifdef DEBUG_SURFACES
			fprintf(fpdebug," wrap loop: i = %2d, wrapopt = %f, dqx = %f, ddqx = %f\n",
					i, wrapopt, dqx, ddqx);
			fflush(fpdebug);
#endif
		}
		wrapopt_deg = wrapopt * 180/M_PI;
		if(i >= j) fprintf(stderr,"MSurfaceRR_BladeElement: WARNING: wrap opt. loop limit!\n");
#endif						// LIN_WRAPOPT

#ifdef DEBUG_SURFACES
		VPRINTF(p1,fpdebug);
		VPRINTF(p2,fpdebug);
		VPRINTF(p3,fpdebug);
		fprintf(fpdebug," delta = %f\n", delta *180/M_PI);
		VPRINTF(q1,fpdebug);
		VPRINTF(q2,fpdebug);
		VPRINTF(q3,fpdebug);

		sprintf(fn,"rr_cambpoly_%02d.txt",ncall);
		if( (fp = fopen(fn,"w+")) == NULL) {
			fprintf(stderr,"file '%s'!\n",fn);
			exit(-1);
		}
		fprintf(fp,"# transformed points p1/2/3 -> q1/2/3\n");
		fprintf(fp," %f	 %f	 %f\n", q1[0], q1[1], q1[2]);
		fprintf(fp," %f	 %f	 %f\n", q2[0], q2[1], q2[2]);
		fprintf(fp," %f	 %f	 %f\n", q3[0], q3[1], q3[2]);
		fprintf(fp," %f	 %f	 %f\n", q4[0], q3[1], q3[2]);
		fprintf(fp,"\n\n");
		fflush(fpdebug);
		fflush(fp);
#endif

		// get maximum camber
		if(q4[0] <= q2[0]) {
			q4[1] = q1[1] + be->camb * ((q2[1]-q1[1])/(q2[0]-q1[0]) * (q4[0] - q1[0]));
		}
		else {
			q4[1] = q3[1] + be->camb * ((q2[1]-q3[1])/(q2[0]-q3[0]) * (q4[0] - q3[0]));
		}

#ifdef DEBUG_SURFACES
		fprintf(fpdebug,"q4[1] = %f\n",q4[1]);
		fprintf(fpdebug,"(q3[1]-q2[1]) = %f, (q3[0]-q2[0]) = %f\n",(q3[1]-q2[1]), (q3[0]-q2[0]));
		fprintf(fpdebug,"(q4[0] - q2[0]) = %f, q2[1] = %f\n",(q4[0] - q2[0]),  q2[1]);

		fprintf(fp,"# transformed points p1/2/3 -> q1/2/3\n");
		fprintf(fp," %f	 %f	 %f\n", q1[0], q1[1], q1[2]);
		fprintf(fp," %f	 %f	 %f\n", q2[0], q2[1], q2[2]);
		fprintf(fp," %f	 %f	 %f\n", q3[0], q3[1], q3[2]);
		fprintf(fp," %f	 %f	 %f\n", q4[0], q4[1], q4[2]);
		fprintf(fp,"\n\n");
#endif

		// create polygons for cl spline
#ifdef DEBUG_SURFACES
		fprintf(fp,"# spline points\n");
#endif
		v1[2] = v3[2] = 0.0;
		if(be->le_para == 0.0 && be->te_para == 0.0) {
			t1 = 0.5;
			t2 = 0.5;
		}
		else {
			t1 = be->le_para; t2 = be->le_para;
		}
		poly2 = AllocPointStruct();
		// spline: q1 to q4
		q[1]  = q4[1];
		q[0]  = q1[0] + (q2[0] - q1[0])*((q[1] - q1[1])/(q2[1] - q1[1]));
#ifdef DEBUG_SURFACES
		fprintf(fp,"\n %f	%f	 %f\n\n", q[0], q[1], q[2]);
#endif
		AddVPoint(poly2, q1);
		poly  = CurvePolygon(q1,q,q4, t1, t2);		// auxiliary polygon
		knot  = BSplineKnot(poly, BSPLN_DEGREE);
		for(i = 1; i < POLY_POINTS; i++) {
			ratio = (float)(i)/(float)(POLY_POINTS-1);
			ratio = pow((float)ratio,1.5f);
			BSplinePoint(BSPLN_DEGREE, poly, knot, ratio, p);
			AddVPoint(poly2,p);
		}
		FreePointStruct(poly);
		FreeFlistStruct(knot);

		// spline polygon: q4 to q3
		q[1]  = q4[1];
		q[0]  = q3[0] + (q2[0] - q3[0])*((q[1] - q3[1])/(q2[1] - q3[1]));
#ifdef DEBUG_SURFACES
		fprintf(fp," %f	  %f   %f\n\n", q[0], q[1], q[2]);
#endif
		ratio = be->camb/4;
		p[0] = q3[0] + ratio*(q2[0]-q3[0]);
		p[1] = q3[1] + ratio*(q2[1]-q3[1]);
		if(be->le_para == 0.0 && be->te_para == 0.0) {
			t1 = 0.5;
			t2 = 0.5;
		}
		else {
			t1 = be->te_para; t2 = be->te_para;
		}
		poly  = CurvePolygon(q4,q,p, t1, t2);
		knot  = BSplineKnot(poly, BSPLN_DEGREE);
		for(i = 1; i < POLY_POINTS; i++) {
			ratio = (float)(i)/(float)(POLY_POINTS-1);
			BSplinePoint(BSPLN_DEGREE, poly, knot, ratio, q);
			AddVPoint(poly2,q);
		}
		ratio = be->camb/8;
		AddPoint(poly2, q3[0] + ratio*(q2[0]-q3[0]),
				 q3[1] + ratio*(q2[1]-q3[1]), 0.0);
		AddVPoint(poly2, q3);
		FreePointStruct(poly);
		FreeFlistStruct(knot);

#ifdef DEBUG_SURFACES
		fprintf(fp," poly2\n");
		for(i = 0; i < poly2->nump; i++) {
			fprintf(fp," %f	  %f   %f\n", poly2->x[i],
					poly2->y[i],poly2->z[i]);
		}

		fprintf(fp,"\n\n# polynom, poly2\n");
		knot = BSplineKnot(poly2, BSPLN_DEGREE);
		for(i = 0; i < be->bp->num; i++) {
			sec = pow(be->bp->c[i], be->bp_shift);
			BSplinePoint(BSPLN_DEGREE, poly2, knot, sec, p);
			fprintf(fp," %f	  %f   %f\n", p[0], p[1], p[2]);
			if(i == 0) len = 0;
			else len += sqrt(pow(p[0]-q[0], 2.0) + pow(p[1]-q[1], 2.0));
			q[0] = p[0];
			q[1] = p[1];
		}
		fprintf(fp,"\n\n# spline, cl-parameters\n");
		for(i = 0; i < be->bp->num; i++) {
			sec = pow(be->bp->c[i], be->bp_shift);
			BSplinePoint(BSPLN_DEGREE, poly2, knot, sec, p);
			if(i == 0) ratio = 0;
			else ratio += sqrt(pow(p[0]-q[0], 2.0) + pow(p[1]-q[1], 2.0))/len;
			if(sec != 0)
				fprintf(fp," ratio, sec: %f	  %f  %8.3f\n",ratio, sec, 100*(ratio-sec)/sec);
			else fprintf(fp," ratio, sec: %f   %f  %8.3f\n",ratio, sec, 0.0);
			q[0] = p[0];
			q[1] = p[1];
		}
		FreeFlistStruct(knot);
#endif

		// transform and modify (if necessary) polynom
#ifdef DEBUG_SURFACES
		fprintf(fp,"\n\n # modified polynom\n");
#endif
		poly = AllocPointStruct();
		warned = 0;
		for(i = 0; i < poly2->nump; i++) {
			if(poly2->y[i] <= q4[1]) {
				p[0]  =	 cdel * poly2->x[i] + sdel * poly2->y[i];
				p[1]  = -sdel * poly2->x[i] + cdel * poly2->y[i];
				AddVPoint(poly, p);
#ifdef DEBUG_SURFACES
				fprintf(fp," %f	 %f	 %f	 # i: %d\n", poly->x[i],poly->y[i],0.0,i);
#endif
			}
			else {
				warned = 1;
				fprintf(stderr,"MSurfaceRR_BladeElement: Warning: Maximum camber exceeded! (para = %f, %f)\n",
						be->para, poly2->y[i]/q4[1] * be->camb);
			}
		}
#ifdef LIN_WRAPOPT
		if(warned || be->para == 0.0 || be->para == 0.5 || be->para == 1.0) {
			fprintf(stderr,"MSurfaceRR_BladeElement: para = %4.2f: optimal wrap angle! %f (%f deg).\n",
					be->para, wrapopt, wrapopt_deg);
		}
		fflush(fpdebug);
#endif

		// check polygon, nump > 2!
		if(poly->nump <= 2 || q4[1] < (q3[1] + be->camb_pos * (q1[1]-q3[1]))) {
			FreePointStruct(poly);
			poly = AllocPointStruct();
			t1	 = 0.5;
			t2	 = 0.5;
			poly = CurvePolygon(p1,p2,p3, t1, t2);
			fprintf(stderr,"MSurfaceRR_BladeElement: Warning: invalid polygon, camber values omitted! (para = %f\n", be->para);
		}

#ifdef DEBUG_SURFACES
		fprintf(fp," %f	 %f	 %f\n",q3[0], q3[1], q3[2]);
		fclose(fp);
#endif
	}						   // if(camb_flag)
	// **************************************************
	else {
		if(be->le_para == 0.0 && be->te_para == 0.0) {
			t1 = 0.5;
			t2 = 0.5;
		}
		else {
			t1 = be->le_para; t2 = be->te_para;
		}
		fprintf(stderr,"MSurfaceRR_BladeElement: camber values omitted!\n");
#ifdef DEBUG_SURFACES
		fprintf(fpdebug,"\n ** t1 = %f, t2 = %f, camb_pos = %f\n",t1, t2,be->camb_pos);
		fflush(fpdebug);
#endif
		// auxiliary Polygon poly2, add extra point
		// to force te blade angle.
		poly2 = CurvePolygon(p1,p2,p3, t1, t2);
		poly  = AllocPointStruct();
		if(poly2) {
			for(i = 0; i < poly2->nump-1; i++)
				AddPoint(poly,poly2->x[i],
						 poly2->y[i], poly2->z[i]);
			AddPoint(poly, p3[0]+(poly2->x[poly2->nump-2]-p3[0])/8,
					 p3[1]+(poly2->y[poly2->nump-2]-p3[1])/8,
					 p3[2]+(poly2->z[poly2->nump-2]-p3[2])/8);
			i = poly2->nump-1;
			AddPoint(poly,poly2->x[i],
					 poly2->y[i], poly2->z[i]);
		}
		else return POLYGON_ERROR;

	}						   // !camb_flag
	// **************************************************
	knot = BSplineKnot(poly, BSPLN_DEGREE);

#ifdef DEBUG_SURFACES
	fprintf(fpdebug,"# polygon, camb_pos = %f, ang0 = %f, ang1 = %f\n",
			be->camb_pos, be->angle[0]*180/M_PI, be->angle[1]*180/M_PI);
	for(i = 0; i < poly->nump; i++) {
		fprintf(fpdebug," %f  %f  %f\n",poly->x[i], poly->y[i], poly->z[i]);
		fprintf(stderr," %f	 %f	 %f\n",poly->x[i], poly->y[i], poly->z[i]);
	}
#endif

#ifdef DEBUG_SURFACES
	sprintf(fn,"rr_be2_%02d.txt",ncall++);
	if( (fp = fopen(fn,"w+")) == NULL) {
		fprintf(stderr,"file '%s'!\n",fn);
		exit(-1);
	}
	fprintf(fp,"# polygon, camb_pos = %f, ang0 = %f, ang1 = %f\n",
			be->camb_pos, be->angle[0]*180/M_PI, be->angle[1]*180/M_PI);
	for(i = 0; i < poly->nump; i++) {
		fprintf(fp," %f	 %f	 %f\n",poly->x[i], poly->y[i], poly->z[i]);
	}
	fprintf(fp," %f	 %f	 %f\n",p2[0], p2[1], p2[2]);
	fprintf(fp," %f	 %f	 %f\n",p1[0], p1[1], p1[2]);
	fprintf(fp," %f	 %f	 %f\n",p3[0], p3[1], p3[2]);
	fprintf(fp,"\n\n");
	fflush(fpdebug);
#endif

	// memory check
	if (be->cl) {
		FreePointStruct(be->cl);
		be->cl = NULL;
		FreePointStruct(be->cl_cart);
		be->cl_cart = NULL;
		FreePointStruct(be->clg);
		be->clg = NULL;
	}
	if (be->ps) {
		FreePointStruct(be->ps);
		be->ps = NULL;
		FreePointStruct(be->ps_cart);
		be->ps_cart = NULL;
	}
	if (be->ss) {
		FreePointStruct(be->ss);
		be->ss = NULL;
		FreePointStruct(be->ss_cart);
		be->ss_cart = NULL;
	}
	be->cl	= AllocPointStruct();
	be->clg = AllocPointStruct();
	be->ps	= AllocPointStruct();
	be->ss	= AllocPointStruct();

	// centre line and gradients
#ifdef DEBUG_SURFACES
	fprintf(fp,"# centre line\n");
#endif
	for(i = 0; i < be->bp->num; i++) {
		sec = pow(be->bp->c[i], be->bp_shift);
		BSplinePoint(BSPLN_DEGREE, poly, knot, sec, p);
		AddVPoint(be->cl, p);
		BSplineNormal(BSPLN_DEGREE, poly, knot, sec, p);
		AddVPoint(be->clg, p);
#ifdef DEBUG_SURFACES
		fprintf(fp," %f	  %f   %f	%f	 %f\n",be->cl->x[i], be->cl->y[i],
				be->cl->z[i],be->clg->x[i], be->clg->y[i]);
#endif
	}
	FreePointStruct(poly);
	FreePointStruct(poly2);
	FreeFlistStruct(knot);
#ifdef DEBUG_SURFACES
	fprintf(fp,"\n\n");
	fprintf(fp," # wrap angles: para, te_wrap, bl_wrap+te_wrap, bllen/rref, cl->x[0]/rref, cl->x[last]/rref, mod_angle[0/1]\n");
	fprintf(fp," %f	 %f	 %f	 %f	 %f	 %f	 %f	 %f\n",be->para, be->te_wrap, be->bl_wrap+be->te_wrap, bllen/rref,
			be->cl->x[0]/rref, be->cl->x[be->cl->nump-1]/rref, be->mod_angle[0], be->mod_angle[1]);
	fprintf(fp,"\n\n");
#endif

	// cl length in projection
	be->cl_len = 0.0;
	for(i = 1; i < be->bp->num; i++) {
		be->cl_len += sqrt(pow( (float)(be->cl->x[i]-be->cl->x[i-1]), 2)
						   + pow( (float)(be->cl->y[i]-be->cl->y[i-1]), 2) );
	}

	scale = be->p_thick;
#ifdef DEBUG_SURFACES
	fprintf(fp,"\n\ncl_len = %f, scale = %f\n",be->cl_len, scale);
#endif
	// calc. pressure side surface
	for (i = 0; i < be->bp->num; i++) {
		t1 = 0.5 * be->cl_len * be->bp->t[i];
		t2 = 0.5 * be->te_thick * be->bp->c[i];
		p[0] = be->cl->x[i] - be->clg->x[i] * (scale * t1 + t2);
		p[1] = be->cl->y[i] - be->clg->y[i] * (scale * t1 + t2);
		p[2] = be->cl->z[i] - be->clg->z[i] * (scale * t1 + t2);
		AddVPoint(be->ps, p);
#ifdef DEBUG_SURFACES
		fprintf(fp," %f	  %f   %f	%f\n",
				be->ps->x[i], be->ps->y[i], be->ps->z[i], 2*(scale*t1+t2));
#endif
	}
#ifdef DEBUG_SURFACES
	fprintf(fp,"\n\ncl_len = %f\n",be->cl_len);
#endif
	// calc. suct. side surface
	for (i = 0; i < be->bp->num; i++) {
		t1 = 0.5 * be->cl_len * be->bp->t[i];
		t2 = 0.5 * be->te_thick * be->bp->c[i];
		p[0] = be->cl->x[i] + be->clg->x[i] * (scale * t1 + t2);
		p[1] = be->cl->y[i] + be->clg->y[i] * (scale * t1 + t2);
		p[2] = be->cl->z[i] + be->clg->z[i] * (scale * t1 + t2);
		AddVPoint(be->ss, p);
#ifdef DEBUG_SURFACES
		fprintf(fp," %f	  %f   %f	%f\n",be->ss->x[i], be->ss->y[i], be->ss->z[i], 2*(scale*t1+t2));
#endif
	}
#ifdef DEBUG_SURFACES
	fprintf(fp,"\n\n");
#endif

#ifdef DEBUG_SURFACES
	fprintf(fp,"\n\n #center line in cylindrical view\n");
	for(i = 1; i < be->cl->nump; i++) {
		fprintf(fp," %10.5f	 %10.5f	 %10.5f	  %7.3f\n",
				be->cl->x[i], be->cl->y[i], be->cl->z[i],
				atan((be->cl->y[i]-be->cl->y[i-1])/
					 (be->cl->x[i]-be->cl->x[i-1]))*180/M_PI);
	}
	fflush(fpdebug);
#endif

	// transform curves to meridian coordinates
	be->cl = GetMeridionalView(be->cl, be->ml, rref, rle, hle,
							   lelen, rte, hte, telen, ile, ite, 3);
	if(be->te_thick == 0.0) {
		telensurf = telen;
		rtesurf	  = rte;
		htesurf	  = hte;
	}
	else telensurf = rtesurf = htesurf = 0.0;
	be->ps = GetMeridionalView(be->ps, be->ml, rref, rle, hle,
							   lelen, rtesurf, htesurf, telensurf,
							   ile, ite, 3);
	be->ss = GetMeridionalView(be->ss, be->ml, rref, rle, hle,
							   lelen, rtesurf, htesurf, telensurf,
							   ile, ite, 3);
	if((be->cl->nump != be->ss->nump)||(be->cl->nump != be->ps->nump)) {
		fprintf(stderr,"\nMSurfaceRR_BladeElement: lost point while transforming to meridional view!\n\n");
		return CONF_ERR;
	}
#ifdef DEBUG_SURFACES
	fprintf(fpdebug," x = %f, y = %f, z = %f\n", be->cl->x[be->cl->nump-1],
			be->cl->y[be->cl->nump-1], be->cl->z[be->cl->nump-1]);
	fprintf(fpdebug,"ss->nump = %d: x = %f, y = %f, z = %f\n", be->ss->nump, be->ss->x[be->ss->nump-1],
			be->cl->y[be->ss->nump-1], be->ss->z[be->ss->nump-1]);
#endif

#ifdef DEBUG_SURFACES
	DumpConformProjection(be->cl, lelen - telen, "rr_beclconf", ndebug-1);
	DumpConformProjection(be->ps, lelen - telen + sqrt(pow((be->cl->y[be->cl->nump-1]- be->ps->y[be->ps->nump-1]),2.0) +
													   pow((be->cl->z[be->cl->nump-1]- be->ps->z[be->ps->nump-1]),2.0)),
						  "rr_bepsconf", ndebug-1);
	DumpConformProjection(be->ss, lelen - telen - sqrt(pow((be->cl->y[be->cl->nump-1]- be->ss->y[be->ss->nump-1]),2.0) +
													   pow((be->cl->z[be->cl->nump-1]- be->ss->z[be->ss->nump-1]),2.0)),
						  "rr_bessconf", ndebug-1);
#endif

#ifdef DEBUG_SURFACES
	fclose(fp);
	sprintf(fn,"rr_be3_%02d.txt",ncall-1);
	if( (fp = fopen(fn,"w+")) == NULL) {
		fprintf(stderr,"file '%s'!\n",fn);
		exit(-1);
	}

	fprintf(fp,"# center line: radius  phi	height\n");
	for(i = 0; i < be->cl->nump; i++) {
		fprintf(fp," %f	 %f	 %f\n",be->cl->x[i],be->cl->y[i],be->cl->z[i]);
	}
	fprintf(fp,"\n\n# ss: radius  phi  height\n");
	for(i = 0; i < be->ss->nump; i++) {
		fprintf(fp," %f	 %f	 %f\n",be->ss->x[i],be->ss->y[i],be->ss->z[i]);
	}
	fprintf(fp,"\n\n# ps: radius  phi  height\n");
	for(i = 0; i < be->ps->nump; i++) {
		fprintf(fp," %f	 %f	 %f\n",be->ps->x[i],be->ps->y[i],be->ps->z[i]);
	}

	fprintf(fpdebug," ss->nump = %d, ps->nump = %d\n", be->ss->nump, be->ps->nump);
#endif

	// change direction of rotation if necessary
	if(clock) {
		for(i = 0; i < be->cl->nump; i++) be->cl->y[i] *= -1.0;
		for(i = 0; i < be->ss->nump; i++) be->ss->y[i] *= -1.0;
		for(i = 0; i < be->ps->nump; i++) be->ps->y[i] *= -1.0;
	}

	// get cartesian coords.
	be->cl_cart = GetCartesianCoord(be->cl);
	be->ps_cart = GetCartesianCoord(be->ps);
	be->ss_cart = GetCartesianCoord(be->ss);

#ifdef DEBUG_SURFACES
	fclose(fp);
	sprintf(fn,"rr_becart_%02d.txt",ncall-1);
	if( (fp = fopen(fn,"w+")) == NULL) {
		fprintf(stderr,"file '%s'!\n",fn);
		exit(-1);
	}

	fprintf(fp,"# center line: x  y	 z\n");
	scale = 0.0;				   // param., cl-length
	for(i = 0; i < be->cl_cart->nump; i++) {
		if(i) {
			scale += sqrt( pow(be->cl_cart->x[i]-be->cl_cart->x[i-1],2)
						   + pow(be->cl_cart->y[i]-be->cl_cart->y[i-1],2)
						   + pow(be->cl_cart->z[i]-be->cl_cart->z[i-1],2));
		}
		fprintf(fp," %f	 %f	 %f\n",be->cl_cart->x[i],be->cl_cart->y[i],be->cl_cart->z[i]);
	}
	fprintf(fp,"\n\n# ss\n");
	for(i = 0; i < be->ss_cart->nump; i++) {
		fprintf(fp," %f	 %f	 %f\n",be->ss_cart->x[i],be->ss_cart->y[i],be->ss_cart->z[i]);
	}
	fprintf(fp,"\n\n# ps\n");
	for(i = 0; i < be->ps_cart->nump; i++) {
		fprintf(fp," %f	 %f	 %f\n",be->ps_cart->x[i],be->ps_cart->y[i],be->ps_cart->z[i]);
	}

	fprintf(fpdebug," ss->nump = %d, ps->nump = %d\n", be->ss->nump, be->ps->nump);
#endif

#ifdef DEBUG_SURFACES
	fprintf(fpdebug," ... MSurfacesRR_BladeElement() ... done!\n");
	fflush(fpdebug);
	fclose(fpdebug);
	fclose(fp);
	sprintf(fn,"rr_bethick_%02d.txt",ncall-1);
	if( (fp = fopen(fn,"w+")) == NULL) {
		fprintf(stderr,"file '%s'!\n",fn);
		exit(-1);
	}
	fprintf(fp,"\n# normalized cl-length, blade-element-thickness\n");
	t2 = 0.0;
	for(i = 0; i < be->cl_cart->nump; i++) {
		if(i) {
			t2 += sqrt( pow(be->cl_cart->x[i]-be->cl_cart->x[i-1],2)
						+ pow(be->cl_cart->y[i]-be->cl_cart->y[i-1],2)
						+ pow(be->cl_cart->z[i]-be->cl_cart->z[i-1],2));
		}
		else t2 = 0.0;
		t1 = sqrt( pow(be->ss_cart->x[i]-be->ps_cart->x[i], 2)
				   + pow(be->ss_cart->y[i]-be->ps_cart->y[i], 2)
				   + pow(be->ss_cart->z[i]-be->ps_cart->z[i], 2));
		fprintf(fp," %8.4f	%8.4f  %8.4f\n", t2/scale, t1, scale);
	}
	fclose(fp);
#endif

	return 0;
}

// **************************************************
// transform points from conf. proj. to meridian
static struct Point *GetMeridionalView(struct Point *cl, struct curve *ml,
									   float rref, float rle, float hle,
									   float lelen, float rte, float hte,
									   float telen, int ile, int ite,
									   int maxiter)
{
	int i, j, iter, jnext, nump, jadd;

	float l, lprev, p[3];

	struct Point *line;				   // transformed curve

	if(rte && hte) {
		nump = cl->nump-1;
	}
	else {
		nump = cl->nump;
	}

	jadd = 8;
	if(ml->p->nump <= ite+jadd) jadd = ml->p->nump - (ite+1);

#ifdef DEBUG_SURFACES
	fprintf(stderr, " nump = %d,cl->nump = %d\n",nump, cl->nump);
#endif
	line = AllocPointStruct();
	AddPoint(line, rle, cl->x[0]/rref, hle);	   // le point
	l = lelen;
	jnext = ile;
	// get initial radius values for points
	for(i = 1; i < nump; i++) {
		lprev = l;
		p[1] = cl->x[i] / rref;
		l = lprev + ( (cl->y[i-1] - cl->y[i]) / rref ) * line->x[i-1];
		for(j = jnext; j < ite+jadd; j++) {
			if(l <= ml->len[j]) {
				jnext = MAX(0,j-1);
				p[2] = (ml->p->z[j] - ml->p->z[j-1])/
					(ml->len[j] - ml->len[j-1])
					* (l - ml->len[j-1]) + ml->p->z[j-1];
				p[0] = (ml->p->x[j] - ml->p->x[j-1])/
					(ml->len[j] - ml->len[j-1])
					* (l - ml->len[j-1]) + ml->p->x[j-1];
				AddVPoint(line,p);
				break;
			}
			continue;
		}
	}

	// te radius and hte pos. are known
	if(rte && hte) {
		AddPoint(line, rte, cl->x[cl->nump-1]/rref, hte);
#ifdef DEBUG_SURFACES
		fprintf(stderr, "hte = %f, rte = %f, telen = %f\n", hte, rte, telen);
#endif
	}

	// maxiter iterations to get real radii
	for(iter = 0; iter < maxiter; iter++) {
		// forward le -> te
		l = lelen;
		jnext = ile;
		for(i = 1; i < nump; i++) {
			lprev = l;
			l = lprev + ( (cl->y[i-1] - cl->y[i]) / rref ) * 0.5*(line->x[i-1]+line->x[i]);
			for(j = jnext; j < ite+jadd; j++) {
				if(l <= ml->len[j])
				{
					jnext = MAX(0,j-1);
					line->z[i] = (ml->p->z[j] - ml->p->z[j-1])/(ml->len[j] - ml->len[j-1])
						* (l - ml->len[j-1]) + ml->p->z[j-1];
					line->x[i] = (ml->p->x[j] - ml->p->x[j-1])/(ml->len[j] - ml->len[j-1])
						* (l - ml->len[j-1]) + ml->p->x[j-1];
					break;
				}
				continue;
			}
		}

#ifdef TRANSFORM_MVIEW_BACKWARDS
		// backwards te -> le
		l = telen;
#ifdef DEBUG_SURFACES
		fprintf(stderr, "backwards: l = %f, telen = %f\n", l, telen);
#endif
		jnext = ite+2;
		for(i = nump-1; i >= 1; i--) {
			lprev = l;
			l = lprev - ( (cl->y[i] - cl->y[i+1]) / rref ) * 0.5*(line->x[i]+line->x[i+1]);
			for(j = jnext; j > ile; j--) {
				if(l >= ml->len[j-1])
				{
					jnext = j;
					line->z[i] = (ml->p->z[j] - ml->p->z[j-1])/(ml->len[j] - ml->len[j-1])
						* (l - ml->len[j-1]) + ml->p->z[j-1];
					line->x[i] = 1.0*line->x[i] + 0.0*(ml->p->x[j] - ml->p->x[j-1])/(ml->len[j] - ml->len[j-1])
						* (l - ml->len[j-1]) + ml->p->x[j-1];
					break;
				}
				continue;
			}
		}
#endif						// TRANSFORM_MVIEW_BACKWARDS
	}						   // end iter
	// set cl ptr. to new points, free old (conf. proj.)
	FreePointStruct(cl);
	cl = line;

#ifdef DEBUG_SURFACES
	fprintf(stderr,"GetMeridionalView: nump = %d\n",cl->nump);
#endif

	return(cl);
}

// **************************************************
struct Point *GetCartesianCoord(struct Point *src)
{
	int i;
	float p[3];

	struct Point *line;

	line = AllocPointStruct();

	for(i = 0; i < src->nump; i++) {
		p[0] = src->x[i] * cos(src->y[i]);
		p[1] = src->x[i] * sin(src->y[i]);
		p[2] = src->z[i];
		AddVPoint(line,p);
	}

	return line;
}

// **************************************************
#ifdef DEBUG_SURFACES
static int DumpConformProjection(struct Point *p, float l0,	 char *name, int n)
{
	int i;
	float l, s, dl, ds, len;

	char fn[111];
	FILE *fp;

	sprintf(fn,"%s_%02d.txt",name, n);
	if( (fp = fopen(fn,"w+")) == NULL) {
		fprintf(stderr,"file '%s'!\n",fn);
		exit(-1);
	}

	s  = l = len = 0;
	l  = l0;
	i  = p->nump-1;
	s  = p->x[i] * p->y[i];
	fprintf(fp," %f	 %f\n", s, l);
	fprintf(fp,"# i: %d, x = %f, y = %f, z = %f\n", i, p->x[i], p->y[i], p->z[i]);
	for(i = p->nump-2; i >= 0; i--) {
		dl = -l;
		l += sqrt( pow(p->x[i+1] - p->x[i], 2.0) + pow(p->z[i+1] - p->z[i], 2.0));
		dl += l;
		ds = -s;
		s += 0.5 * (p->x[i+1] + p->x[i]) * (p->y[i] - p->y[i+1]);
		ds += s;
		len += sqrt(dl*dl + ds*ds);
		fprintf(fp," %f	 %f	 %f	 %f\n", s, l, atan(dl/ds)*180/M_PI, len);
		fprintf(fp,"# i: %d, x = %f, y = %f, z = %f\n", i, p->x[i], p->y[i], p->z[i]);
	}

	fclose(fp);
	return 0;
}
#endif // DEBUG_SURFACES
