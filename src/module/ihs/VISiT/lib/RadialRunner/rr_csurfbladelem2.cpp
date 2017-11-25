#include <stdio.h>
#include <stdlib.h>
#ifdef WIN32
#include <windows.h>
#else
#include <strings.h>
#endif
#ifdef WIN32
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
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
#include "../General/include/common.h"
#include "../General/include/log.h"
#include "../General/include/fatal.h"
#include "../General/include/v.h"
#include "../BSpline/include/bspline.h"

#define A -1.0
#define NUMP 51

// from rr_comp.cpp
extern struct Point *GetCartesianCoord(struct Point *src);

static float alpha(float a, float x, float *beta, float dbeta);
static float cfunct(float a, float x);
static int Surface2Cl(struct curve *ml, struct Point *cl, 
		      struct Point *beclmerid, struct profile *bp,
		      float scale, float cl_len, 
		      float te_thick, int dfact, struct Point *surf, 
		      struct Point *smerid);

#ifdef DEBUG_CSURF
static struct Point *GetConformalView(struct Point *src, float l0);
#endif

// in contrast to the first version (starting at the trailing edge) 
// this routine starts at the leading edge. The bl_wrap is taken to describe
// the ABSOLUTE position of the leading edge.
int CSurfacesRR_BladeElement2(struct be *be, float lepar, float tepar,
			     float rle, float rte, float hle, float hte,
			     int clock, float camb_flag)
{
	int i, j, jnext, err;
	int ite, ile;

	float dbeta, dpar, beta[2], l[2], deltal, dl, ll, ll2;
	float bb, s, ds, rr, r[2], x, a;
	float p[3], p2[3], z, dphi, par, scale;

	struct Point *cl;
	struct Point *clmerid;
	struct Point *beclmerid;
	struct Point *cl2;
	struct Point *cl2merid;
	struct Flist *cl2par = NULL;

#ifdef CHECK_LEN
	struct Flist *lpar   = NULL; //see below
	float dl2;
#endif

	struct Point *psmerid = NULL;

#ifdef DEBUG_CSURF
	static int call = 0;
	float alp, p_l, alpha_deg;
	int ixcount = 0;
	char fn[222];
	FILE *fp;

	sprintf(fn,"rr_debugcsurf_%02d.txt",call++);
	if( (fp = fopen(fn,"w+")) == NULL) {
		fprintf(stderr," Could not open file '%s'\n");
		exit(1);
	}
	psmerid = AllocPointStruct();
	fprintf(fp,"# camb_para = %f\n",be->camb_para);
	fprintf(fp,"# x,bb,(bb-beta[1])/dbeta,ll (index %d)\n",ixcount++);
#endif

	// **************************************************
	// inits
	err = 0;
	beta[0] = be->angle[0];
	beta[1] = be->angle[1];
	dbeta = beta[0]-beta[1];
	dpar  = lepar - tepar;
	a     = be->camb_para;

	// **************************************************
	// free old data, get new memory
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
	be->cl  = AllocPointStruct();
	be->clg = AllocPointStruct();
	be->ps  = AllocPointStruct();
	be->ss  = AllocPointStruct();

	// get index of prev. meridian point before blade edge intersection
	if( (ile = GetPointIndex(be->ml->p->nump,
							 be->ml->par, lepar, 0)) == -1) {
	    fprintf(stderr," be->para = %.5f\n",be->para);
		fatal(" le point not found!");
	}
	if( (ite = GetPointIndex(be->ml->p->nump,
				 be->ml->par, tepar, ile)) == -1) {
	    fprintf(stderr," be->para = %.5f\n",be->para);
		fatal("te point not found!");
	}

	// get meridian length of intersection point
	l[0] = lepar * be->ml->len[be->ml->p->nump-1];
	l[1] = tepar * be->ml->len[be->ml->p->nump-1];
	deltal = l[1]-l[0];
	dl = deltal / (float)(NUMP-1);

	// position of le on circumference
	s = be->bl_wrap * rle;

	// **************************************************
	// create smooth center line (many points)
	p2[2] = 0.0;
	cl = AllocPointStruct();
	clmerid = AllocPointStruct();
	AddPoint(cl, rle, be->bl_wrap, hle);
	AddPoint(clmerid, s, l[0], 0.0);
	ll   = ll2 = l[0];
	r[0] = rle;
	x    = 1.0;
	bb   = alpha(a, x, beta, dbeta);
	jnext = MAX(0,ile-1);
	for(i = 1; i < NUMP; i++) {
		ll = l[0]+i*dl;
		x  = (l[1]-ll) / deltal;
		bb = alpha(a, x, beta, dbeta);
#ifdef DEBUG_CSURF
		fprintf(fp,"%14.5f %14.5f %14.5f %14.5f\n",
			x,bb,(bb-beta[1])/dbeta,ll);
#endif
		// get radius of next point
		for(j = jnext; j < be->ml->p->nump; j++) {
			if( (be->ml->len[j] <= ll) &&
			    (be->ml->len[j+1]) >= ll) {
				jnext = MAX(0,j-1);
				par = (ll - be->ml->len[j]) / 
				    (be->ml->len[j+1] - be->ml->len[j]);
				r[1] = be->ml->p->x[j] + 
				    par * (be->ml->p->x[j+1]- be->ml->p->x[j]);
				z = be->ml->p->z[j] + 
				    par * (be->ml->p->z[j+1]- be->ml->p->z[j]);
				break;
			}
			continue;
		}
		rr = 0.5f*(r[1] + r[0]);
		ds = (ll2 - ll) / float(tan(bb));
		dphi = ds/rr;
		p[0] = r[1];
		p[1] = cl->y[cl->nump-1] + dphi;
		p[2] = z;
		AddVPoint(cl,p);
		p2[0] = r[1]*p[1];
		p2[1] = ll;
		AddVPoint(clmerid,p2);

		ll2  = ll;
		r[0] = r[1];
	}

	// **************************************************
	// check length parametrisation, holding off on that
#ifdef CHECK_LEN
	ll = ll2 = 0.0;
	lpar = AllocFlistStruct(cl->nump+1);
	Add2Flist(lpar,ll);
#ifdef DEBUG_CSURF
	fprintf(fp,"\n\n# i, ll, ll2, ll-ll2, beta (index %d)\n",ixcount++);
#endif
	for(i = 1; i < cl->nump; i++) {
		rr  = 0.5*(cl->x[i-1]+cl->x[i]);
		ds  = rr*(cl->y[i] - cl->y[i-1]);
		ll += sqrt(ds*ds + pow( (cl->z[i] - cl->z[i-1]), 2.0 ) );
#ifdef DEBUG_CSURF
		fprintf(fp," %4d %10.5f %10.5f %10.5f %10.5f\n",
				i,ll,ll2,ll-ll2, atan((cl->z[i] - cl->z[i-1])/ds));
#endif
		ll2 = ll;
	}

	FreeFlistStruct(lpar);
#endif
	// **************************************************
	// inverse direction (le->te) from cl (te->le)
	cl2 = AllocPointStruct();
	cl2merid = AllocPointStruct();
	for(i = 0; i < cl->nump; i++) {
	    AddPoint(cl2,cl->x[i], cl->y[i], cl->z[i]);
	    AddPoint(cl2merid,clmerid->x[i], clmerid->y[i], clmerid->z[i]);
	}

	// **************************************************
	// get cl2-param
	cl2par   = AllocFlistStruct(cl2->nump+1);
	ll    = 0.0;
	Add2Flist(cl2par,ll);
	for(i = 1; i < cl2->nump; i++) {
	    r[0] = (cl2->x[i]+cl2->x[i-1])*0.5f;
	    dphi = cl2->y[i]-cl2->y[i-1];
	    ll2  = float(sqrt( pow(cl2->z[i]-cl2->z[i-1],2) +
			pow(cl2->x[i]-cl2->x[i-1],2) ));
	    ll   = float(sqrt( pow(ll2,2) + pow(r[0]*dphi,2)));
	    Add2Flist(cl2par,ll+cl2par->list[(cl2par->num-1)]);
	}

	ll = 1.0f/cl2par->list[(cl2par->num-1)];
	for(i = 0; i < cl2->nump; i++) {
	    cl2par->list[i] *= ll;
	}
	cl2par->list[ (cl2par->num-1)] = 1.0;

	// **************************************************
	// interpolate to real be->cl
	j = 0;
	beclmerid = AllocPointStruct();
	for(i = 0; i < be->bp->num; i++) {
	    x   = float(pow(be->bp->c[i], be->bp_shift));
	    par = 0.0;
	    while( (x >= cl2par->list[j]) && (j < cl2par->num)) j++;
	    if(j == 0) j = 1;
	    par  = (x-cl2par->list[j-1]) / (cl2par->list[j]-cl2par->list[j-1]);
	    p[0] = cl2->x[j-1] + par*(cl2->x[j]-cl2->x[j-1]);
	    p[1] = cl2->y[j-1] + par*(cl2->y[j]-cl2->y[j-1]);
	    p[2] = cl2->z[j-1] + par*(cl2->z[j]-cl2->z[j-1]);
	    AddVPoint(be->cl,p);
	    p2[0] = cl2merid->x[j-1] + par*(cl2merid->x[j]-cl2merid->x[j-1]);
	    p2[1] = cl2merid->y[j-1] + par*(cl2merid->y[j]-cl2merid->y[j-1]);
	    AddVPoint(beclmerid,p2);

	    j = MAX(0,j-1);
	}

	be->cl_len = 0.0;
	for(i = 1; i < be->bp->num; i++) {
		be->cl_len += float(sqrt(pow( (beclmerid->x[i]-
					 beclmerid->x[i-1]), 2)
				   + pow( (beclmerid->y[i]-
					   beclmerid->y[i-1]), 2) ));
	}

	// **************************************************
	// blade profile to cl
	scale = be->p_thick;
	Surface2Cl(be->ml, be->cl, beclmerid, be->bp, scale, be->cl_len,
		   be->te_thick,-1,be->ss,psmerid);
	Surface2Cl(be->ml, be->cl, beclmerid, be->bp, scale, be->cl_len,
		   be->te_thick,1,be->ps,psmerid);
	// **************************************************
	// blade in cartesian coords.
	be->cl_cart = GetCartesianCoord(be->cl);
	be->ps_cart = GetCartesianCoord(be->ps);
	be->ss_cart = GetCartesianCoord(be->ss);
	

	// **************************************************
#ifdef DEBUG_CSURF
	fprintf(fp,"\n\n# index %d, cl: r   phi   z     dphi*r_mitt  dl  l\n",
		ixcount++);
	fprintf(fp,"%14.5f %14.5f %14.5f\n",
		cl->x[0], cl->y[0], cl->z[0], 1.0,0.0,0.0);
	ll   = ll2  = 0.0;
	r[0] = r[1] = cl->x[0];
	for(i = 1; i < cl->nump; i++) {
		r[0] = (cl->x[i]+cl->x[i-1])*0.5;
		dphi = cl->y[i]-cl->y[i-1];
		ll   = sqrt( pow(cl->z[i]-cl->z[i-1],2) + 
			     pow(r[0]-r[1],2) );
		ll2 += ll;
		bb   = atan(ll/(dphi*r[0]));
		r[1] = r[0];
		fprintf(fp,"%14.5f %14.5f %14.5f     %14.5f %14.5f %14.5f\n",
			cl->x[i], cl->y[i], cl->z[i], bb, ll, ll2);
	}

	// cl2merid
	fprintf(fp,"\n\n# index %d, cl2merid: r*phi, l_merid, z\n", ixcount++);
	for(i = 0; i < cl2merid->nump; i++) {
	    fprintf(fp,"%14.5f %14.5f %14.5f\n",
		    cl2merid->x[i], cl2merid->y[i], cl2merid->z[i]);
	}

	// beclmerid
	fprintf(fp,"\n\n# index %d, beclmerid: r*phi, l_merid, z\n", ixcount++);
	for(i = 0; i < beclmerid->nump; i++) {
	    fprintf(fp,"%14.5f %14.5f %14.5f\n",
		    beclmerid->x[i], beclmerid->y[i], beclmerid->z[i]);
	}

	// be->cl_cart
	fprintf(fp,"\n\n# index %d, be->cl_cart: x, y, z\n", ixcount++);
	for(i = 0; i < be->cl_cart->nump; i++) {
	    fprintf(fp,"%14.5f %14.5f %14.5f\n",
		    be->cl_cart->x[i], be->cl_cart->y[i], be->cl_cart->z[i]);
	}

	// be->ps_cart
	fprintf(fp,"\n\n# index %d, be->ps_cart: x, y, z\n", ixcount++);
	for(i = 0; i < be->ps_cart->nump; i++) {
	    fprintf(fp,"%14.5f %14.5f %14.5f\n",
		    be->ps_cart->x[i], be->ps_cart->y[i], be->ps_cart->z[i]);
	}

	// be->ss_cart
	fprintf(fp,"\n\n# index %d, be->ss_cart: x, y, z\n", ixcount++);
	for(i = 0; i < be->ss_cart->nump; i++) {
	    fprintf(fp,"%14.5f %14.5f %14.5f\n",
		    be->ss_cart->x[i], be->ss_cart->y[i], be->ss_cart->z[i]);
	}

	// psmerid
	fprintf(fp,"\n\n# index %d, psmerid: r*phi, l_merid, z\n", ixcount++);
	for(i = 0; i < psmerid->nump; i++) {
	    fprintf(fp,"%14.5f %14.5f %14.5f\n",
		    psmerid->x[i], psmerid->y[i], psmerid->z[i]);
	}

	fclose(fp);
	FreePointStruct(psmerid);
#endif

	FreePointStruct(cl);
	FreePointStruct(clmerid);
	FreePointStruct(cl2);
	FreePointStruct(cl2merid);
	FreeFlistStruct(cl2par);
	FreePointStruct(beclmerid);

#ifdef DEBUG_CSURF
	sprintf(fn,"rr_debugccamb_%02d.txt",call-1);
	if( (fp = fopen(fn,"w+")) == NULL) {
		fprintf(stderr," Could not open file '%s'\n");
		exit(1);
	}
	cl = GetConformalView(be->cl, 0.0);

	ll = cl->z[cl->nump-2];
	for(i = 1; i < cl->nump; i++) {
		alp	  = atan((cl->y[i-1]-cl->y[i])/(cl->x[i-1]-cl->x[i]));
		p_l	  = cl->z[i-1]/ll;
		if((alpha_deg = alp*180.0/M_PI) < 0.0) alpha_deg += 180.0;
		fprintf(fp,"%6d  %14.5f  %14.5f\n",i,p_l,alpha_deg);
	}

	FreePointStruct(cl);

	fclose(fp);
#endif

	return err;
}

static float alpha(float a, float x, float *beta, float dbeta)
{
	return ( (beta[1] + dbeta*cfunct(a,x)) );
}

static float cfunct(float a, float x)
{
	return float( (a*x + (1.0-a))*pow(x,2));
}

static int Surface2Cl(struct curve *ml, struct Point *cl, 
		      struct Point *beclmerid, struct profile *bp,
		      float scale, float cl_len, 
		      float te_thick, int dfact, struct Point *surf,
		      struct Point *smerid)
{
    int i, j, jnext, ii;
    float n[3], p[3], p2[3], t1, t2, ll, par, r, rr, phi, z;

    jnext = 1;
    n[2]  = 0.0;
    AddPoint(surf, cl->x[0], cl->y[0], cl->z[0]);
    for(i = 1; i < cl->nump; i++) {
	if(i == cl->nump-1) ii = i;
	else ii = i+1;
	n[0]  = -(beclmerid->y[ii] - beclmerid->y[i-1]);
	n[1]  =   beclmerid->x[ii] - beclmerid->x[i-1];
	ll    = float(sqrt(pow(n[0],2) + pow(n[1],2)));
	n[0] /= ll; n[1] /= ll;
	    
	t1 = 0.5f * cl_len * bp->t[i];
	t2 = 0.5f * te_thick * bp->c[i];
	p[0] = beclmerid->x[i] + dfact*n[0] * (scale * t1 + t2);
	p[1] = beclmerid->y[i] + dfact*n[1] * (scale * t1 + t2);
	p[2] = beclmerid->z[i] + dfact*n[2] * (scale * t1 + t2);
	if(smerid) AddVPoint(smerid,p);
	for(j = jnext; j < ml->p->nump; j++) {
	    if( (ml->len[j] >= p[1]) &&
		(ml->len[j-1] <= p[1]) ) {
		jnext = MAX(j-1, 0);
		par = (p[1] - ml->len[j]) / 
		    (ml->len[j-1] - ml->len[j]);
		r = ml->p->x[j] + 
		    par * (ml->p->x[j-1]- ml->p->x[j]);
		z = ml->p->z[j] + 
		    par * (ml->p->z[j-1]- ml->p->z[j]);
		break;
	    }
	    continue;
	}
	rr    = 0.5f * (r + cl->x[i]);
	phi   = p[0]/rr;
	p2[0] = r;
	p2[1] = phi;
	p2[2] = z;
	AddVPoint(surf, p2);
    }

    return 0;
}

#ifdef DEBUG_CSURF
static struct Point *GetConformalView(struct Point *src, float l0)
{
	int i;
	float l, s, dl, ds, len;

	struct Point *line;

	line = AllocPointStruct();

	len = 0.0;
	l = l0;
	s = src->x[src->nump-1] * src->y[src->nump-1];
	AddPoint(line,s,l,len);
	for(i = src->nump-2; i >= 0; i--) {
		dl	= -l;
		l  += sqrt(pow(src->x[i+1]-src->x[i],2) + pow(src->z[i+1]-src->z[i],2));
		dl += l;
		ds	= -s;
		s  += 0.5*(src->x[i+1] + src->x[i]) * (src->y[i] - src->y[i+1]);
		ds += s;
		len+= sqrt(dl*dl + ds*ds);
		AddPoint(line,s,l,len);
	}
	return (line);
}
#endif
