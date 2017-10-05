#include <stdio.h>
#include <stdlib.h>
#ifdef WIN32
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#endif
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

#ifndef SMALL
#define SMALL 0.0001
#endif
#ifndef TINY
#define TINY 1.0e-8
#endif
#ifndef ABS
#define ABS(a)	  ( (a) >= (0) ? (a) : -(a) )
#endif
#ifndef MAX
#define MAX(a,b)	( (a) > (b) ? (a) : (b) )
#endif
#define SIGN(a)	   ( (a) >= (0) ? (1) : -(1) )
#ifndef BSPLN_DEGREE
#define BSPLN_DEGREE 3
#endif

#ifdef DEBUG_REGIONS
#define VPRINT(x) fprintf(stderr,"%10s = [%4f, %4f, %4f]\n",#x,x[0], x[1], x[2])
#define VPRINTF(x,f) fprintf(f,"%10s = [%4f, %4f, %4f]\n",#x,x[0], x[1], x[2])
#endif

struct cgrid **SwitchCGridElements(struct cgrid **cge, int index1, int index2)
{
	struct cgrid *cgetmp;

	cgetmp		= cge[index1];
	cge[index1] = cge[index2];
	cge[index2] = cgetmp;

	return cge;
}


struct ge **SwitchBladeElements(struct ge **ge, int index1, int index2)
{
	struct ge *getmp;

	getmp	   = ge[index1];
	ge[index1] = ge[index2];
	ge[index2] = getmp;

	return ge;
}

// line->x,y,z->phi,l,r
struct Flist *GetCircleArclen(struct Point *line)
{
	int i;
	struct Flist *arc;

	arc = AllocFlistStruct(line->nump+1);

	for(i = 0; i < line->nump; i++)
	{
		Add2Flist(arc, line->x[i]*line->z[i]);
	}

	return arc;
}


struct Flist *GetCircleArclenXY(struct Point *line)
{
	int i;
	struct Flist *arc;

	arc = AllocFlistStruct(line->nump+1);

	for(i = 0; i < line->nump; i++)
	{
		Add2Flist(arc, line->x[i]*line->y[i]);
	}

	return arc;
}


int InterpolPoint(struct Point *ml, float *len, float dphi,
				  struct Point *cl, struct Point *line)
{
	int i, j;
	int jnext;
	float p[3], dz, dr;
	float v1[3], v2[3];

	jnext = 1;
	for(i = 0; i < cl->nump; i++)				   // loop over blade points
	{
		for(j = jnext; j < ml->nump; j++)
		{
			// blade in meridian plane (phi = 0)
			v1[0] = ml->x[j-1]-cl->x[i];
			v1[1] = 0.0;
			v1[2] = ml->z[j-1]-cl->z[i];;
			v2[0] = ml->x[j]-cl->x[i];
			v2[1] = 0.0;
			v2[2] = ml->z[j]-cl->z[i];
			if( V_ScalarProduct(v1,v2) < 0.0)
			{
				jnext = MAX(j - 1,1);
				p[2] = cl->x[i];					  // radius
				p[0] = cl->y[i]-dphi;				  // circumf. angle
				// use appropriate value for interpolation, radius or height
				dz = fabs(2.0f*(ml->z[j-1]-ml->z[j]) / (ml->z[j-1]+ml->z[j]));
				dr = fabs(2.0f*(ml->x[j-1]-ml->x[j]) / (ml->x[j-1]+ml->x[j]));
				if( dz/dr <= 1.0)
				{
					p[1] = (cl->x[i] - ml->x[j-1])
						/ (ml->x[j] - ml->x[j-1])
						* (len[j] - len[j-1])
						+ len[j-1];
				}
				else if( ABS((cl->z[i] - ml->z[j-1])
							 / ml->z[j-1]) <= SMALL)
				{
					if( dr <= SMALL)
					{
						p[1] = (cl->z[i] - ml->z[j-1]) /
							(ml->z[j] - ml->z[j-1])
							* (len[j] - len[j-1]) + len[j-1];
					}
					else
					{
						p[1] = (cl->x[i] - ml->x[j-1])
							/ (ml->x[j] - ml->x[j-1])
							* (len[j] - len[j-1])
							+ len[j-1];
					}
				}
				else
				{
					p[1] = (cl->z[i] - ml->z[j-1])
						/ (ml->z[j] - ml->z[j-1])
						* (len[j] - len[j-1])
						+ len[j-1];
				}
				AddVPoint(line, p);
				break;
			}										 // end if y[j-1] < y[i] < y[j]
			continue;
		}
	}
	return 0;
}


// **************************************************

int InterpolCurve(struct Point *ml, float *len, float dphi,
				  struct Point *cl, struct curve *c)
{
	int i, j;
	int jnext;
	float p[3],dz,dr;
	float v1[3], v2[3];

	jnext = 1;
	for(i = 0; i < cl->nump; i++)				   // loop over blade points
	{
		for(j = jnext; j < ml->nump; j++)
		{
			// center line
			v1[0] = ml->x[j-1] - cl->x[i];
			v1[1] = 0.0;
			v1[2] = ml->z[j-1] - cl->z[i];
			v2[0] = ml->x[j] - cl->x[i];
			v2[1] = 0.0;
			v2[2] = ml->z[j] - cl->z[i];
			if( V_ScalarProduct(v1,v2) < 0.0)
			{
				jnext = MAX(j - 2,1);
				p[0] = cl->y[i]-dphi;				  // circumf. angle
				p[2] = cl->x[i];					  // radius
				// use appropriate value for interpolation, radius or height
				dz = fabs(2.0f*(ml->z[j-1]-ml->z[j]) / (ml->z[j-1]+ml->z[j]));
				dr = fabs(2.0f*(ml->x[j-1]-ml->x[j]) / (ml->x[j-1]+ml->x[j]));
				if( dz/dr <= 1.0)
				{
					p[1] = (cl->x[i] - ml->x[j-1]) / (ml->x[j] - ml->x[j-1])
						* (len[j] - len[j-1]) + len[j-1];
				}
				else if( ABS((cl->z[i] - ml->z[j-1])
							 / ml->z[j-1]) <= SMALL)
				{
					if( dr <= SMALL)
					{
						p[1] = (cl->z[i] - ml->z[j-1]) /
							(ml->z[j] - ml->z[j-1])
							* (len[j] - len[j-1]) + len[j-1];
					}
					else
					{
						p[1] = (cl->x[i] - ml->x[j-1])
							/ (ml->x[j] - ml->x[j-1])
							* (len[j] - len[j-1])
							+ len[j-1];
					}
				}
				else
				{
					p[1] = (cl->z[i] - ml->z[j-1]) / (ml->z[j] - ml->z[j-1])
						* (len[j] - len[j-1]) + len[j-1];
				}
				AddCurvePoint(c, p[0], p[1], p[2], 0.0, 0.0);
				break;
			}										 // end if y[j-1] < y[i] < y[j]
			continue;
		}											// end j
	}											   // end i
	CalcCurveArclen2(c);
	return 0;
}


// **************************************************

int InterpolBlade(float *srcpara, struct Point *src, struct Flist *tgtpara,
				  struct Point *tgt, float dphi)
{

	int i,j;
	int next;
	float delta;
	float p[3];

	next = 1;
	for(i = 0; i < tgtpara->num; i++)
	{
		for(j = next; j < src->nump; j++)
		{
			if( (srcpara[j-1] <= tgtpara->list[i])&&
				(srcpara[j] >= tgtpara->list[i]) )
			{
				next = j-1;
				delta = (tgtpara->list[i] - srcpara[j-1]) /
					(srcpara[j] - srcpara[j-1]);
				p[0] = delta * (src->x[j]+dphi)*src->z[j]
					+(1-delta) * (src->x[j-1]+dphi)*src->z[j-1];
				p[1] = delta * (src->y[j] - src->y[j-1]) + src->y[j-1];
				p[2] = delta * (src->z[j] - src->z[j-1]) + src->z[j-1];
				AddVPoint(tgt,p);
				break;
			}
			continue;
		}											// end j
	}											   // end i
	return 0;
}


int InterpolBladeSpline(struct Point *src, struct Flist *tgtpara,
						struct Point *tgt, float *n, float dphi)
{

	int i;
	float t1, t2;
	float p[3], p1[3], p2[3], p3[3], u[3], v[3];

	struct Point *poly = NULL;
	struct Flist *knot = NULL;

	t1 = 0.5; t2 = 0.5;
	p[2] = p1[2] = p2[2]= p3[2] = u[2] = v[2] = 0.0;
	// **************************************************
	// get spline polygon
	p1[0] = (src->x[0]+dphi)*src->z[0];
	p1[1] = src->y[0];
	//
	p3[0] = (src->x[1]+dphi)*src->z[1];
	p3[1] = src->y[1];
	u[0]  = src->x[2]*src->z[2] - src->x[1]*src->z[1];
	u[1]  = src->y[2] - src->y[1];
	LineIntersect(p1,n, p3,u, p2);
	// check if polygon point makes geometrical sense
	u[0] =	p3[0] - p1[0];
	u[1] =	p3[1] - p1[1];
	v[0] =	n[1];
	v[1] = -n[0];
	if(acos(V_Angle(u,v)) > M_PI/2)
	{
		poly = AllocPointStruct();
		AddVPoint(poly, p1);
		p[0] = p1[0] + t1*(p3[0] - p1[0]);
		p[1] = p1[1] + t1*(p3[1] - p1[1]);
		AddVPoint(poly, p);
#ifdef DEBUG_REGIONS
		fprintf(stderr,"InterpolBladeSpline: acos(V_Angle(u,v)) = %f\n",
				acos(V_Angle(u,v))*180/M_PI);
#endif
	}
	else
	{
		poly = CurvePolygon(p1,p2,p3, t1, t2);
	}
	for(i = 2; i < src->nump; i++)
	{
		p[0] = (src->x[i]+dphi)*src->z[i];
		p[1] = src->y[i];
		AddVPoint(poly, p);
	}
	knot = BSplineKnot(poly, BSPLN_DEGREE);

	// get new nodes
	for(i = 0; i < tgtpara->num; i++)
	{
		BSplinePoint(BSPLN_DEGREE, poly, knot, tgtpara->list[i], p);
		AddVPoint(tgt,p);
	}											   // end i
	FreePointStruct(poly);
	FreeFlistStruct(knot);

	return 0;
}


// **************************************************

struct region **GetRegionsMemory(struct region **reg, int rnum)
{
	int i, j;

	for(i = 0; i < rnum; i++)
	{
		if(!reg[i]) continue;
		for(j = 0; j < reg[i]->numl; j++)
		{
			if(reg[i]->arc[j])
			{
				FreeFlistStruct(reg[i]->arc[j]);
				reg[i]->arc[j] = NULL;
			}
			if(reg[i]->para[j])
			{
				FreeFlistStruct(reg[i]->para[j]);
				reg[i]->para[j] = NULL;
			}
			reg[i]->line[j] = GetPointMemory(reg[i]->line[j]);
		}
	}
	return reg;
}


// **************************************************

int GetPointRadius(struct Point *line, struct curve *ml)
{
	int i, j, jnext;

	for (i = 0; i < line->nump; i++)
	{
		jnext = 1;
		for (j = jnext; j < ml->p->nump; j++)
		{
			if ((ml->len[j-1] <= line->y[i]) &&
				(ml->len[j] >= line->y[i]))
			{
				jnext = j - 1;
				line->z[i] = (ml->p->x[j-1] - ml->p->x[j]) / (ml->len[j-1] - ml->len[j])
					* (line->y[i] - ml->len[j]) + ml->p->x[j];
				break;
			}
			continue;
		}
		if(ABS(line->z[i]) <= TINY)
		{
			fprintf(stderr," GetPointRadius: WARNING: point with zero radius, might cause further difficulties!\n");
		}
	}
	return 0;
}


// **************************************************

int CalcPointCoords(struct Point *line, struct Flist **arc, struct curve *ml)
{
	int i;
	struct Flist *parc;
	int GetPointRadius(struct Point *line, struct curve *ml);

	GetPointRadius(line, ml);
	parc = AllocFlistStruct(line->nump+1);
	for(i = 0; i < line->nump; i++)
	{
		Add2Flist(parc,line->x[i]);
		line->x[i] /= (line->z[i]);
	}

	*arc = &parc[0];

	return 0;
}


int CalcLinearCurve(struct Point *line, struct Flist *para, float *u, float *p1)
{
	int i;
	float p[3];

	for(i = 0; i < para->num; i++)
	{
		p[0] = u[0]*para->list[i] + p1[0];
		p[1] = u[1]*para->list[i] + p1[1];
		p[2] = u[2]*para->list[i] + p1[2];
#ifdef DEBUG_REGIONS
		fprintf(stderr," CalcLinearCurve()\n");
		VPRINT(u);
		VPRINT(p1);
		fprintf(stderr,"   para = %f\n",para->list[i]);
#endif
		AddVPoint(line, p);
	}
	return 0;
}


int XShiftCurve(struct Point *srcline, struct Point *destline, float shift)
{
	int i;
	float p[3];

	for(i = 0; i < srcline->nump; i++)
	{
		p[0] = srcline->x[i] + shift;
		p[1] = srcline->y[i];
		p[2] = srcline->z[i];
		AddVPoint(destline, p);
	}
	return 0;
}


int	  CalcEnvelopeCurve(struct Point *envline, struct Flist *learc, struct Point *leline,
						struct Flist *blarc, struct Point *blline, struct Point *cl,
						struct Flist *para, float dphi, float lscale, int sign)
{
	int i, zero_thick;
	float p[3], v1[3], v2[3];
	float u1[3], u2[3];
	float alpha, beta;
	float len, delta, scale, lend;

	u1[2] = u2[2] = v1[2] = v2[2] = p[2] = 0.0;
	zero_thick = 0;

	u1[0] = blarc->list[blline->nump-1] - (cl->x[cl->nump-1]+dphi)*cl->z[cl->nump-1];
	u1[1] = blline->y[blline->nump-1] - cl->y[cl->nump-1];
	if(V_Len(u1) < SMALL) zero_thick = 1;
	u2[0] = blarc->list[blline->nump-1] - blarc->list[blline->nump-2];
	u2[1] = blline->y[blline->nump-1] - blline->y[blline->nump-2];
	if(zero_thick) beta = (float)M_PI/2.0f;
	else beta  = float(acos(V_Angle(u1, u2)));
	u1[0] = learc->list[0] - learc->list[leline->nump-1];
	u1[1] = leline->y[0] - leline->y[leline->nump-1];
	len = V_Len(u1);
	u2[0] = blarc->list[1] - blarc->list[0];
	u2[1] = blline->y[1] - blline->y[0];
	alpha = float(acos(V_Angle(u1, u2)));
	if(alpha > M_PI)
	{
#ifdef DEBUG_REGIONS
		fprintf(stderr,"CalcEnvelopeCurve: alpha = %f\n",
				alpha*180/M_PI);
#endif
		alpha = (alpha + beta) * 0.5f;
	}
	AddPoint(envline,learc->list[0],
			 leline->y[0], 0.0);
	lend = lscale * len;
	for(i = 1; i < blline->nump-1; i++)
	{
		u2[0] = ( blarc->list[i+1] + blarc->list[i] )*0.5f
			- blarc->list[i-1] ;
		u2[1] = ( blline->y[i+1] + blline->y[i] )*0.5f
			- blline->y[i-1] ;
		delta = sign*(alpha + para->list[i]*(beta-alpha));
		u1[0] = float(cos(delta)*u2[0] - sin(delta)*u2[1]);
		u1[1] = float(sin(delta)*u2[0] + cos(delta)*u2[1]);
		V_Norm(u1);
		scale = (1.0f - para->list[i]) + para->list[i] * lscale;
		p[0] = u1[0]*len*scale + blarc->list[i];
		p[1] = u1[1]*len*scale + blline->y[i];
		AddVPoint(envline, p);
	}											   // end i, blline->nump
	if(zero_thick)
	{
		u1[0] =	 sign*(blline->y[blline->nump-2] - blline->y[blline->nump-1]);
		u1[1] = -sign*(blarc->list[blline->nump-2] - blarc->list[blline->nump-1]);
	}
	else
	{
		u1[0] = ( blarc->list[blline->nump-1] - (cl->x[cl->nump-1]+dphi)*cl->z[cl->nump-1] );
		u1[1] = ( blline->y[blline->nump-1] - cl->y[cl->nump-1] );
	}
	V_Norm(u1);
	AddPoint(envline,u1[0]*len*lscale+blarc->list[blline->nump-1],
			 u1[1]*len*lscale+blline->y[blline->nump-1], 0.0);
#ifdef DEBUG_REGIONS
	VPRINT(u1);
	fprintf(stderr," len, lscale, blarc->list[blline->nump-1]: %f, %f, %f\n",
			len, lscale, blarc->list[blline->nump-1]);
	fprintf(stderr," envline->x[envline->nump-1] = %f\ndone!\n\n",envline->x[envline->nump-1]);
#endif
	return 0;
}


int	  CalcEnvelopeCurveSpline(struct Point *envline, struct Flist *learc, struct Point *leline,
							  struct Flist *blarc, struct Point *blline, struct Point *cl,
							  struct Flist *para, float dphi, float lscale, int sign)
{
	int i;
	float p[3];

	struct Point *poly = NULL;
	struct Flist *knot = NULL;

	poly = AllocPointStruct();
	CalcEnvelopeCurve(poly, learc, leline, blarc, blline, cl, para, dphi, lscale, sign);
	knot = BSplineKnot(poly, BSPLN_DEGREE);
	// get new nodes
	p[2] = 0.0;
	for(i = 0; i < para->num; i++)
	{
		BSplinePoint(BSPLN_DEGREE, poly, knot, para->list[i], p);
		AddVPoint(envline,p);
#ifdef DEBUG_REGIONS
		fprintf(stderr,"CalcEnvelopeCurveSpline\n");
		fprintf(stderr,"i: %d: p = [ %f	  %f], para = %f\n",
				i, p[0], p[1], para->list[i]);
#endif
	}											   // end i
	FreePointStruct(poly);
	FreeFlistStruct(knot);

	return 0;
}


int	  CalcEnvelopeCurve2(struct Point *envline, struct Flist *learc, struct Point *leline,
						 struct Flist *blarc, struct Point *blline, struct Point *cl,
						 struct Flist *para, int le_dis, float dphi, float lscale, int sign)
{
	int i, zero_thick;
	float p[3], v1[3], v2[3];
	float u1[3], u2[3];
	float alpha, beta, gamma;
	float *pp, pstart;
	float len, delta, scale, lend, lenv;

	u1[2] = u2[2] = v1[2] = v2[2] = p[2] = 0.0;
	zero_thick = 0;

	u1[0] = blarc->list[blline->nump-1] - (cl->x[cl->nump-1]+dphi)*cl->z[cl->nump-1];
	u1[1] = blline->y[blline->nump-1] - cl->y[cl->nump-1];
	if(V_Len(u1) < SMALL) zero_thick = 1;
	u2[0] = blarc->list[blline->nump-1] - blarc->list[blline->nump-2];
	u2[1] = blline->y[blline->nump-1] - blline->y[blline->nump-2];
	if(zero_thick) beta = (float) M_PI/2.0f;
	else beta  = float(acos(V_Angle(u1, u2)));
	u1[0] = learc->list[0] - learc->list[leline->nump-1];
	u1[1] = leline->y[0] - leline->y[leline->nump-1];
	len = V_Len(u1);
	u2[0] = blarc->list[1] - blarc->list[0];
	u2[1] = blline->y[1] - blline->y[0];
	alpha = float(acos(V_Angle(u1, u2)));
	if(alpha > M_PI)
	{
#ifdef DEBUG_REGIONS
		fprintf(stderr,"CalcEnvelopeCurve: alpha = %f\n",
				alpha*180/M_PI);
#endif
		alpha = (alpha + beta) * 0.5f;
	}
	AddPoint(envline,learc->list[0],
			 leline->y[0], 0.0);
	lend = lscale * len;
	// angle at le_dis-point
	gamma = (float) M_PI/2.0f;
	lenv  = 0.0f;
	// modify parameters to get 90 degree angle at certain point
	if( (pp = (float*)calloc(para->num,sizeof(float))) == NULL)
	{
		fatal("mem. for float!");
		exit(-1);
	}
	scale = para->list[le_dis-1];
	for(i = 0; i < le_dis; i++)
	{
		pp[i] = para->list[i]/scale;
	}
	pstart = para->list[le_dis-1];
	delta  = para->list[para->num-1] - pstart;
	for(i = le_dis; i < blline->nump; i++)
	{
		pp[i] = para->list[i] - pstart * ((para->list[i]-pstart)/delta);
	}
	// calc points
	for(i = 1; i < blline->nump-1; i++)
	{
		u2[0] = ( blarc->list[i+1] + blarc->list[i] )*0.5f
			- blarc->list[i-1] ;
		u2[1] = ( blline->y[i+1] + blline->y[i] )*0.5f
			- blline->y[i-1] ;
		if(i < le_dis)
		{
			delta = sign*(alpha + pp[i]*(gamma-alpha));
		}
		else
		{
			delta = sign*(gamma + pp[i]*(beta-gamma));
		}
#ifdef DEBUG_REGIONS
		fprintf(stderr,"pp[%03d] = %8.6f, delta = %10.5f\n",i,
				pp[i],delta*180/M_PI);
#endif
		u1[0] = float(cos(delta)*u2[0] - sin(delta)*u2[1]);
		u1[1] = float(sin(delta)*u2[0] + cos(delta)*u2[1]);
		V_Norm(u1);
		scale = (1.0 - para->list[i]) + para->list[i] * lscale;
		p[0] = u1[0]*len*scale + blarc->list[i];
		p[1] = u1[1]*len*scale + blline->y[i];
		AddVPoint(envline, p);
		lenv += float(sqrt(pow(p[0] - envline->x[envline->nump-2],2) +
					 pow(p[1] - envline->y[envline->nump-2],2)));
	}											   // end i, blline->nump
	if(zero_thick)
	{
		u1[0] =	 sign*(blline->y[blline->nump-2] - blline->y[blline->nump-1]);
		u1[1] = -sign*(blarc->list[blline->nump-2] - blarc->list[blline->nump-1]);
	}
	else
	{
		u1[0] = ( blarc->list[blline->nump-1] - (cl->x[cl->nump-1]+dphi)*cl->z[cl->nump-1] );
		u1[1] = ( blline->y[blline->nump-1] - cl->y[cl->nump-1] );
	}
	V_Norm(u1);
	AddPoint(envline,u1[0]*len*lscale+blarc->list[blline->nump-1],
			 u1[1]*len*lscale+blline->y[blline->nump-1], 0.0);
	// recalc. parameters
	for(i = 1; i < para->num; i++)
	{
		len += float(sqrt(pow(envline->x[i-1] - envline->x[i],2) +
					pow(envline->y[i-1] - envline->y[i],2)));
		para->list[i] = len/lenv;
	}
	free(pp);
	return 0;
}


int	  CalcEnvelopeCurveSpline2(struct Point *envline, struct Flist *learc, struct Point *leline,
							   struct Flist *blarc, struct Point *blline, struct Point *cl,
							   struct Flist *para, int le_dis, float dphi, float lscale, int sign)
{
	int i, ix, dir_sign;
	float pp, ratio, pend, delta;
	float p[3], p1[3], p2[3], p3[3];
	float v1[3], v2[3], n[3];

	struct Point *poly = NULL;
	struct Flist *knot = NULL;

	p[2] = p1[2] = p2[2] = p3[2] = v1[2] = v2[2] = n[2] = 0.0;
	poly = AllocPointStruct();
	CalcEnvelopeCurve(poly, learc, leline, blarc, blline, cl, para, dphi, lscale, sign);
	knot = BSplineKnot(poly, BSPLN_DEGREE);

	// get new nodes
	for(i = 0; i < para->num; i++)
	{
		BSplinePoint(BSPLN_DEGREE, poly, knot, para->list[i], p);
		AddVPoint(envline,p);
	}											   // end i
	// get intersection point for right angle
	// between connection between env. and surface -> recalc. params.
	// vector, normal to blade surf. at point le_dis
	ix	  =	 le_dis-1;
#ifdef DEBUG_REGIONS
	fprintf(stderr,"ix = %d, le_dis-1 = %d\n", ix, le_dis-1);
#endif
	p[0]  =	 blarc->list[ix];
	p[1]  =	 blline->y[ix];
	n[0]  = -sign * (blline->y[le_dis]	 - p[1]);
	n[1]  =	 sign * (blarc->list[le_dis] - p[0]);
	p1[0] =	 envline->x[ix];
	p1[1] =	 envline->y[ix];
	v1[0] =	 envline->x[le_dis] - p1[0];
	v1[1] =	 envline->y[le_dis] - p1[1];
	LineIntersect(p,n, p1,v1, p2);
	v2[0] =	 p2[0] - p1[0];
	v2[1] =	 p2[1] - p1[1];
	dir_sign = SIGN(V_ScalarProduct(v1,v2));
	if(dir_sign > 0)
	{
		// p2 is at same side as x/y[le_dis]
		i = ix;
		while(dir_sign > 0)
		{
#ifdef DEBUG_REGIONS
			fprintf(stderr,"i = %3d, dir_sign = %2d\n",i,dir_sign);
#endif
			i++;									 // next point
			p1[0] =	 envline->x[i];
			p1[1] =	 envline->y[i];
			v1[0] =	 envline->x[i+1] - p1[0];
			v1[1] =	 envline->y[i+1] - p1[1];
			LineIntersect(p,n, p1,v1, p2);
			v2[0] =	 p2[0] - p1[0];
			v2[1] =	 p2[1] - p1[1];
			dir_sign = SIGN(V_ScalarProduct(v1,v2));
		}
		i--;										// jump back one point
		p1[0] =	 envline->x[i];
		p1[1] =	 envline->y[i];
		v1[0] =	 envline->x[i+1] - p1[0];
		v1[1] =	 envline->y[i+1] - p1[1];
		LineIntersect(p,n, p1,v1, p2);
		v2[0] =	 p2[0] - p1[0];
		v2[1] =	 p2[1] - p1[1];
	}
	else
	{
		i = ix;
		while(dir_sign < 0)
		{
#ifdef DEBUG_REGIONS
			fprintf(stderr,"i = %3d, dir_sign = %2d\n",i,dir_sign);
#endif
			i--;									 // prev. point
			p1[0] =	 envline->x[i];
			p1[1] =	 envline->y[i];
			v1[0] =	 envline->x[i+1] - p1[0];
			v1[1] =	 envline->y[i+1] - p1[1];
			LineIntersect(p,n, p1,v1, p2);
			v2[0] =	 p2[0] - p1[0];
			v2[1] =	 p2[1] - p1[1];
			dir_sign = SIGN(V_ScalarProduct(v1,v2));
		}
	}
	pp = para->list[i] + (V_Len(v2)/V_Len(v1))
		* (para->list[i+1]-para->list[i]);
	ratio = pp / para->list[le_dis-1];
	pend  = para->list[para->num-1];
	delta = pend - para->list[le_dis-1];
	pp	 -= para->list[le_dis-1];
	for(i = 0; i < le_dis; i++)
	{
		para->list[i] *= ratio;
	}
	for(i = le_dis; i < para->num; i++)
	{
		para->list[i] +=  pp * (pend - para->list[i])/delta;
	}
#ifdef DEBUG_REGIONS
	DumpFlist(para);
#endif
	// get new nodes with modified parameters
	envline->nump = 0;
	for(i = 0; i < para->num; i++)
	{
		BSplinePoint(BSPLN_DEGREE, poly, knot, para->list[i], p);
		AddVPoint(envline,p);
#ifdef DEBUG_REGIONS
		fprintf(stderr,"p: %f  %f  %f\n",p[0],p[1],p[2]);
#endif
	}											   // end i
	FreePointStruct(poly);
	FreeFlistStruct(knot);

	return 0;
}


struct region *AddMeshLines(struct region *reg, int start, int end, int first,
							int last, int initnuml, int addnuml)
{
	// start ... index of curve for spline to start at
	// end	 ... index of curve where spline ends
	// first ... index of first curve (ident./w 1st spline)
	// last	 ... index of last curve

	int i,j;
	int numl;

	float para, ratio;

#ifdef DEBUG_REGIONS
	char fn[111];
	FILE *fp;
	static int call = 0;

	sprintf(fn,"rr_debugaddmeshlines_%02d.txt",call++);
	if( (fp = fopen(fn,"w+")) == NULL)
	{
		fprintf(stderr, "Shit happened opening file '%s'!\n",fn);
		exit(-1);
	}
#endif

	// add memory, only at first time, reg->numl = 4 initialized
	if(reg->numl == initnuml)
	{
		numl = addnuml + reg->numl;
		if( (reg->line = (struct Point **)realloc(reg->line, numl*sizeof(struct Point *))) == NULL)
		{
			fatal("memory for realloc(struct Point *)!");
		}
		if( (reg->arc = (struct Flist **)realloc(reg->arc, numl*sizeof(struct Flist *))) == NULL)
		{
			fatal("memory for realloc(struct Flist *)!");
		}
		if( (reg->para = (struct Flist **)realloc(reg->para, numl*sizeof(struct Flist *))) == NULL)
		{
			fatal("memory for realloc(struct Flist *)!");
		}

		for(i = reg->numl; i < numl; i++)
		{
			reg->line[i] = GetPointMemory(reg->line[i]);
		}
		reg->numl = numl;
	}											   // end if reg->numl == initnuml

	if(reg->para[start]->num != reg->para[end]->num)
	{
		fatal("mismatch of parameter numbers!");
	}
	if(reg->para[first]->num != reg->para[last]->num)
	{
		fatal("mismatch of parameter numbers!");
	}

	// create addtl. spline curves between region boundaries
#ifdef DEBUG_REGIONS
	fprintf(fp,"new!\n");
	DumpFlist2File(reg->para[first], fp);
	DumpFlist2File(reg->para[last], fp);
#endif
	for(i = initnuml; i < reg->numl; i++)
	{
		reg->para[i] = AllocFlistStruct(reg->para[first]->num+1);
		reg->arc[i]	 = AllocFlistStruct(reg->arc[first]->num+1);
		ratio = (float)((float)(reg->numl - i-1)/(float)(reg->line[start]->nump-1));
#ifdef DEBUG_REGIONS
		fprintf(fp,"reg->line[start]->nump = %d\n",reg->line[start]->nump);
		fprintf(fp,"reg->numl = %d\n",reg->numl);
		fprintf(fp,"i		  = %d\n",i);
		fprintf(fp,"ratio	  = %f\n",ratio);
#endif
		for(j = 0; j < reg->para[first]->num; j++)
		{
			// interpolate parameter value
			para = ratio*reg->para[first]->list[j] + (1.0f-ratio)*reg->para[last]->list[j];
			Add2Flist(reg->para[i],para);
		}
#ifdef DEBUG_REGIONS
		DumpFlist2File(reg->para[i],fp);
#endif
	}
	return reg;
}


int CalcTERatio(struct region *reg, struct Flist *clarc,
				struct Point *cl, int iblade, int ienv, float dphi)
{
	int ratio;
	float len1, len2;
	float u1[3], u2[3];

	u1[0] = reg->arc[ienv]->list[reg->arc[ienv]->num-1]
		- reg->arc[iblade]->list[reg->arc[iblade]->num-1];
	u1[1] = reg->line[ienv]->y[reg->line[ienv]->nump-1]
		- reg->line[iblade]->y[reg->line[iblade]->nump-1];
	u1[2] = 0.0;
	u2[0] = reg->arc[iblade]->list[reg->arc[iblade]->num-1]
		- clarc->list[clarc->num-1] - dphi * cl->z[cl->nump-1];
	u2[1] = reg->line[iblade]->y[reg->line[iblade]->nump-1]
		- cl->y[cl->nump-1];
	u2[2] = 0.0;
	// trailing edge
	len1 = V_Len(u1);
	len2 = V_Len(u2);
	ratio = (int)( 2*(len2*(float)(reg->line[3]->nump-1)) / (2.0*len1) + 1);
	if(len2 < SMALL) return 0;
	if(ratio < 2) return 2;

	return ratio;
}


float CalcTEParameter(struct region *reg, struct Flist *clarc,
					  struct Point *cl, int iblade, int ienv, float dphi)
{
	float len1, len2;
	float u1[3], u2[3];

	u1[0] = reg->arc[ienv]->list[reg->arc[ienv]->num-1]
		- reg->arc[iblade]->list[reg->arc[iblade]->num-1];
	u1[1] = reg->line[ienv]->y[reg->line[ienv]->nump-1]
		- reg->line[iblade]->y[reg->line[iblade]->nump-1];
	u1[2] = 0.0;
	u2[0] = reg->arc[iblade]->list[reg->arc[iblade]->num-1]
		- (clarc->list[clarc->num-1] + dphi * cl->z[cl->nump-1]);
	u2[1] = reg->line[iblade]->y[reg->line[iblade]->nump-1]
		- cl->y[cl->nump-1];
	u2[2] = 0.0;
	// trailing edge
	len1 = V_Len(u1);
	len2 = V_Len(u2);

	if(len2 < SMALL) return 0.0;
	return (len2/(len1+len2));
}


#ifdef PARA_OUT
int PutRR_GridParams(struct rr_grid *grid)
{
	char *fn = (char *)"rr_gridparams.dat.new";
	FILE *fp;

	if( (fp = fopen(fn,"w+")) == NULL)
	{
		fprintf(stderr,"file '%s!\n",fn);
		exit(-1);
	}
	fprintf(fp,"#######################################\n");
	fprintf(fp,"#									  #\n");
	fprintf(fp,"# discretization  & domain parameters #\n");
	fprintf(fp,"#									  #\n");
	fprintf(fp,"#######################################\n");
	fprintf(fp,"#\n# as follows: number of nodes, bias, type\n#\n");
	fprintf(fp,"# discretization hub -> shroud, grid->ge_XXX\n");
	fprintf(fp,"  %3d, %6.2f, %3d\n",grid->ge_num, grid->ge_bias, grid->ge_type);
#ifdef GAP
	fprintf(fp,"#\n# discretization of tip clearance region, grid->gp_XXX\n");
	fprintf(fp,"  %3d, %6.2f, %3d\n",grid->gp_num, grid->gp_bias, grid->gp_type);
#endif
	fprintf(fp,"#\n# inlet extension, grid->extXXX\n");
	fprintf(fp,"  %3d, %6.2f, %3d\n",grid->extdis, grid->extbias, grid->extbias_type);
	fprintf(fp,"#\n# circumf. discretization, runner inlet, grid->cXXX\n");
	fprintf(fp,"  %3d, %6.2f, %3d\n",grid->cdis, grid->cbias, grid->cbias_type);
	fprintf(fp,"# circumf. discretization at leading edge, grid->cleXXX\n");
	fprintf(fp,"  %3d, %6.2f, %3d\n",grid->cledis, grid->clebias, grid->clebias_type);
	fprintf(fp,"# meridional, runner inlet region, grid->ssmXXX\n");
	fprintf(fp,"  %3d, %6.2f, %3d\n",grid->ssmdis, grid->ssmbias, grid->ssmbias_type);
	fprintf(fp,"# meridional, boundary layer, grid->pseXXX\n");
	fprintf(fp,"  %3d, %6.2f, %3d\n",grid->psedis, grid->psebias, grid->psebias_type);
	fprintf(fp,"# pressure side alongside surface, grid->psXXX\n");
	fprintf(fp,"  %3d, %6.2f, %3d\n",grid->psdis, grid->psbias, grid->psbias_type);
	fprintf(fp,"# suction side alongside surface, grid->ssXXX\n");
	fprintf(fp,"  %3d, %6.2f, %3d\n",grid->ssdis, grid->ssbias, grid->ssbias_type);
	fprintf(fp,"# duct, ps-te -> ss-surface, grid->midXXX\n");
	fprintf(fp,"  ---, %6.2f, %3d\n",grid->midbias, grid->midbias_type);
	fprintf(fp,"# outlet section, ss-te -> outlet, grid->lowXXX\n");
	fprintf(fp,"  %3d, %6.2f, %3d\n",grid->lowdis, grid->lowbias, grid->lowbias_type);
	fprintf(fp,"# outlet section, inner part (5.3), grid->lowinXXX\n");
	fprintf(fp,"  ---, %6.2f, %3d\n", grid->lowinbias, grid->lowin_type);
	fprintf(fp,"# outlet, ss, ps center, grid->ssx/psx/cxXXX\n");
	fprintf(fp,"  ---, %6.2f, %3d\n", grid->ssxbias, grid->ssxbias_type);
	fprintf(fp,"  ---, %6.2f, %3d\n", grid->psxbias, grid->psxbias_type);
	fprintf(fp,"  ---, %6.2f, %3d\n", grid->cxbias, grid->cxbias_type);

	fprintf(fp,"#\n# domain parameters\n#\n");
	fprintf(fp,"# phi scaling factors; inlet, outlet, grid->phi0/1_scale\n");
	fprintf(fp,"  %7.4f, %7.4f\n",grid->phi_scale[0], grid->phi_scale[1]);
	fprintf(fp,"# phi skew factors; hub, shroud, grid->phi_skew\n");
	fprintf(fp,"  %7.4f, %7.4f\n",grid->phi_skew[0], grid->phi_skew[1]);
	fprintf(fp,"# skew factor for inlet ext., grid->phi0_ext\n");
	fprintf(fp,"  %7.4f\n",grid->phi0_ext);
	fprintf(fp,"# 2nd comp. tang. vect. (ss->ps), grid->angle_ext[0/1]\n");
	fprintf(fp,"  %7.4f, %7.4f\n", grid->angle_ext[0], grid->angle_ext[1]);
	fprintf(fp,"# min y-comp. for ss-vector and scaling factor for 1.4\n");
	fprintf(fp,"  %7.4f, %7.4f\n",grid->v14_angle[0],grid->v14_angle[1]);
	fprintf(fp,"# partition ratio, bound. layer/1.4, only for modified\n");
	fprintf(fp,"  %7.4f, %7.4f\n",grid->bl_v14_part[0], grid->bl_v14_part[1]);
	fprintf(fp,"# (inverse) boundary layer thickness, grid->bl_scale[0/1]\n");
	fprintf(fp,"  %7.4f, %7.4f\n", grid->bl_scale[0], grid->bl_scale[1]);
	fprintf(fp,"# ss-partition, hub/shroud, grid->ss_part[0/1]\n");
	fprintf(fp,"  %7.4f, %7.4f\n", grid->ss_part[0], grid->ss_part[1]);
	fprintf(fp,"# ps-partition, hub/shroud, grid->ps_part[0/1]\n");
	fprintf(fp,"  %7.4f, %7.4f\n", grid->ps_part[0], grid->ps_part[1]);
	fprintf(fp,"# ratio (le/blade), hub/shroud, grid->ssle_part[0/1]\n");
	fprintf(fp,"  %7.4f, %7.4f\n", grid->ssle_part[0], grid->ssle_part[1]);
	fprintf(fp,"# ratio (le/blade), hub/shroud, grid->psle_part[0/1]\n");
	fprintf(fp,"  %7.4f, %7.4f\n", grid->psle_part[0], grid->psle_part[1]);
	fprintf(fp,"# outlet, te-fraction, grid->out_part[0]\n");
	fprintf(fp,"  %7.4f\n", grid->out_part[0]);
	fprintf(fp,"# mesh inlet extension switch\n");
	fprintf(fp,"  %d\n",grid->mesh_ext);
	fprintf(fp,"#\n# THAT's IT!\n");

	fclose(fp);
	return 0;
}
#endif											  // PARA_OUT

#ifdef PARA_IN
#define MAX_LEN 200
int GetRR_GridParams(struct rr_grid *grid)
{
	int i;

	int *dis, *bias_type;
	float *bias, *scale[2];

	char *input;
	char dummy[123];
	char fn[120];
	FILE *fp = NULL;

	dis		  = (int*)calloc(1,sizeof(int));
	bias_type = (int*)calloc(1,sizeof(int));
	bias	  = (float*)calloc(1,sizeof(float));
	scale[0]  = (float*)calloc(1,sizeof(float));
	scale[1]  = (float*)calloc(1,sizeof(float));
	input	  = (char*)calloc(MAX_LEN,sizeof(char));

#ifdef GRID_IN_DEFAULT
	sprintf(fn,"rr_gridparams.dat");
#else
	fprintf(stdout,"\n grid parameter input file:");fflush(stdout);
	fscanf(stdin,"%s",fn);
#endif
	fprintf(stdout,"\n Reading grid parameters from '%s' ... ",fn);fflush(stdout);
	if( (fp = fopen(fn,"r")) == NULL)
	{
		fprintf(stderr,"\n\n could not open file '%s!\n",fn);
		return INPUT_FILE_ERROR;
	}
	memset(input,0,MAX_LEN);
	for(i = 0; i < 9; i++)
	{
		fgets(input, MAX_LEN-1,fp);
	}
	fscanf(fp,"	 %d, %f, %d\n",dis, bias, bias_type);
	grid->ge_num  = *dis;
	grid->ge_bias = *bias;
	grid->ge_type = *bias_type;
#ifdef GAP
	fscanf(fp,"	 %d, %f, %d\n",dis, bias, bias_type);
	grid->gp_num  = *dis;
	grid->gp_bias = *bias;
	grid->gp_type = *bias_type;
#endif
	fgets(input, MAX_LEN-1,fp);
	fgets(input, MAX_LEN-1,fp);
	fscanf(fp,"	 %d, %f, %d\n",dis, bias, bias_type);
	grid->extdis	   = *dis;
	grid->extbias	   = *bias;
	grid->extbias_type = *bias_type;
	fgets(input, MAX_LEN-1,fp);
	fgets(input, MAX_LEN-1,fp);
	fscanf(fp,"	 %d, %f, %d\n",dis, bias, bias_type);
	grid->cdis		 = *dis;
	grid->cbias		 = *bias;
	grid->cbias_type = *bias_type;

	fgets(input, MAX_LEN-1,fp);
	fscanf(fp,"	 %s  %f, %d\n",dummy, bias, bias_type);
	grid->cledis       = atoi(dummy);
	grid->clebias	   = *bias;
	grid->clebias_type = *bias_type;
	fgets(input, MAX_LEN-1,fp);
	fscanf(fp,"	 %d, %f, %d\n",dis, bias, bias_type);
	grid->ssmdis	   = *dis;
	grid->ssmbias	   = *bias;
	grid->ssmbias_type = *bias_type;

	fgets(input, MAX_LEN-1,fp);
	fscanf(fp,"	 %d, %f, %d\n",dis, bias, bias_type);
	grid->psedis	   = *dis;
	grid->psebias	   = *bias;
	grid->psebias_type = *bias_type;

	fgets(input, MAX_LEN-1,fp);
	fscanf(fp,"	 %d, %f, %d\n",dis, bias, bias_type);
	grid->psdis		  = *dis;
	grid->psbias	  = *bias;
	grid->psbias_type = *bias_type;

	fgets(input, MAX_LEN-1,fp);
	fscanf(fp,"	 %d, %f, %d\n",dis, bias, bias_type);
	grid->ssdis		  = *dis;
	grid->ssbias	  = *bias;
	grid->ssbias_type = *bias_type;

	fgets(input, MAX_LEN-1,fp);
	fscanf(fp,"	 %*s %f, %d\n",bias, bias_type);
	grid->midbias	   = *bias;
	grid->midbias_type = *bias_type;

	fgets(input, MAX_LEN-1,fp);
	fscanf(fp,"	 %d, %f, %d\n",dis, bias, bias_type);
	grid->lowdis	   = *dis;
	grid->lowbias	   = *bias;
	grid->lowbias_type = *bias_type;

	fgets(input, MAX_LEN-1,fp);
	fscanf(fp,"	 %*s %f, %d\n",bias, bias_type);
	grid->lowinbias		 = *bias;
	grid->lowin_type	 = *bias_type;

	fgets(input, MAX_LEN-1,fp);
	fscanf(fp,"	 %*s %f, %d\n",bias, bias_type);
	grid->ssxbias	   = *bias;
	grid->ssxbias_type = *bias_type;
	fscanf(fp,"	 %*s %f, %d\n",bias, bias_type);
	grid->psxbias	   = *bias;
	grid->psxbias_type = *bias_type;
	fscanf(fp,"	 %*s %f, %d\n",bias, bias_type);
	grid->cxbias	  = *bias;
	grid->cxbias_type = *bias_type;

	fgets(input, MAX_LEN-1,fp);
	fgets(input, MAX_LEN-1,fp);
	fgets(input, MAX_LEN-1,fp);
	fgets(input, MAX_LEN-1,fp);
	fscanf(fp,"	 %f, %f\n",scale[0], scale[1]);
	grid->phi_scale[0] = *scale[0];
	grid->phi_scale[1] = *scale[1];
	fgets(input, MAX_LEN-1,fp);
	fscanf(fp,"	 %f, %f\n",scale[0], scale[1]);
	grid->phi_skew[0] = *scale[0];
	grid->phi_skew[1] = *scale[1];
	fgets(input, MAX_LEN-1,fp);
	fscanf(fp,"	 %f\n",scale[0]);
	grid->phi0_ext = *scale[0];
	fgets(input, MAX_LEN-1,fp);
	fscanf(fp,"	 %f, %f\n", scale[0], scale[1]);
	grid->angle_ext[0] = *scale[0];
	grid->angle_ext[1] = *scale[1];
	fgets(input, MAX_LEN-1,fp);
	fscanf(fp,"	 %f, %f\n",&grid->v14_angle[0], &grid->v14_angle[1]);
	fgets(input, MAX_LEN-1,fp);
	fscanf(fp,"	 %f, %f\n",&grid->bl_v14_part[0], &grid->bl_v14_part[1]);
	fgets(input, MAX_LEN-1,fp);
	fscanf(fp,"	 %f, %f\n", scale[0], scale[1]);
	grid->bl_scale[0] = *scale[0];
	grid->bl_scale[1] = *scale[1];
	fgets(input, MAX_LEN-1,fp);
	fscanf(fp,"	 %f, %f\n", scale[0], scale[1]);
	grid->ss_part[0] = *scale[0];
	grid->ss_part[1] = *scale[1];
	fgets(input, MAX_LEN-1,fp);
	fscanf(fp,"	 %f, %f\n", scale[0], scale[1]);
	grid->ps_part[0] = *scale[0];
	grid->ps_part[1] = *scale[1];
	fgets(input, MAX_LEN-1,fp);
	fscanf(fp,"	 %f, %f\n", &grid->ssle_part[0], &grid->ssle_part[1]);
	fgets(input, MAX_LEN-1,fp);
	fscanf(fp,"	 %f, %f\n", &grid->psle_part[0], &grid->psle_part[1]);
	fgets(input, MAX_LEN-1,fp);
	fscanf(fp,"	 %f\n",scale[0]);
	grid->out_part[0] = *scale[0];
	fgets(input, MAX_LEN-1,fp);
	fscanf(fp,"%d\n",&grid->mesh_ext);
	// not finished yet!!

	fclose(fp);
	fprintf(stdout,"done!\n\n");
	return 0;
}
#endif											  // PARA_IN

#ifdef DEBUG_REGIONS
int DumpCGridElement(struct cgrid *cge, struct ge *ge, float dphi,
					 float phi1)
{
	int i, count = 0;
	static int cnum = 0;

	struct region *reg = NULL;

	char fn[111];
	FILE *fp;

	sprintf(fn,"rr_gridreg_%02d.txt", cnum++);
	if( (fp = fopen(fn,"w+")) == NULL)
	{
		fprintf(stderr, "Shit happened opening file '%s'!\n",fn);
		exit(-1);
	}
	// inlet region
	reg = cge->reg[0];
	fprintf(fp,"# inlet edge, cdis = %d (%3d)\n", reg->line[0]->nump, count++);
	for(i = 0; i < reg->line[0]->nump; i++)
	{
		fprintf(fp, " %8.6f	 %8.6f	%8.6f  %8.6f\n",
				reg->line[0]->x[i], reg->line[0]->y[i], reg->line[0]->z[i], reg->arc[0]->list[i]);
	}
	fprintf(fp,"\n\n# left line, inlet, ssdis_in = %d (%3d)\n",
			reg->line[1]->nump, count++);
	for(i = 0; i < reg->line[1]->nump; i++)
	{
		fprintf(fp, " %8.6f	 %8.6f	%8.6f  %8.6f\n",
				reg->line[1]->x[i], reg->line[1]->y[i], reg->line[1]->z[i], reg->arc[1]->list[i]);
	}
	fprintf(fp,"\n\n# right line, inlet, ssdis_in = %d (%3d)\n",
			reg->line[2]->nump, count++);
	for(i = 0; i < reg->line[2]->nump; i++)
	{
		fprintf(fp, " %8.6f	 %8.6f	%8.6f  %8.6f\n",
				reg->line[2]->x[i], reg->line[2]->y[i], reg->line[2]->z[i], reg->arc[2]->list[i]);
	}
	fprintf(fp,"\n\n# line 1.4 (%3d)\n", count++);
	for(i = 0; i < reg->line[3]->nump; i++)
	{
		fprintf(fp, " %8.6f	 %8.6f	%8.6f  %8.6f\n",
				reg->line[3]->x[i], reg->line[3]->y[i], reg->line[3]->z[i], reg->arc[3]->list[i]);
	}
	fprintf(fp,"\n\n# left center line (%3d)\n", count++);
	for(i = 0; i < ge->cl->nump; i++)
	{
		fprintf(fp, "%8.6f	%8.6f  %8.6f  %8.6f\n",
				cge->cl->x[i], cge->cl->y[i], cge->cl->z[i], cge->clarc->list[i]);
	}
	fprintf(fp,"\n\n# left pressure side (%3d)\n", count++);
	for(i = 0; i < ge->ps->nump; i++)
	{
		fprintf(fp, "%8.6f	%8.6f  %8.6f  %8.6f\n",
				cge->ps->p->x[i], cge->ps->p->y[i], cge->ps->p->z[i],
				cge->ps->p->x[i]*cge->ps->p->z[i]);
	}
	fprintf(fp,"\n\n# right center line (%3d)\n", count++);
	for(i = 0; i < ge->cl->nump; i++)
	{
		fprintf(fp, "%8.6f	%8.6f  %8.6f %8.6f\n",
				(cge->cl->x[i]+dphi), cge->cl->y[i], cge->cl->z[i],
				(cge->cl->x[i]+dphi)*cge->cl->z[i]);
	}
	fprintf(fp,"\n\n# left suction side (%3d)\n", count++);
	for(i = 0; i < ge->ss->nump; i++)
	{
		fprintf(fp, "%8.6f	%8.6f  %8.6f %8.6f\n",
				(cge->ss->p->x[i]), cge->ss->p->y[i], cge->ss->p->z[i],
				(cge->ss->p->x[i])*cge->ss->p->z[i]);
	}
	fprintf(fp,"\n\n# right suction side (%3d)\n", count++);
	for(i = 0; i < ge->ss->nump; i++)
	{
		fprintf(fp, "%8.6f	%8.6f  %8.6f %8.6f\n",
				(cge->ss->p->x[i]+dphi), cge->ss->p->y[i], cge->ss->p->z[i],
				(cge->ss->p->x[i]+dphi)*cge->ss->p->z[i]);
	}

	// ps-envelope region
	reg = cge->reg[3];
	fprintf(fp,"\n\n");
	for(i = 0; i < reg->line[0]->nump; i++)
	{
		fprintf(fp, "%8.6f	%8.6f  %8.6f  %8.6f\n",
				reg->line[0]->x[i], reg->line[0]->y[i], reg->line[0]->z[i], reg->arc[0]->list[i]);
	}
	fprintf(fp,"\n\n# ps envelope\n");
	for(i = 0; i < reg->line[1]->nump; i++)
	{
		fprintf(fp, "%8.6f	%8.6f  %8.6f  %8.6f\n",
				reg->line[1]->x[i], reg->line[1]->y[i], reg->line[1]->z[i], reg->arc[1]->list[i]);
	}
	fprintf(fp,"\n\n# ps blade surface\n");
	for(i = 0; i < reg->line[2]->nump; i++)
	{
		fprintf(fp, "%8.6f	%8.6f  %8.6f  %8.6f\n",
				reg->line[2]->x[i], reg->line[2]->y[i], reg->line[2]->z[i], reg->arc[2]->list[i]);
	}
	fprintf(fp,"\n\n# ps trailing edge\n");
	for(i = 0; i < reg->line[3]->nump; i++)
	{
		fprintf(fp, "%8.6f	%8.6f  %8.6f  %8.6f\n",
				reg->line[3]->x[i], reg->line[3]->y[i], reg->line[3]->z[i], reg->arc[3]->list[i]);
	}

	// ss-envelope region
	reg = cge->reg[1];
	fprintf(fp,"\n\n# ss-envelope region\n");
	for(i = 0; i < reg->line[0]->nump; i++)
	{
		fprintf(fp, "%8.6f	%8.6f  %8.6f  %8.6f\n",
				reg->line[0]->x[i], reg->line[0]->y[i], reg->line[0]->z[i], reg->arc[0]->list[i]);
	}
	fprintf(fp,"\n\n# ss blade surface\n");
	for(i = 0; i < reg->line[1]->nump; i++)
	{
		fprintf(fp, "%8.6f	%8.6f  %8.6f  %8.6f\n",
				reg->line[1]->x[i], reg->line[1]->y[i], reg->line[1]->z[i], reg->arc[1]->list[i]);
	}
	fprintf(fp,"\n\n# ss envelope, %d\n", reg->line[2]->nump);
	for(i = 0; i < reg->line[2]->nump; i++)
	{
		fprintf(fp, "%8.6f	%8.6f  %8.6f  %8.6f\n",
				reg->line[2]->x[i], reg->line[2]->y[i], reg->line[2]->z[i], reg->arc[2]->list[i]);
	}
	fprintf(fp,"\n\n# ss te-ext.-envelope\n");
	for(i = 0; i < reg->line[3]->nump; i++)
	{
		fprintf(fp, "%8.6f	%8.6f  %8.6f  %8.6f\n",
				reg->line[3]->x[i], reg->line[3]->y[i], reg->line[3]->z[i], reg->arc[3]->list[i]);
	}
	// upper core region
	reg = cge->reg[2];
	fprintf(fp,"\n\n# upper core region, no. 3\n");
	fprintf(fp,"\n\n# le-line, 3.1\n");
	for(i = 0; i < reg->line[0]->nump; i++)
	{
		fprintf(fp, "%8.6f	%8.6f  %8.6f  %8.6f\n",
				reg->line[0]->x[i], reg->line[0]->y[i], reg->line[0]->z[i], reg->arc[0]->list[i]);
	}
	fprintf(fp,"\n\n# upper core ss envelope, 3.2\n");
	for(i = 0; i < reg->line[1]->nump; i++)
	{
		fprintf(fp, "%8.6f	%8.6f  %8.6f  %8.6f\n",
				reg->line[1]->x[i], reg->line[1]->y[i], reg->line[1]->z[i], reg->arc[1]->list[i]);
	}
	fprintf(fp,"\n\n# upper core ps envelope, 3.3\n");
	for(i = 0; i < reg->line[2]->nump; i++)
	{
		fprintf(fp, "%8.6f	%8.6f  %8.6f  %8.6f\n",
				reg->line[2]->x[i], reg->line[2]->y[i], reg->line[2]->z[i], reg->arc[2]->list[i]);
	}
	fprintf(fp,"\n\n# core outlet line, 3.4\n");
	for(i = 0; i < reg->line[3]->nump; i++)
	{
		fprintf(fp, "%8.6f	%8.6f  %8.6f  %8.6f\n",
				reg->line[3]->x[i], reg->line[3]->y[i], reg->line[3]->z[i], reg->arc[3]->list[i]);
	}
	// trailing edge extension, (ss) no. 5
	reg = cge->reg[4];
	fprintf(fp,"\n\n# trailing edge extension, 5.1\n");
	for(i = 0; i < reg->line[0]->nump; i++)
	{
		fprintf(fp, "%8.6f	%8.6f  %8.6f  %8.6f\n",
				reg->line[0]->x[i], reg->line[0]->y[i], reg->line[0]->z[i], reg->arc[0]->list[i]);
	}
	fprintf(fp,"\n\n");
	fprintf(fp, "%8.6f	%8.6f  %8.6f  %8.6f\n",
			reg->line[0]->x[0], reg->line[0]->y[0], reg->line[0]->z[0], reg->arc[0]->list[0]);
	fprintf(fp, "%8.6f	%8.6f  %8.6f  %8.6f\n",
			phi1, ge->ml->len[ge->ml->p->nump-1], 0.0, phi1*ge->ml->p->x[ge->ml->p->nump-1]);

	fprintf(fp,"\n\n# trailing edge center line extension, 5.2\n");
	for(i = 0; i < reg->line[1]->nump; i++)
	{
		fprintf(fp, "%8.6f	%8.6f  %8.6f  %8.6f\n",
				reg->line[1]->x[i], reg->line[1]->y[i], reg->line[1]->z[i], reg->arc[1]->list[i]);
	}
	fprintf(fp,"\n\n# trailing edge envelope curve extension, 5.3\n");
	for(i = 0; i < reg->line[2]->nump; i++)
	{
		fprintf(fp, "%8.6f	%8.6f  %8.6f  %8.6f\n",
				reg->line[2]->x[i], reg->line[2]->y[i], reg->line[2]->z[i], reg->arc[2]->list[i]);
	}
	fprintf(fp,"\n\n# te extension exit curve, 5.4\n");
	for(i = 0; i < reg->line[3]->nump; i++)
	{
		fprintf(fp, "%8.6f	%8.6f  %8.6f  %8.6f\n",
				reg->line[3]->x[i], reg->line[3]->y[i], reg->line[3]->z[i], reg->arc[3]->list[i]);
	}
	// ps trailing edge, no. 7
	reg = cge->reg[6];
	fprintf(fp,"\n\n# trailing edge extension, 7.1\n");
	for(i = 0; i < reg->line[0]->nump; i++)
	{
		fprintf(fp, "%8.6f	%8.6f  %8.6f  %8.6f\n",
				reg->line[0]->x[i], reg->line[0]->y[i], reg->line[0]->z[i], reg->arc[0]->list[i]);
	}
	fprintf(fp,"\n\n# right envelope extension, 7.2\n");
	for(i = 0; i < reg->line[1]->nump; i++)
	{
		fprintf(fp, "%8.6f	%8.6f  %8.6f  %8.6f\n",
				reg->line[1]->x[i], reg->line[1]->y[i], reg->line[1]->z[i], reg->arc[1]->list[i]);
	}
	fprintf(fp,"\n\n# right center line extension, 7.3\n");
	for(i = 0; i < reg->line[2]->nump; i++)
	{
		fprintf(fp, "%8.6f	%8.6f  %8.6f  %8.6f\n",
				reg->line[2]->x[i], reg->line[2]->y[i], reg->line[2]->z[i], reg->arc[2]->list[i]);
	}
	fprintf(fp,"\n\n# ps te extension exit line, 7.4\n");
	for(i = 0; i < reg->line[3]->nump; i++)
	{
		fprintf(fp, "%8.6f	%8.6f  %8.6f  %8.6f\n",
				reg->line[3]->x[i], reg->line[3]->y[i], reg->line[3]->z[i], reg->arc[3]->list[i]);
	}
	fprintf(fp,"\n\n# ps envelope ext.- ss env, te connection\n");
	fprintf(fp, "%8.6f	%8.6f  %8.6f  %8.6f\n",
			reg->line[1]->x[cge->reg[1]->line[2]->nump - cge->reg[3]->line[1]->nump],
			reg->line[1]->y[cge->reg[1]->line[2]->nump - cge->reg[3]->line[1]->nump],
			reg->line[1]->z[cge->reg[1]->line[2]->nump - cge->reg[3]->line[1]->nump],
			reg->arc[1]->list[cge->reg[1]->line[2]->nump - cge->reg[3]->line[1]->nump]);
	fprintf(fp, "%8.6f	%8.6f  %8.6f  %8.6f\n",
			cge->reg[4]->line[2]->x[0], cge->reg[4]->line[2]->y[0],
			cge->reg[4]->line[2]->z[0], cge->reg[4]->arc[2]->list[0]);

	// lower core region, no. 6
	reg = cge->reg[5];
	fprintf(fp,"\n\n# lower core inlet, 6.1\n");
	for(i = 0; i < reg->line[0]->nump; i++)
	{
		fprintf(fp, "%8.6f	%8.6f  %8.6f  %8.6f\n",
				reg->line[0]->x[i], reg->line[0]->y[i], reg->line[0]->z[i], reg->arc[0]->list[i]);
	}
	fprintf(fp,"\n\n# ss, left, core curve, 6.2\n");
	for(i = 0; i < reg->line[1]->nump; i++)
	{
		fprintf(fp, "%8.6f	%8.6f  %8.6f  %8.6f\n",
				reg->line[1]->x[i], reg->line[1]->y[i], reg->line[1]->z[i], reg->arc[1]->list[i]);
	}
	fprintf(fp,"\n\n# ps, right, curve, 6.3\n");
	for(i = 0; i < reg->line[2]->nump; i++)
	{
		fprintf(fp, "%8.6f	%8.6f  %8.6f  %8.6f\n",
				reg->line[2]->x[i], reg->line[2]->y[i], reg->line[2]->z[i], reg->arc[2]->list[i]);
	}
	fprintf(fp,"\n\n# core exit line, 6.4\n");
	for(i = 0; i < reg->line[3]->nump; i++)
	{
		fprintf(fp, "%8.6f	%8.6f  %8.6f  %8.6f\n",
				reg->line[3]->x[i], reg->line[3]->y[i], reg->line[3]->z[i], reg->arc[3]->list[i]);
	}

#ifndef NO_INLET_EXT
	// lower core region, no. 8
	if(cge->reg_num > 7)
	{
		reg = cge->reg[7];
		fprintf(fp,"\n\n# inlet ext., 8.1\n");
		if(reg->line[0] && reg->line[1] && reg->line[2] && reg->line[3])
		{
			for(i = 0; i < reg->line[0]->nump; i++)
			{
				fprintf(fp, "%8.6f	%8.6f  %8.6f  %8.6f\n",
						reg->line[0]->x[i], reg->line[0]->y[i], reg->line[0]->z[i], reg->arc[0]->list[i]);
			}
			fprintf(fp,"\n\n# ss, left, inlet ext., 8.2\n");
			for(i = 0; i < reg->line[1]->nump; i++)
			{
				fprintf(fp, "%8.6f	%8.6f  %8.6f  %8.6f\n",
						reg->line[1]->x[i], reg->line[1]->y[i], reg->line[1]->z[i], reg->arc[1]->list[i]);
			}
			fprintf(fp,"\n\n# ps, right, 8.3\n");
			for(i = 0; i < reg->line[2]->nump; i++)
			{
				fprintf(fp, "%8.6f	%8.6f  %8.6f  %8.6f\n",
						reg->line[2]->x[i], reg->line[2]->y[i], reg->line[2]->z[i], reg->arc[2]->list[i]);
			}
			fprintf(fp,"\n\n# inlet ext. exit, 8.4\n");
			for(i = 0; i < reg->line[3]->nump; i++)
			{
				fprintf(fp, "%8.6f	%8.6f  %8.6f  %8.6f\n",
						reg->line[3]->x[i], reg->line[3]->y[i], reg->line[3]->z[i], reg->arc[3]->list[i]);
			}
		}
	}
#endif										   // NO_INLET_EXT

	// grid lines, ss & ps
	reg = cge->reg[1];
	fprintf(fp,"\n\n# ss, grid lines between 2.2 & 2.3\n");
	for(i = 0; i < reg->line[1]->nump; i++)
	{
		fprintf(fp, "%8.6f	%8.6f  %8.6f  %8.6f\n",
				reg->line[1]->x[i], reg->line[1]->y[i], reg->line[1]->z[i], reg->arc[1]->list[i]);
		fprintf(fp, "%8.6f	%8.6f  %8.6f  %8.6f\n\n",
				reg->line[2]->x[i], reg->line[2]->y[i], reg->line[2]->z[i], reg->arc[2]->list[i]);
	}
	reg = cge->reg[3];
	fprintf(fp,"\n\n# ps, grid lines between 4.3 & 4.2\n");
	for(i = 0; i < reg->line[1]->nump; i++)
	{
		fprintf(fp, "%8.6f	%8.6f  %8.6f  %8.6f\n",
				reg->line[2]->x[i], reg->line[2]->y[i], reg->line[2]->z[i], reg->arc[2]->list[i]);
		fprintf(fp, "%8.6f	%8.6f  %8.6f  %8.6f\n\n",
				reg->line[1]->x[i], reg->line[1]->y[i], reg->line[1]->z[i], reg->arc[1]->list[i]);
	}
	fclose(fp);
	return 0;
}


#ifdef GAP
int DumpGapRegions(struct cgrid *cge, int n)
{
	int i;

	struct region *reg;

	char fn[100];
	FILE *fp;

	sprintf(fn,"rr_gridreg_%02d.txt", n);
	if( (fp = fopen(fn,"a")) == NULL)
	{
		fprintf(stderr, "Shit happened opening file '%s'!\n",fn);
		exit(-1);
	}

	reg = cge->reg[cge->reg_num-2];
	// ss gap region
	fprintf(fp,"\n\n# cl, 9.2\n");
	for(i = 0; i < reg->line[1]->nump; i++)
	{
		fprintf(fp, "%8.6f	%8.6f  %8.6f  %8.6f\n",
				reg->line[1]->x[i], reg->line[1]->y[i], reg->line[1]->z[i], reg->arc[1]->list[i]);
	}
	fprintf(fp,"\n\n# cl, 9.2 -> ss-surface, 2.2\n");
	for(i = 0; i < reg->line[1]->nump; i++)
	{
		fprintf(fp, "%8.6f	%8.6f  %8.6f  %8.6f\n",
				reg->line[1]->x[i], reg->line[1]->y[i], reg->line[1]->z[i], reg->arc[1]->list[i]);
		fprintf(fp, "%8.6f	%8.6f  %8.6f  %8.6f\n\n",
				cge->reg[1]->line[1]->x[i], cge->reg[1]->line[1]->y[i],
				cge->reg[1]->line[1]->z[i], cge->reg[1]->arc[1]->list[i]);
	}
	fprintf(fp,"\n\n# ss-surface, 9.3\n");
	for(i = 0; i < reg->line[2]->nump; i++)
	{
		fprintf(fp, "%8.6f	%8.6f  %8.6f  %8.6f\n",
				reg->line[2]->x[i], reg->line[2]->y[i], reg->line[2]->z[i], reg->arc[2]->list[i]);
	}
	fprintf(fp,"\n\n# trailing edge, 9.4\n");
	for(i = 0; i < reg->line[3]->nump; i++)
	{
		fprintf(fp, "%8.6f	%8.6f  %8.6f  %8.6f\n",
				reg->line[3]->x[i], reg->line[3]->y[i], reg->line[3]->z[i], reg->arc[3]->list[i]);
	}

	// ps gap region
	reg = cge->reg[cge->reg_num-1];
	fprintf(fp,"\n\n# cl, 10.3\n");
	for(i = 0; i < reg->line[2]->nump; i++)
	{
		fprintf(fp, "%8.6f	%8.6f  %8.6f  %8.6f\n",
				reg->line[2]->x[i], reg->line[2]->y[i], reg->line[2]->z[i], reg->arc[2]->list[i]);
	}
	fprintf(fp,"\n\n# cl, 10.2 -> ps-surface, 4.2\n");
	for(i = 0; i < reg->line[2]->nump; i++)
	{
		fprintf(fp, "%8.6f	%8.6f  %8.6f  %8.6f\n",
				reg->line[2]->x[i], reg->line[2]->y[i], reg->line[2]->z[i], reg->arc[2]->list[i]);
		fprintf(fp, "%8.6f	%8.6f  %8.6f  %8.6f\n\n",
				cge->reg[3]->line[2]->x[i], cge->reg[3]->line[2]->y[i],
				cge->reg[3]->line[2]->z[i], cge->reg[3]->arc[2]->list[i]);
	}
	fprintf(fp,"\n\n# bl-surface (ps), 10.2\n");
	for(i = 0; i < reg->line[1]->nump; i++)
	{
		fprintf(fp, "%8.6f	%8.6f  %8.6f  %8.6f\n",
				reg->line[1]->x[i], reg->line[1]->y[i], reg->line[1]->z[i], reg->arc[1]->list[i]);
	}
	fprintf(fp,"\n\n# trailing edge, 10.4\n");
	for(i = 0; i < reg->line[3]->nump; i++)
	{
		fprintf(fp, "%8.6f	%8.6f  %8.6f  %8.6f\n",
				reg->line[3]->x[i], reg->line[3]->y[i], reg->line[3]->z[i], reg->arc[3]->list[i]);
	}

	fclose(fp);
	return 0;
}
#endif											  // GAP
#endif											  // DEBUG_REGIONS
