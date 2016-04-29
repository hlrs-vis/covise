#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef WIN32
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#endif
#include <math.h>
#include "../General/include/geo.h"
#include "../General/include/cov.h"
#include "../General/include/points.h"
#include "../General/include/flist.h"
#include "../General/include/curve.h"
#include "../General/include/profile.h"
#include "../General/include/common.h"
#include "../General/include/fatal.h"
#include "../General/include/v.h"
#include "include/rr2cov.h"
#include "include/radial.h"

#define NUMBER_OF_SECTIONS 36
#define CHAR_LEN			   200
#define PLOT_SCALE_ENLARGE	   0.05

static int last_err = 0;
const char *err_msg[] =
{
	NULL,"Could not create conformal view!",
	"Design data missing in input file!",
	"There was at least one element with le-angle < te-angle!",
	"Could not create blade element polygon!",
	"Could not write cartesian blade data!",
	"No blade-meridian-intersection point found!"
};

int CreateRR_BladeElements(struct radial *rr);
static struct Point *GetConformalView(struct Point *src, float l0);
static int AddLine2PlotVector(struct Point *line, float *xpl, float *ypl,
							  float *xy, int *pcount);
static void FitPSSSCurves(struct Point *cl, struct Point *ps, struct Point *ss);

struct covise_info *Radial2Covise(struct radial *rr)
{
	struct covise_info *ci = NULL;
	int err = 0;
	int i, j, jmax;
	int npblade, nphub;
	int offs, nose, te;
	float x, y, z;
#ifdef DEBUG_POLYGONS
	FILE *ferr;
	char fname[255];
	static int fcount = 0;
#endif										   // DEBUG_POLYGONS

	// memory for polygon data, delete previous data
	FreeCoviseInfo(ci);
	if (!(err = CreateRR_BladeElements(rr))) {
#ifdef DEBUG_POLYGONS
		sprintf(fname, "rr_polygons_%02d.txt", fcount++);
		ferr = fopen(fname, "w");
		fprintf(ferr, "after CreateRR_BladeElements():\n");
		fprintf(ferr, "rr->be_num		  = %d\n", rr->be_num);
		fprintf(ferr, "rr->be[i]->bp->num = %d\n", rr->be[0]->bp->num);
#endif										// DEBUG_POLYGONS
		if ((ci = AllocCoviseInfo(0)) != NULL)
		{
			// points to copy, distinguish cases: te_thick
			if (rr->be[0]->te_thick)
			{
				jmax = rr->be[0]->ss_cart->nump;
				offs = 2 * rr->be[0]->ss_cart->nump - 1;
				nose = rr->be[0]->ss_cart->nump - 1;
				te	 = 1;
			}
			else
			{
				jmax = rr->be[0]->bp->num - 1;
				offs = 2 * rr->be[0]->bp->num - 2;
				nose = rr->be[0]->ss_cart->nump - 1;
				te	 = 0;
			}
			// assign points to global array from blade elements:
			for (i = 0; i < rr->be_num; i++)
			{
				for (j = rr->be[0]->ps_cart->nump-1; j >= 0; j--)
				{
					x = rr->be[i]->ps_cart->x[j];
					y = rr->be[i]->ps_cart->y[j];
					z = rr->be[i]->ps_cart->z[j];
					AddPoint(ci->p, x, y, z);
				}
				for (j = 1; j < jmax; j++)
				{
					x = rr->be[i]->ss_cart->x[j];
					y = rr->be[i]->ss_cart->y[j];
					z = rr->be[i]->ss_cart->z[j];
					AddPoint(ci->p, x, y, z);
				}
				if (i)
					CreateRR_BEPolygons(ci, i, offs, te);
			}
			CreateRR_BEPolygons(ci, rr->be_num-1, offs, te);
			CreateRR_TipPolygons(ci, nose, te);
		}
		// generate all blades of the runner
		npblade = RotateBlade4Covise(ci, rr->nob);
		// runner hub and shroud
		nphub = CreateRR_Contours(ci, rr);
	}											   // !err
	else ci = NULL;
	last_err = err;

#ifdef DEBUG_POLYGONS
	if (ferr) {
		fprintf(ferr, "\nblade polygon vertices:\n");
		fprintf(ferr, "ci->vx->num = %d\t, ci->vx->max = %d\n", ci->vx->num, ci->vx->max);
		fprintf(ferr, "number of points after RotateBlade4Covise() = %d\n", npblade);
		j = 1;
		fprintf(ferr, "%3d: ", j++);
		for (i = 0; i < ci->vx->num; i++)
		{
			fprintf(ferr, "vx[%3d] = %3d   ", i, ci->vx->list[i]);
			if (!((i+1)%3)) fprintf(ferr, "\n%3d: ", j++);
		}
		fprintf(ferr, "\nindices of blade polygon start vertices:\n");
		fprintf(ferr, "ci->pol->num = %d\t, ci->pol->max = %d\n", ci->pol->num, ci->pol->max);
		for (i = 0; i < ci->pol->num; i++)
		{
			fprintf(ferr, "pol[%3d] = %3d  ", i, ci->pol->list[i]);
			if (!((i+1)%4)) fprintf(ferr, "\n");
		}
		fprintf(ferr, "\nhub polygon vertices:\n");
		fprintf(ferr, "ci->cvx->num = %d\t, ci->cvx->max = %d\n", ci->cvx->num, ci->cvx->max);
		fprintf(ferr, "number of points after CreateRR_Contours() = %d\n", nphub);
		j = 1;
		fprintf(ferr, "%4d: ", j++);
		for (i = 0; i < ci->cvx->num; i++)
		{
			fprintf(ferr, "cvx[%4d] = %4d	", i, ci->cvx->list[i]);
			if (!((i+1)%3)) fprintf(ferr, "\n%3d: ", j++);
		}
		fprintf(ferr, "\nindices of hub polygon start vertices:\n");
		fprintf(ferr, "ci->cpol->num = %d\t, ci->cpol->max = %d\n", ci->cpol->num, ci->cpol->max);
		for (i = 0; i < ci->cpol->num; i++)
		{
			fprintf(ferr, "cpol[%4d] = %4d	", i, ci->cpol->list[i]);
			if (!((i+1)%4)) fprintf(ferr, "\n");
		}
		fprintf(ferr, "\ncoordinates of all vertices:\n");
		fprintf(ferr, "ci->p->nump = %d\t, ci->p->maxp = %d\n", ci->p->nump, ci->p->maxp);
		for (i = 0; i < ci->p->nump; i++)
		{
			fprintf(ferr, "p[%4d].x = %7.3f\t", i, ci->p->x[i]);
			fprintf(ferr, "p[%4d].y = %7.3f\t", i, ci->p->y[i]);
			fprintf(ferr, "p[%4d].z = %7.3f\n", i, ci->p->z[i]);
		}
		fclose(ferr);
	}
#endif										   // DEBUG_POLYGONS
	return ci;
}


char *GetLastErr(void)
{
	return (char *)err_msg[last_err];
}


void CreateRR_BEPolygons(struct covise_info *ci, int be, int offs, int te)
{
	int i, ivx[3];
	static int ipol;

	if (be == 1) ipol = 0;
	// surface polygons
	for (i = 0; i < offs-1; i++) {
		// 1st polygon
		ivx[0] = (be - 1) * offs + i;
		ivx[1] = ivx[0] + offs;
		ivx[2] = ivx[1] + 1;
		Add2Ilist(ci->pol, ipol);
		Add2Ilist(ci->vx, ivx[0]);
		Add2Ilist(ci->vx, ivx[1]);
		Add2Ilist(ci->vx, ivx[2]);
		ipol += 3;
		// 2nd polygon
		ivx[0] = be * offs + 1 + i;
		ivx[1] = ivx[0] - offs;
		ivx[2] = ivx[1] - 1;
		Add2Ilist(ci->pol, ipol);
		Add2Ilist(ci->vx, ivx[0]);
		Add2Ilist(ci->vx, ivx[1]);
		Add2Ilist(ci->vx, ivx[2]);
		ipol += 3;
	}
	// case: te_thick > 0
	if (te) {
		// trailing edge polygons
		// 1st polygon
		ivx[0] = be * offs - 1;
		ivx[1] = ivx[0] + offs;
		ivx[2] = ivx[0] + 1;
		Add2Ilist(ci->pol, ipol);
		Add2Ilist(ci->vx, ivx[0]);
		Add2Ilist(ci->vx, ivx[1]);
		Add2Ilist(ci->vx, ivx[2]);
		ipol += 3;
		// 2nd polygon
		ivx[0] = be * offs;
		ivx[1] = ivx[0] - offs;
		ivx[2] = ivx[0] - 1;
		Add2Ilist(ci->pol, ipol);
		Add2Ilist(ci->vx, ivx[0]);
		Add2Ilist(ci->vx, ivx[1]);
		Add2Ilist(ci->vx, ivx[2]);
		ipol += 3;
	}
}


void CreateRR_TipPolygons(struct covise_info *ci, int npoin, int te)
{
	int i, ivx[3];
	int ipol = ci->vx->num;
	const int np = ci->p->nump - 2 * npoin - 1;

	// nose polygon
	ivx[0] = np + npoin + 1;
	ivx[1] = ivx[0] - 1;
	ivx[2] = ivx[1] - 1;
	Add2Ilist(ci->pol, ipol);
	Add2Ilist(ci->vx, ivx[0]);
	Add2Ilist(ci->vx, ivx[1]);
	Add2Ilist(ci->vx, ivx[2]);
	ipol += 3;
	for (i = 1; i < npoin-1; i++) {
		// 1st polygon
		ivx[0] = np + npoin + 1 + i;
		ivx[1] = ivx[0] - 1;
		ivx[2] = np + npoin - i;
		Add2Ilist(ci->pol, ipol);
		Add2Ilist(ci->vx, ivx[0]);
		Add2Ilist(ci->vx, ivx[1]);
		Add2Ilist(ci->vx, ivx[2]);
		ipol += 3;
		// 2nd polygon
		ivx[0] = np + npoin - i;
		ivx[1] = ivx[0] - 1;
		ivx[2] = np + npoin + 1 + i;
		Add2Ilist(ci->pol, ipol);
		Add2Ilist(ci->vx, ivx[0]);
		Add2Ilist(ci->vx, ivx[1]);
		Add2Ilist(ci->vx, ivx[2]);
		ipol += 3;
	}
	// te polygon(s), case: (te)
	if (te) {
		// 1st polygon
		ivx[0] = np + 2 * npoin;
		ivx[1] = ivx[0] - 1;
		ivx[2] = np + 1;
		Add2Ilist(ci->pol, ipol);
		Add2Ilist(ci->vx, ivx[0]);
		Add2Ilist(ci->vx, ivx[1]);
		Add2Ilist(ci->vx, ivx[2]);
		ipol += 3;
		// 2nd polygon
		ivx[0] = np + 1;
		ivx[1] = np;
		ivx[2] = np + 2 * npoin;
		Add2Ilist(ci->pol, ipol);
		Add2Ilist(ci->vx, ivx[0]);
		Add2Ilist(ci->vx, ivx[1]);
		Add2Ilist(ci->vx, ivx[2]);
		ipol += 3;
	}
	else										   // te polygon, case (!te)
	{
		ivx[0] = np + 1;
		ivx[1] = np;
		ivx[2] = np + 2 * npoin - 1;
		Add2Ilist(ci->pol, ipol);
		Add2Ilist(ci->vx, ivx[0]);
		Add2Ilist(ci->vx, ivx[1]);
		Add2Ilist(ci->vx, ivx[2]);
		ipol += 3;
	}
}


int RotateBlade4Covise(struct covise_info *ci, int nob)
{
	int i, j, ipol, ivx;
	int np, npol, nvx;
	int nstart, nend;
	float rot, roma[2][2];
	float x, y, z;

	np		   = ci->p->nump;
	npol	   = ci->pol->num;
	nvx		   = ci->vx->num;
	rot		   = 2 * M_PI / nob;
	roma[0][0] =  cos(rot);
	roma[0][1] = -sin(rot);
	roma[1][0] =  sin(rot);
	roma[1][1] =  cos(rot);

	for (i = 0; i < nob-1; i++) {
		nstart = i * np;
		nend   = nstart + np;
		// calculate rotated blade point coordinates
		for (j = nstart; j < nend; j++)
		{
			x = ci->p->x[j] * roma[0][0] + ci->p->y[j] * roma[0][1];
			y = ci->p->x[j] * roma[1][0] + ci->p->y[j] * roma[1][1];
			z = ci->p->z[j];
			AddPoint(ci->p, x, y, z);
		}
		// assign rotated polygon vertices
		for (j = i*nvx; j < (i+1)*nvx; j++)
		{
			ivx = ci->vx->list[j] + np;
			Add2Ilist(ci->vx, ivx);
		}
		// assign rotated polygon start vertices
		ipol = ci->pol->list[ci->pol->num-1];
		for (j = 0; j < npol; j++)
		{
			ipol  += 3;
			Add2Ilist(ci->pol, ipol);
		}
	}
	return(ci->p->nump);
}


int CreateRR_Contours(struct covise_info *ci, struct radial *rr)
{
	int i, j, hub, ind;
	int iend, istart;
	int be_max, nphub,p_meridian;
	float x, y, z;
	float angle, roma[2][2];
	const int npblade  = ci->p->nump;
	const float rot = 2.0f * (float) M_PI / NUMBER_OF_SECTIONS;

	// add hub cap middle point coordinates to global array
	x = 0.0;
	y = 0.0;
#ifdef NO_INLET_EXT
	iend   = NPOIN_MERIDIAN-1;
	istart = 0;
#else
	iend   = NPOIN_EXT+NPOIN_MERIDIAN-2;
	istart = NPOIN_EXT-1;
#endif
	p_meridian = NPOIN_MERIDIAN;
	if(rr->showExt) {
		iend   = rr->be[0]->ml->p->nump-1;
		istart = 0;
		p_meridian = iend+1;
	}
#ifdef DEBUG
	fprintf(stderr," CreateRR_Contours(): istart = %d, iend = %d, p_meridian = %d\n",
			istart, iend, p_meridian);
#endif

	z = rr->be[0]->ml->p->z[iend];
	AddPoint(ci->p, x, y, z);
	// append hub contour point coordinates
	for (i = iend; i >= istart; i--)
		AddPoint(ci->p, rr->be[0]->ml->p->x[i], 0.0, rr->be[0]->ml->p->z[i]);
	// rotate hub contour and append points
	for (i = 1; i < NUMBER_OF_SECTIONS; i++) {
		angle	   = i * rot;
		roma[0][0] =  cos(angle);
		roma[0][1] = -sin(angle);
		roma[1][0] =  sin(angle);
		roma[1][1] =  cos(angle);
		for (j = 0; j < p_meridian+1; j++)
		{
			ind = npblade + j;
			x	= ci->p->x[ind] * roma[0][0] + ci->p->y[ind] * roma[0][1];
			y	= ci->p->x[ind] * roma[1][0] + ci->p->y[ind] * roma[1][1];
			z	= ci->p->z[ind];
			AddPoint(ci->p, x, y, z);
		}
	}
	// create hub polygons
	hub = 1;
	for (i = 1; i <= NUMBER_OF_SECTIONS; i++) {
		CreateContourPolygons(ci->lpol, ci->lvx, i, (p_meridian+1), npblade, hub);
	}

	// append shroud contour point coordinates to global array
#ifdef GAP
	be_max = rr->be_num;
#else
	be_max = rr->be_num - 1;
#endif
	nphub = ci->p->nump - npblade;
	for (i = iend; i >= istart; i--)
		AddPoint(ci->p, rr->be[be_max]->ml->p->x[i], 0.0, rr->be[be_max]->ml->p->z[i]);
	// rotate shroud contour and append points
	for (i = 1; i < NUMBER_OF_SECTIONS; i++) {
		angle	   = i * rot;
		roma[0][0] =  cos(angle);
		roma[0][1] = -sin(angle);
		roma[1][0] =  sin(angle);
		roma[1][1] =  cos(angle);
		for (j = 0; j < p_meridian; j++)
		{
			ind = npblade + nphub + j;
			x	= ci->p->x[ind] * roma[0][0] + ci->p->y[ind] * roma[0][1];
			y	= ci->p->x[ind] * roma[1][0] + ci->p->y[ind] * roma[1][1];
			z	= ci->p->z[ind];
			AddPoint(ci->p, x, y, z);
		}
	}
	// create shroud polygons
	hub = 0;
	for (i = 1; i <= NUMBER_OF_SECTIONS; i++) {
		CreateContourPolygons(ci->cpol, ci->cvx, i, p_meridian, (npblade+nphub), hub);
	}

	return(nphub);
}


void CreateContourPolygons(struct Ilist *ci_pol, struct Ilist *ci_vx, int sec, int offs, int np, int hub)
{
	int i;
	int vx[3];
	static int ipol;

	if (sec == 1) ipol = 0;
	if (sec < NUMBER_OF_SECTIONS) {
		if (hub)									// single hub cap polygon
		{
			vx[0]= np  + sec * offs;
			vx[1]= vx[0] + 1;
			vx[2]= vx[1] - offs;
			Add2Ilist(ci_vx, vx[0]);
			Add2Ilist(ci_vx, vx[1]);
			Add2Ilist(ci_vx, vx[2]);
			Add2Ilist(ci_pol, ipol);
			ipol += 3;
		}
		for (i = 1; i < offs-1; i++)
		{
			// 1st polygon
			vx[0]= np  + (sec - 1) * offs + i;
			vx[1]= vx[0] + offs;
			vx[2]= vx[1] + 1;
			Add2Ilist(ci_vx, vx[0]);
			Add2Ilist(ci_vx, vx[1]);
			Add2Ilist(ci_vx, vx[2]);
			Add2Ilist(ci_pol, ipol);
			ipol += 3;
			// 2nd polygon
			vx[0]= np  + sec * offs + 1 + i;
			vx[1]= vx[0] - offs;
			vx[2]= vx[1] - 1;
			Add2Ilist(ci_vx, vx[0]);
			Add2Ilist(ci_vx, vx[1]);
			Add2Ilist(ci_vx, vx[2]);
			Add2Ilist(ci_pol, ipol);
			ipol += 3;
		}
	}
	else										   // (sec == NUMBER_OF_SECTIONS)
	{
		if (hub)									// single hub cap polygon
		{
			vx[0]= np;
			vx[1]= np + 1;
			vx[2]= np + (sec - 1) * offs + 1;
			Add2Ilist(ci_vx, vx[0]);
			Add2Ilist(ci_vx, vx[1]);
			Add2Ilist(ci_vx, vx[2]);
			Add2Ilist(ci_pol, ipol);
			ipol += 3;
		}
		for (i = 1; i < offs-1; i++)
		{
			// 1st polygon
			vx[0]= np + (sec - 1) * offs + i;
			vx[1]= np + i;
			vx[2]= vx[1] + 1;
			Add2Ilist(ci_vx, vx[0]);
			Add2Ilist(ci_vx, vx[1]);
			Add2Ilist(ci_vx, vx[2]);
			Add2Ilist(ci_pol, ipol);
			ipol += 3;
			// 2nd polygon
			vx[0]= np + 1 + i;
			vx[1]= np + (sec - 1) * offs + 1 + i;
			vx[2]= vx[1] - 1;
			Add2Ilist(ci_vx, vx[0]);
			Add2Ilist(ci_vx, vx[1]);
			Add2Ilist(ci_vx, vx[2]);
			Add2Ilist(ci_pol, ipol);
			ipol += 3;
		}
	}
}


void GetXMGRCommands(char *plbuf, float *xy,const char *title,const char *xlabel,const char *ylabel, int q_flag)
{
	int i, d, m;
	char buf[CHAR_LEN];

	float xy_min, xy_max, b, factor, ytic;

	// make boundaries even numbers
	for(i = 0; i < 4; i++) {
		if((b = xy[i]))
		{
			d = m = 0;
			while( ABS(b) > 10)
			{
				b /= 10;
				d++;
			}
			while( ABS(b) < 1)
			{
				b *= 10;
				m++;
			}
			if( ((i >= 2) && (((int)(b*10000))%((int)(b)*10000))!=0) ||
				(i < 2 && (b < 0)))
				b += 0.5*SIGN(b);					  // max-val. -> step up
			b *= 2;									 // double to get 0.5,5,50 ... steps
			b  = (float)((int)(b))/2.0;
			if(d)
				b *= pow((float)10,(int)d);
			else if(m)
				b /= pow((float)10,(int)m);
			xy[i] = b;
		}
	}

	// quadratic plot scale if q_flag
	if(q_flag) {
		xy_min = MIN(xy[0], xy[1]);
		xy_max = MAX(xy[2], xy[3]);
		sprintf(buf, "WORLD %f,%f,%f,%f\n", xy_min, xy_min, xy_max, xy_max);
	}
	else {
		sprintf(buf, "WORLD %f,%f,%f,%f\n", xy[0], xy[1], xy[2], xy[3]);
	}
	sprintf(plbuf,"AUTOSCALE\n");
	strcat(plbuf, buf);
	strcat(plbuf, "SETS SYMBOL 27\n");
	strcat(plbuf, "SETS LINESTYLE 0\n");
	sprintf(buf,"title \"%s\"\n", title);
	strcat(plbuf, buf);
	sprintf(buf, "xaxis	 label \"%s\"\n", xlabel);
	strcat(plbuf, buf);
	sprintf(buf, "yaxis	 label \"%s\"\n", ylabel);
	strcat(plbuf, buf);
	factor = 10.0;
	ytic = (float)(((int)(((xy[3]-xy[1])*factor)))/(9.0*factor));
	while(ytic == 0.0) {
		factor *= 10.0;
		ytic = (float)(((int)(((xy[3]-xy[1])*factor)))/(9.0*factor));
	}
	sprintf(buf, "yaxis	 tick major %f\nyaxis  tick minor %f\n",
			ytic,ytic/2.0);
	strcat(plbuf, buf);

}


void AddXMGRSet2Plot(char *plbuf, float *xy, float *x, float *y, int num,
					 char *title, char *xlabel, char *ylabel,
					 int q_flag, int graph, int set)
{
	int i;
	char buf[CHAR_LEN], gname[CHAR_LEN], sname[CHAR_LEN];

	float xy_min, xy_max;

	sprintf(gname,"g%d",graph);
	sprintf(sname,"s%d",set);

	strcat(plbuf,"VIEW 0.15,0.15,0.85,0.55\n");	   // prev. graph.
	for(i = 0; i < num; i++) {
		sprintf(buf,"%s.%s POINT %10.6f %10.6f\n",gname,sname,x[i],y[i]);
		strcat(plbuf,buf);
	}
	strcat(plbuf,"VIEW 0.15,0.58,0.85,0.85\n");
	if(q_flag) {
		xy_min = MIN(xy[0], xy[1]);
		xy_max = MAX(xy[2], xy[3]);
		sprintf(buf, "WORLD %f,%f,%f,%f\n", xy_min, xy_min, xy_max, xy_max);
	}
	else {
		sprintf(buf, "WORLD %f,%f,%f,%f\n", xy[0], xy[1], xy[2], xy[3]);
	}
	strcat(plbuf,buf);
	sprintf(buf,"title \"%s\"\n", title);
	strcat(plbuf, buf);
	sprintf(buf, "xaxis	 label \"%s\"\n", xlabel);
	strcat(plbuf, buf);
	sprintf(buf, "yaxis	 label \"%s\"\n", ylabel);
	strcat(plbuf, buf);
}

void GetMeridianContourNumbers(int *num_points, float *xy, struct radial *rr, int ext_flag)
{
	int ml_firstp, ml_lastp, be_last, npoin_edge_max;

#ifdef NO_INLET_EXT
	ml_firstp = 0;
#else
	ml_firstp = NPOIN_EXT-1;
#endif										   // NO_INLET_EXT
	ml_lastp = rr->be[0]->ml->p->nump - NPOIN_EXT;
	be_last	 = rr->be_num-1;

	npoin_edge_max = MAX(NPOIN_EDGE,rr->be_num);

	// extended meridian contour
	if(ext_flag) {
		ml_firstp = 0;
		ml_lastp = rr->be[0]->ml->p->nump-1;
		*num_points = 4*(rr->be[0]->ml->p->nump)+4*(npoin_edge_max-1);
	}
	else {
		*num_points = 4*(NPOIN_MERIDIAN)+4*(npoin_edge_max-1);
	}
	xy[0]		= 0.0;
	xy[1]		= rr->be[be_last]->ml->p->z[ml_lastp]
		* (1.0 - (SIGN( rr->be[be_last]->ml->p->z[ml_lastp]))*
		   PLOT_SCALE_ENLARGE);
	xy[2]		= rr->be[be_last]->ml->p->x[ml_firstp]
		* (1.0 + (SIGN( rr->be[be_last]->ml->p->x[ml_firstp]))*
		   PLOT_SCALE_ENLARGE);
	xy[3]		= rr->be[0]->ml->p->z[ml_firstp]
		* (1.0 + (SIGN(rr->be[0]->ml->p->z[ml_firstp]))*
		   PLOT_SCALE_ENLARGE);
#ifdef DEBUG
	fprintf(stderr," num_points = %d\n",*num_points);
#endif
}


void GetMeridianContourPlotData(struct radial *rr, float *xpl, float *ypl, int num_points, int ext_flag)
{
	int i, istart, iend, j, j0;
	int ibe_max;

	// meridian contour
#ifdef NO_INLET_EXT
	istart = 0;
#else
	istart = NPOIN_EXT-1;
#endif										   // NO_INLET_EXT
	ibe_max = rr->be_num-1;
	iend   = rr->be[0]->ml->p->nump-NPOIN_EXT;

	if(ext_flag) {
		istart = 0;
		iend = rr->be[0]->ml->p->nump-1;
	}

	j = 0;
	for(i = istart; i < iend; i++) {
		xpl[j] = rr->be[0]->ml->p->x[i];
		ypl[j] = rr->be[0]->ml->p->z[i];
		j++;
		xpl[j] = rr->be[0]->ml->p->x[i+1];
		ypl[j] = rr->be[0]->ml->p->z[i+1];
		j++;
	}
	for(i = istart; i < iend; i++) {
		xpl[j] = rr->be[ibe_max]->ml->p->x[i];
		ypl[j] = rr->be[ibe_max]->ml->p->z[i];
		j++;
		xpl[j] = rr->be[ibe_max]->ml->p->x[i+1];
		ypl[j] = rr->be[ibe_max]->ml->p->z[i+1];
		j++;
	}

	// blade edges
	istart = 0;
	iend   = NPOIN_EDGE;
	for(i = istart; i < iend; i++) {
		xpl[j] = rr->le->c->p->x[i];
		ypl[j] = rr->le->c->p->z[i];
		j++;
		xpl[j] = rr->le->c->p->x[i+1];
		ypl[j] = rr->le->c->p->z[i+1];
		j++;
	}
	for(i = istart; i < iend; i++) {
		xpl[j] = rr->te->c->p->x[i];
		ypl[j] = rr->te->c->p->z[i];
		j++;
		xpl[j] = rr->te->c->p->x[i+1];
		ypl[j] = rr->te->c->p->z[i+1];
		j++;
	}

#ifdef DEBUG
	for(i = 0; i < j; i++) {
		fprintf(stderr,"%4d	  %10.5f  %10.5f\n",i,xpl[i],ypl[i]);
	}
	fprintf(stderr,"istart = %d, iend = %d\n",istart,iend);
#endif

	// set remaining values
	j0 = j-1;
	for(j = j0+1; j < num_points; j++) {
		xpl[j] = xpl[j0];
		ypl[j] = ypl[j0];
	}

}

void GetMeridianContourPlotData2(struct radial *rr, float *xpl, float *ypl, int num_points, int ext_flag)
{
	int i, istart, iend, j, ite, j0;
	int ibe_max;

	// meridian contour
#ifdef NO_INLET_EXT
	istart = 0;
#else
	istart = NPOIN_EXT-1;
#endif										   // NO_INLET_EXT
	ibe_max = rr->be_num-1;
	iend   = rr->be[0]->ml->p->nump-NPOIN_EXT;

	if(ext_flag) {
		istart = 0;
		iend = rr->be[0]->ml->p->nump-1;
	}

	j = 0;
	for(i = istart; i < iend; i++) {
		xpl[j] = rr->be[0]->ml->p->x[i];
		ypl[j] = rr->be[0]->ml->p->z[i];
		j++;
		xpl[j] = rr->be[0]->ml->p->x[i+1];
		ypl[j] = rr->be[0]->ml->p->z[i+1];
		j++;
	}
	for(i = istart; i < iend; i++) {
		xpl[j] = rr->be[ibe_max]->ml->p->x[i];
		ypl[j] = rr->be[ibe_max]->ml->p->z[i];
		j++;
		xpl[j] = rr->be[ibe_max]->ml->p->x[i+1];
		ypl[j] = rr->be[ibe_max]->ml->p->z[i+1];
		j++;
	}

	// blade edges
	istart = 0;
	iend   = rr->be_num-1;
	ite    = rr->be[0]->cl->nump-1;
	for(i = istart; i < iend; i++) {
		xpl[j] = sqrt(pow(rr->be[i]->cl_cart->x[0],2.0f)+
					  pow(rr->be[i]->cl_cart->y[0],2.0f));
		ypl[j] = rr->be[i]->cl_cart->z[0];
		j++;
		xpl[j] = sqrt(pow(rr->be[i+1]->cl_cart->x[0],2.0f)+
					  pow(rr->be[i+1]->cl_cart->y[0],2.0f));
		ypl[j] = rr->be[i+1]->cl_cart->z[0];
		j++;
	}
	for(i = istart; i < iend; i++) {
		xpl[j] = sqrt(pow(rr->be[i]->cl_cart->x[ite],2.0f)+
					  pow(rr->be[i]->cl_cart->y[ite],2.0f));
		ypl[j] = rr->be[i]->cl_cart->z[ite];
		j++;
		xpl[j] = sqrt(pow(rr->be[i+1]->cl_cart->x[ite],2.0f)+
					  pow(rr->be[i+1]->cl_cart->y[ite],2.0f));
		ypl[j] = rr->be[i+1]->cl_cart->z[ite];
		j++;
	}

	// set remaining values
	j0 = j-1;
	for(j = j0+1; j < num_points; j++) {
		xpl[j] = xpl[j0];
		ypl[j] = ypl[j0];
	}


#ifdef DEBUG
	for(i = 0; i < j; i++) {
		fprintf(stderr,"%4d	  %10.5f  %10.5f\n",i,xpl[i],ypl[i]);
	}
	fprintf(stderr,"istart = %d, iend = %d\n",istart,iend);
#endif

}

void GetConformalViewPlotData(struct radial *rr, float *xpl, float *ypl, float *xy, int c, int v_count)
{
	static int pcount;
	int i;
	float l0;
	struct Point *cl,*ps, *ss;

#ifdef DEBUG
	fprintf(stderr,"GetConformalViewPlotData\n");
#endif

	// start length coord. of cl.
	l0 = rr->te->bmpar->list[c]*
		rr->be[c]->ml->len[rr->be[c]->ml->p->nump-1]; 
	cl = GetConformalView(rr->be[c]->cl, l0);
	ps = GetConformalView(rr->be[c]->ps, l0 + sqrt(pow(rr->be[c]->ps->x[rr->be[c]->ps->nump-1]
													   -rr->be[c]->cl->x[rr->be[c]->ps->nump-1],2)
												   + pow(rr->be[c]->ps->z[rr->be[c]->ps->nump-1]
														 -rr->be[c]->cl->z[rr->be[c]->ps->nump-1],2)));
	ss = GetConformalView(rr->be[c]->ss, l0 - sqrt(pow(rr->be[c]->ss->x[rr->be[c]->ss->nump-1]
													   -rr->be[c]->cl->x[rr->be[c]->ss->nump-1],2)
												   + pow(rr->be[c]->ss->z[rr->be[c]->ss->nump-1]
														 -rr->be[c]->cl->z[rr->be[c]->ss->nump-1],2)));
	FitPSSSCurves(cl,ps,ss);
	// create plot vectors
	if(!v_count) {
		pcount = 0;
		for(i = 0; i < 4; i++) xy[i] = 0.0;
	}
	AddLine2PlotVector(cl, xpl, ypl, xy, &pcount);
	AddLine2PlotVector(ps, xpl, ypl, xy, &pcount);
	AddLine2PlotVector(ss, xpl, ypl, xy, &pcount);
#ifdef DEBUG
	fprintf(stderr,"GetConformalViewPlotData: pcount = %3d\n",pcount);
#endif

	if(xy[0] > ps->x[0]) xy[0] = ps->x[0];		   // xmin
	if(xy[1] > ps->x[1]) xy[1] = ps->x[1];		   // ymin

	if((xy[2]-xy[0]) > (xy[3]-xy[1]) ) xy[3] = xy[1] + (xy[2]-xy[0]);
	else xy[2] = xy[0] + (xy[3]-xy[1]);

	FreePointStruct(cl);
	FreePointStruct(ps);
	FreePointStruct(ss);

#ifdef DEBUG
	fprintf(stderr,"conformal view data\n");
	for(i = 0; i < pcount; i++) {
		fprintf(stderr,"%4d	  %10.5f  %10.5f\n",i,xpl[i],ypl[i]);
	}
	fprintf(stderr,"this was: conf. view data, pcount = %d\n",pcount);
#endif
}


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
#ifdef DEBUG
	fprintf(stderr,"\n GetConformalView():\n");
#endif
	for(i = src->nump-2; i >= 0; i--) {
		dl	= -l;
		l  += sqrt(pow(src->x[i+1]-src->x[i],2) + pow(src->z[i+1]-src->z[i],2));
		dl += l;
		ds	= -s;
		s  += 0.5*(src->x[i+1] + src->x[i]) * (src->y[i] - src->y[i+1]);
		ds += s;
		len+= sqrt(dl*dl + ds*ds);
		AddPoint(line,s,l,len);
#ifdef DEBUG
		fprintf(stderr,"\t %3d	%16.8f	%16.8f	%16.8f (%16.8f)\n",
				i,s,l,len,180.0/M_PI*atan((l-line->y[line->nump-2])/
										  (s-line->x[line->nump-2])) );
#endif
	}
	return (line);
}


static int AddLine2PlotVector(struct Point *line, float *xpl, float *ypl,
							  float *xy, int *pcount)
{
	int i;

	for(i = 1; i < line->nump; i++) {
		xpl[(*pcount)] = line->x[i-1];
		ypl[(*pcount)] = line->y[i-1];
		(*pcount)++;
		xpl[(*pcount)] = line->x[i];
		ypl[(*pcount)] = line->y[i];
		(*pcount)++;
		if(line->x[i] > xy[2]) xy[2] = line->x[i];
		if(line->y[i] > xy[3]) xy[3] = line->y[i];
	}
	return (*pcount);
}


static void FitPSSSCurves(struct Point *cl, struct Point *ps, struct Point *ss)
{
	int i, ilast;

	float dx, dy, len, para;

	if( ((ilast = ps->nump-1) != cl->nump-1) || ilast != ss->nump-1) {
		fprintf(stderr,"\n!!! lost blade profile point somehow!!!\n");
		exit(1);
	}
	// modify ps points to fit to cl!
	len	  = ps->z[ilast];
	dx	  = cl->x[ilast]-ps->x[ilast];
	dy	  = cl->y[ilast]-ps->y[ilast];
	for(i = 1; i <= ilast; i++) {
		para = 1.0 - (len - ps->z[i])/len;
		ps->x[i] += para*dx;
		ps->y[i] += para*dy;
	}
	// ss points
	len	  = ss->z[ilast];
	dx	  = cl->x[ilast]-ss->x[ilast];
	dy	  = cl->y[ilast]-ss->y[ilast];
	for(i = 1; i <= ilast; i++) {
		para = 1.0 - (len - ss->z[i])/len;
		ss->x[i] += para*dx;
		ss->y[i] += para*dy;
	}
}


void GetCamberPlotData(struct radial *rr, float *xpl, float *ypl, float *xy,
					   int c, int v_count)
{
	static int pcount;
	int i;

	float l0, alpha, p_l, len, alpha_deg, rad2deg;
	struct Point *cl;

	rad2deg = 180.0f / (float) M_PI;

	// ****************************************
	// get conformal view
	l0 = 0.0;									   // start length coord. of cl.
	cl = GetConformalView(rr->be[c]->cl, l0);

#ifdef DEBUG
	fprintf(stderr,"\n GetCamberPlotData(): rr->be[%d]->cl: nump = %d\n",c,rr->be[c]->cl->nump);
	for(i = 0; i < cl->nump; i++) {
		fprintf(stderr,"\t %3d	%16.8f	%16.8f	%16.8f\n",i,rr->be[c]->cl->x[i],
				rr->be[c]->cl->y[i], rr->be[c]->cl->z[i]);
	}
	fprintf(stderr," GetCamberPlotData(): done!\n\n");

	fprintf(stderr,"\n GetCamberPlotData(): cl: nump = %d\n",cl->nump);
	for(i = 0; i < cl->nump; i++) {
		fprintf(stderr,"\t %3d	%16.8f	%16.8f	%16.8f\n",i,cl->x[i],
				cl->y[i], cl->z[i]);
	}
	fprintf(stderr," GetCamberPlotData(): done!\n\n");
#endif

	// ****************************************
	// get camber values
	alpha = atan((cl->y[0]-cl->y[1])/(cl->x[0]-cl->x[1]));
	len = cl->z[cl->nump-2];
	if(!v_count) {
		pcount = 0;
		for(i = 0; i < 4; i++) xy[i] = 0.0;
		xy[0] = 0.0;
		xy[1] = 180.0;
		xy[2] = 1.0;
	}
	xpl[pcount] = cl->z[0]/len;
	if(xy[1] > (ypl[pcount] = alpha*rad2deg)) xy[1] = ypl[pcount];
	pcount++;
	for(i = 2; i < cl->nump; i++) {
		alpha	  = atan((cl->y[i-1]-cl->y[i])/(cl->x[i-1]-cl->x[i]));
		p_l		  = cl->z[i-1]/len;
		if((alpha_deg = alpha*rad2deg) < 0.0) alpha_deg += 180.0;
		xpl[pcount] = p_l;
		if(xy[3] < (ypl[pcount] = alpha_deg)) xy[3] = ypl[pcount];
		pcount++;
		if(i < cl->nump-1)
		{
			xpl[pcount] = p_l;
			ypl[pcount] = alpha_deg;
			pcount++;
		}
	}

#ifdef DEBUG
	fprintf(stderr,"\n GetCamberPlotData():\n");
	for(i = 0; i < pcount; i++) {
		fprintf(stderr,"\t %16.8f	%16.8f\n",xpl[i], ypl[i]);
	}
	fprintf(stderr," GetCamberPlotData(): done!\n\n");
#endif

	//xy[1] -= ABS(xy[1])*PLOT_SCALE_ENLARGE;
	xy[1] = 0.0;
	//xy[3] += xy[3]*PLOT_SCALE_ENLARGE;
	xy[3] = 90.0;

	FreePointStruct(cl);
}


void GetNormalizedCamber(struct radial *rr, float *xpl, float *ypl, float *xy,
						 int c, int v_count)
{
	static int pcount;
	int i, istart;

	float l0, alpha, p_l, len, alpha_deg, rad2deg;
	float alphamin, alphamax, delta;
	struct Point *cl;

	rad2deg = 180.0f/(float)M_PI;

	// ****************************************
	// get conformal view
	l0 = alphamax = 0.0;	// start length coord. of cl. / max angle
	cl = GetConformalView(rr->be[c]->cl, l0);

	// ****************************************
	// get camber values
	alpha = atan((cl->y[0]-cl->y[1])/(cl->x[0]-cl->x[1]));
	len = cl->z[cl->nump-2];
	if(!v_count) {
		pcount = 0;
		xy[0] = 0.0;
		xy[2] = 1.0;
	}
	istart = pcount;
	xpl[pcount] = cl->z[0]/len;
	ypl[pcount] = alphamin = alpha*rad2deg;
	pcount++;
	for(i = 2; i < cl->nump; i++) {
		alpha	  = atan((cl->y[i-1]-cl->y[i])/(cl->x[i-1]-cl->x[i]));
		p_l		  = cl->z[i-1]/len;
		if((alpha_deg = alpha*rad2deg) < 0.0) alpha_deg += 180.0;
		xpl[pcount] = p_l;
		ypl[pcount] = alpha_deg;
		pcount++;
		if(i < cl->nump-1)
		{
			xpl[pcount] = p_l;
			ypl[pcount] = alphamax = alpha_deg;
			pcount++;
		}
	}
	delta = 1.0/(alphamax-alphamin);
	for(i = istart; i < pcount; i++) {
		ypl[i] -= alphamin;
		ypl[i] *= delta;
	}

	xy[1] = 0.0;
	xy[3] = 1.0;

	FreePointStruct(cl);
}


void GetMaxThicknessData(struct radial *rr, float *xpl, float *ypl, float *xy)
{
	int i, j, pcount;
	float t, t_max;

	pcount = 0;
	// ss
#ifdef DEBUG
	fprintf(stderr,"GetMaxThicknessData: pcount = %d\n",pcount);
#endif
	for(i = 0; i < rr->be_num; i++) {
		t_max = 0.0;
		for(j = 0; j < rr->be[i]->cl->nump; j++)
		{
			t = sqrt(pow(rr->be[i]->cl_cart->x[j]-rr->be[i]->ss_cart->x[j],2) +
					 pow(rr->be[i]->cl_cart->y[j]-rr->be[i]->ss_cart->y[j],2) +
					 pow(rr->be[i]->cl_cart->z[j]-rr->be[i]->ss_cart->z[j],2));
			if(t_max < t) t_max = t;
		}
		xpl[pcount] = rr->be[i]->para;
		ypl[pcount] = -t_max;
		pcount++;
		if(i && i < rr->be_num-1)
		{
			xpl[pcount] = rr->be[i]->para;
			ypl[pcount] = -t_max;
			pcount++;
		}
	}
#ifdef DEBUG
	fprintf(stderr,"GetMaxThicknessData: pcount = %d\n",pcount);
#endif

	// ps
	for(i = 0; i < rr->be_num; i++) {
		t_max = 0.0;
		for(j = 0; j < rr->be[i]->cl->nump; j++)
		{
			t = sqrt(pow(rr->be[i]->cl_cart->x[j]-rr->be[i]->ps_cart->x[j],2) +
					 pow(rr->be[i]->cl_cart->y[j]-rr->be[i]->ps_cart->y[j],2) +
					 pow(rr->be[i]->cl_cart->z[j]-rr->be[i]->ps_cart->z[j],2));
			if(t_max < t) t_max = t;
		}
		xpl[pcount] = rr->be[i]->para;
		ypl[pcount] = t_max;
		pcount++;
		if(i && i < rr->be_num-1)
		{
			xpl[pcount] = rr->be[i]->para;
			ypl[pcount] = t_max;
			pcount++;
		}
	}
#ifdef DEBUG
	fprintf(stderr,"GetMaxThicknessData: pcount = %d\n",pcount);
#endif
	// hub-line
	xpl[pcount] = xpl[0];
	ypl[pcount] = xy[1] =  1.2*ypl[0];			   // max. thickness supposed to be at hub (0)
	pcount++;
	xpl[pcount] = xpl[0];
	ypl[pcount] = xy[3] = -1.2*ypl[0];			   // max. thickness supposed to be at hub (0)
	pcount++;
#ifdef DEBUG
	fprintf(stderr,"GetMaxThicknessData: pcount = %d\n",pcount);
#endif

	xy[0] =	 0.0;
	xy[2] =	 1.0;

}


void GetOverlapPlotData(struct radial *rr, float *xpl, float *ypl, float *xy)
{
	int i, pcount;
	float theta0, rad2deg, ratio;

	pcount	= 0;
	xy[3]	= 0.0;
	xy[1]	= 200.0;
	rad2deg = 180.0f/(float)M_PI;
	theta0	= 360.0f/rr->nob;
	ratio	= 100 * ( ((rr->be[0]->cl->y[0] - rr->be[0]->cl->y[rr->be[0]->cl->nump-1]) * rad2deg)
					  / theta0 - 1.0);
	if(xy[1] > ratio) xy[1] = ratio;
	if(xy[3] < ratio) xy[3] = ratio;
	xpl[pcount] = rr->be[0]->para;
	ypl[pcount] = ratio;
	pcount++;
	for(i = 1; i < rr->be_num; i++) {
		ratio	= 100.0 * ( ((rr->be[i]->cl->y[0] - rr->be[i]->cl->y[rr->be[i]->cl->nump-1]) * rad2deg)
							/ theta0 - 1.0);
		xpl[pcount] = rr->be[i]->para;
		ypl[pcount] = ratio;
		pcount++;
		if(i < rr->be_num-1)
		{
			xpl[pcount] = rr->be[i]->para;
			ypl[pcount] = ratio;
			pcount++;
		}
		if(xy[1] > ratio) xy[1] = ratio;
		if(xy[3] < ratio) xy[3] = ratio;
	}
	xy[0]	=  0.0;
	xy[2]	=  1.0;
}


void GetBladeAnglesPlotData(struct radial *rr, float *xpl, float *ypl, float *xy)
{
	int i, num, pcount;
	float dl, ds, rad2deg, alpha;

	struct Flist *le_ang;
	struct Flist *te_ang;

	struct Point *cl;

	le_ang = AllocFlistStruct(rr->be_num+1);
	te_ang = AllocFlistStruct(rr->be_num+1);

	rad2deg = 180.0f/(float)M_PI;
	pcount	= 0;

	// calc blade angles at inlet & outlet
	for(i = 0; i < rr->be_num; i++) {
		if( (cl	 = rr->be[i]->cl))
		{
			num = cl->nump-1;
			// le angle
			dl	= sqrt(pow(cl->x[1]-cl->x[0],2) + pow(cl->z[1]-cl->z[0],2));
			ds	= 0.5*(cl->x[1] + cl->x[0]) * (cl->y[0] - cl->y[1]);
			if( (alpha = atan(dl/ds)*rad2deg) < 0.0) alpha += 180.0;
			Add2Flist(le_ang, alpha);
			// te_angle
			dl	= sqrt(pow(cl->x[num]-cl->x[num-1],2) +
					   pow(cl->z[num]-cl->z[num-1],2));
			ds	= 0.5*(cl->x[num] + cl->x[num-1]) * (cl->y[num-1] - cl->y[num]);
			if( (alpha = atan(dl/ds)*rad2deg) < 0.0) alpha += 180.0;
			Add2Flist(te_ang, alpha);
		}
	}
	if((rr->be_num != le_ang->num) ||(rr->be_num != te_ang->num)) {
		fatal("point number mismatch for blade angles!");
		return;
	}
	// fill plot vectors
	xpl[pcount] = rr->be[0]->para;
	xy[1] = xy[3] = ypl[pcount] = le_ang->list[0];
	pcount++;
	for(i = 1; i < rr->be_num; i++) {
		xpl[pcount] = rr->be[i]->para;
		if(xy[1] > (ypl[pcount] = le_ang->list[i])) xy[1] = ypl[pcount];
		else if (xy[3] < ypl[pcount]) xy[3] = ypl[pcount];
		pcount++;
		if(i < rr->be_num-1)
		{
			xpl[pcount] = rr->be[i]->para;
			ypl[pcount] = le_ang->list[i];
			pcount++;
		}
	}
	xpl[pcount] = rr->be[0]->para;
	ypl[pcount] = te_ang->list[0];
	pcount++;
	for(i = 1; i < rr->be_num; i++) {
		xpl[pcount] = rr->be[i]->para;
		if(xy[1] > (ypl[pcount] = te_ang->list[i])) xy[1] = ypl[pcount];
		else if (xy[3] < ypl[pcount]) xy[3] = ypl[pcount];
		pcount++;
		if(i < rr->be_num-1)
		{
			xpl[pcount] = rr->be[i]->para;
			ypl[pcount] = te_ang->list[i];
			pcount++;
		}
	}
	//xy[1]	 *=	 (1.0 - PLOT_SCALE_ENLARGE);
	xy[1]	= 0.0;
	//xy[3]	 *=	 (1.0 + PLOT_SCALE_ENLARGE);
	xy[3]	= 90.0;
	xy[0]	=  0.0;
	xy[2]	=  1.0;
	FreeFlistStruct(le_ang);
	FreeFlistStruct(te_ang);
}


void GetEulerAnglesPlotData(struct radial *rr, float *xpl,
							float *ypl, float *xy)
{
	int i, j, pcount;

	float rad2deg, maxangle;

	pcount = 0;
	rad2deg = 180.0f/(float)M_PI;
	maxangle = 0.0;
	// le(j = 0), te(j = 1) angles
	for(j = 0; j < 2; j++) {
		for(i = 1; i < rr->be_num; i++)
		{
			xpl[pcount] = rr->be[i-1]->para;
			ypl[pcount] = rad2deg*rr->be[i-1]->angle[j];
			maxangle = MAX(maxangle,ypl[pcount]);
			pcount++;
			xpl[pcount] = rr->be[i]->para;
			ypl[pcount] = rad2deg*rr->be[i]->angle[j];
			maxangle = MAX(maxangle,ypl[pcount]);
			pcount++;
		}
	}
	// plot range
	xy[0]	=  0.0;
	xy[2]	=  1.0;
	xy[1]	= 0.0;
	xy[3]	= 90.0;
	if(maxangle > xy[3]) xy[3] = maxangle+10.0;
	if(rr->be[0]->angle[0] > xy[3]) {
		xy[3]  = rr->be[0]->angle[0]+10.0;
		xy[3] *= 0.1f;
		xy[3]  = (float)((int)xy[3])/0.1;
	}
}


void GetMeridianVelocityPlotData(struct radial *rr, float *xpl,
								 float *ypl, float *xy)
{
	int i, j, pcount;
	pcount = 0;
	// le(j = 0), te(j = 1) angles
	for(j = 0; j < 2; j++) {
		for(i = 1; i < rr->be_num; i++)
		{
			xpl[pcount] = rr->be[i-1]->para;
			ypl[pcount] = rr->be[i-1]->mer_vel[j];
			pcount++;
			xpl[pcount] = rr->be[i]->para;
			ypl[pcount] = rr->be[i]->mer_vel[j];
			pcount++;
		}
	}
	// plot range
	xy[0]	=  0.0;
	xy[2]	=  1.0;
	xy[1]	=  0.0;
	xy[3]	= -1.0;
	for(i = 0; i < pcount; i++) {
		if(ypl[i] > xy[3]) xy[3] = ypl[i];
	}
}

void GetCircumferentialVelocityPlotData(struct radial *rr, float *xpl,
								 float *ypl, float *xy)
{
	int i, j, pcount;
	pcount = 0;
	// le(j = 0), te(j = 1) angles
	for(j = 0; j < 2; j++) {
		for(i = 1; i < rr->be_num; i++)
		{
			xpl[pcount] = rr->be[i-1]->para;
			ypl[pcount] = rr->be[i-1]->rot_abs[j];
			pcount++;
			xpl[pcount] = rr->be[i]->para;
			ypl[pcount] = rr->be[i]->rot_abs[j];
			pcount++;
		}
	}
	// plot range
	xy[0]	=  0.0;
	xy[2]	=  1.0;
	xy[1]	=  0.0;
	xy[3]	= -1.0;
	for(i = 0; i < pcount; i++) {
		if(ypl[i] < xy[1]) xy[1] = ypl[i];
		else if(ypl[i] > xy[3]) xy[3] = ypl[i];
	}
}

int PutBladeData(struct radial *rr)
{
	int i, j;

	char fn[200];
	FILE *fp;

	// **************************************************
	// blades in cartesian coords.
	sprintf(fn,"blade.data");
	if( (fp = fopen(fn,"w+")) == NULL) {
		last_err = PUT_BLADEDATA_ERR;
		return 1;
	}
	fprintf(fp,"# blade data sorted by blade elements (%d)\n",
			rr->be_num);
	fprintf(fp,"# rated Q, H, n, z: %.3f, %.3f, %.3f, %d\n#\n",
			rr->des->dis, rr->des->head, rr->des->revs, rr->nob);
	fprintf(fp,"# x	  y	  z [m]\n");
	// ps
	fprintf(fp,"# pressure sides\n");
	for(i = 0; i < rr->be_num; i++) {
		fprintf(fp,"\n\n# ps %d/%d, %d points\n",i+1,rr->be_num,
				rr->be[i]->ps_cart->nump);
		for(j = 0; j < rr->be[i]->ps_cart->nump; j++)
			fprintf(fp,"%14.5f	%14.5f	%14.5f\n",
					rr->be[i]->ps_cart->x[j], rr->be[i]->ps_cart->y[j],
					rr->be[i]->ps_cart->z[j]);
	}
	// ss
	fprintf(fp,"# suction sides\n");
	for(i = 0; i < rr->be_num; i++) {
		fprintf(fp,"\n\n# ss %d/%d, %d points\n",i+1,rr->be_num,
				rr->be[i]->ss_cart->nump);
		for(j = 0; j < rr->be[i]->ss_cart->nump; j++)
			fprintf(fp,"%14.5f	%14.5f	%14.5f\n",
					rr->be[i]->ss_cart->x[j], rr->be[i]->ss_cart->y[j],
					rr->be[i]->ss_cart->z[j]);
	}
	// center lines
	fprintf(fp,"# center lines\n");
	for(i = 0; i < rr->be_num; i++) {
		fprintf(fp,"\n\n# cl %d/%d, %d points\n",i+1,rr->be_num,
				rr->be[i]->cl_cart->nump);
		for(j = 0; j < rr->be[i]->cl_cart->nump; j++)
			fprintf(fp,"%14.5f	%14.5f	%14.5f\n",
					rr->be[i]->cl_cart->x[j], rr->be[i]->cl_cart->y[j],
					rr->be[i]->cl_cart->z[j]);
	}

	fclose(fp);
	// **************************************************
	// blades for proE.
	sprintf(fn,"blade.ibl");
	if( (fp = fopen(fn,"w+")) == NULL) {
		last_err = PUT_BLADEDATA_ERR;
		return 1;
	}
	fprintf(fp,"Closed Index Arclength\n");
	for(i = 0; i < rr->be_num; i++) {
		fprintf(fp,"begin section ! %3d\n",i);
		fprintf(fp,"begin curve\n");
		for(j = rr->be[i]->ps_cart->nump-1; j > 0; j--)
			fprintf(fp,"%14.5f	%14.5f	%14.5f\n",
					rr->be[i]->ps_cart->x[j],
					rr->be[i]->ps_cart->y[j],
					rr->be[i]->ps_cart->z[j]);
		for(j = 0; j < rr->be[i]->ss_cart->nump; j++)
			fprintf(fp,"%14.5f	%14.5f	%14.5f\n",
					rr->be[i]->ss_cart->x[j],
					rr->be[i]->ss_cart->y[j],
					rr->be[i]->ss_cart->z[j]);
	}
	fclose(fp);
	// center line
	sprintf(fn,"centerline.ibl");
	if( (fp = fopen(fn,"w+")) == NULL) {
		last_err = PUT_BLADEDATA_ERR;
		return 1;
	}
	fprintf(fp,"Closed Index Arclength\n");
	for(i = 0; i < rr->be_num; i++) {
		fprintf(fp,"begin section ! %3d\n",i);
		fprintf(fp,"begin curve\n");
		for(j = 0; j < rr->be[i]->cl_cart->nump; j++)
			fprintf(fp,"%14.5f	%14.5f	%14.5f\n",
					rr->be[i]->cl_cart->x[j], rr->be[i]->cl_cart->y[j],
					rr->be[i]->cl_cart->z[j]);
	}
	fclose(fp);

#ifndef NO_INLET_EXT
	j = NPOIN_EXT-1;
#else
	j = 0;
#endif
	// hub contour
	sprintf(fn,"hub.ibl");
	if( (fp = fopen(fn,"w+")) == NULL) {
		last_err = PUT_BLADEDATA_ERR;
		return 1;
	}
	fprintf(fp,"Closed Index Arclength\n");
	fprintf(fp,"begin section\n");
	fprintf(fp,"begin curve\n");
	for(i = j; i < rr->be[0]->ml->p->nump-(NPOIN_EXT-1); i++) {
		fprintf(fp,"%14.5f	%14.5f	%14.5f\n",
				rr->be[0]->ml->p->x[i], rr->be[0]->ml->p->y[i],
				rr->be[0]->ml->p->z[i]);
	}
	fclose(fp);

	// shroud contour
	sprintf(fn,"shroud.ibl");
	if( (fp = fopen(fn,"w+")) == NULL) {
		last_err = PUT_BLADEDATA_ERR;
		return 1;
	}
	fprintf(fp,"Closed Index Arclength\n");
	fprintf(fp,"begin section\n");
	fprintf(fp,"begin curve\n");
	for(i = j; i < rr->be[rr->be_num-1]->ml->p->nump-(NPOIN_EXT-1); i++) {
		fprintf(fp,"%14.5f	%14.5f	%14.5f\n",
				rr->be[rr->be_num-1]->ml->p->x[i],
				rr->be[rr->be_num-1]->ml->p->y[i],
				rr->be[rr->be_num-1]->ml->p->z[i]);
	}
	fclose(fp);

	return 0;
}
