#include <stdio.h>
#include <stdlib.h>
#ifdef WIN32
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <windows.h>
#else 
#include <string.h>
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

#include "include/rr_dbpreset.h"
#define LLEN 24

static float nqinterpolate(float x[][2], float nq);

int getContourData(struct radial *rr)
{
	int err=0;

	float D1, D2, D1i, b0, Q11, nq;
	float contour_angle_inout[2];
	float open_angle_in[2], open_angle_out[2];

	char buf[200];
	char msg[2000];

	if(!rr->des->spec_revs) {
		fprintf(stdout,"\n getContourData(): No design data available!\n\n");
		return err;
	}

	nq = rr->des->spec_revs;

	// get data from tables
	Q11 = nqinterpolate(Q11opt, nq);
	D2  = sqrt(rr->des->dis/(Q11*sqrt(rr->des->head)));

	D1  = D2/nqinterpolate(d2d1, nq);
	D1i = D1*nqinterpolate(d1id1, nq);
	b0  = D1*nqinterpolate(b0d1, nq);
	
	contour_angle_inout[0] = nqinterpolate(shroudangle_in, nq);
	contour_angle_inout[1] = nqinterpolate(shroudangle_out, nq);

	open_angle_in[0]  = nqinterpolate(hubopen_in, nq);
	open_angle_in[1]  = nqinterpolate(shroudopen_in, nq);
	open_angle_out[0] = nqinterpolate(hubopen_out, nq);
	open_angle_out[1] = nqinterpolate(shroudopen_out, nq);

	// output, fill buf if msg. should be handed to upper routines
	sprintf(buf,"\n +--------------------------------------------------\n");
	sprintf(msg,"%s",buf);
	sprintf(buf," | Suggested meridian parametres for\n");
	strcat(msg,buf);
	sprintf(buf," | %s = %9.4f /min\n","nq",nq);
	strcat(msg,buf);
	sprintf(buf," +--------------------------------------------------\n");
	strcat(msg,buf);
	sprintf(buf," | %-*s = %9.4f /%9.4f m (%7.4f)\n",LLEN,"D2",D2,D2/2,D2/D2);
	strcat(msg,buf);
	sprintf(buf," | %-*s = %9.4f /%9.4f m (%7.4f)\n",LLEN,"D1",D1,D1/2,D1/D2);
	strcat(msg,buf);
	sprintf(buf," | %-*s = %9.4f /%9.4f m (%7.4f)\n",
			LLEN,"D1i",D1i,D1i/2,D1i/D2);
	strcat(msg,buf);
	sprintf(buf," | %-*s = %9.4f m (%7.4f)\n",LLEN,"b0",b0,b0/D2);
	strcat(msg,buf);
	strcat(msg," +--------------------------------------------------\n");
	strcat(msg," | angles (in degree)\n");
	sprintf(buf," | %-*s = %9.4f / %9.4f\n",LLEN,"contour, in/out",
			contour_angle_inout[0],contour_angle_inout[1]);
	strcat(msg,buf);
	sprintf(buf," | %-*s = %9.4f / %9.4f\n",LLEN,"opening, in, hub/shroud",
			open_angle_in[0],open_angle_in[1]);
	strcat(msg,buf);
	sprintf(buf," | %-*s = %9.4f / %9.4f\n",LLEN,"opening, out, hub/shroud",
			open_angle_out[0],open_angle_out[1]);
	strcat(msg,buf);
	strcat(msg," +--------------------------------------------------\n\n");

	fprintf(stdout,"%s",msg);

	return err;
}

static float nqinterpolate(float x[][2], float nq)
{
	int i;

	float y;

	y = -99.0;
	i = 1;
	while(x[i][0] != -99.0) {
		if( (x[i-1][0] <= nq) &&
			(x[i][0] >= nq)) {
			y = x[i][1] + (x[i-1][1]-x[i][1]) * 
				(nq-x[i][0])/(x[i-1][0]-x[i][0]);
			return y;
		}
		i++;
	}
	return y;
}
