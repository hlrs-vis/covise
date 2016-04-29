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

#include "../General/include/profile.h"
#include "../General/include/points.h"

#define INTERPOL_PARTS 4
// interpolate between blade profile points given in cfg-file
struct profile *InterpolBladeProfile(struct profile *bp_src)
{
	int i, j, istart, iend;
	float c, t, ratio;
	struct profile *bp_tgt;

	if( (istart = GetPointIndex(bp_src->num, bp_src->c, 0.1f, 0)) < 2) istart = 2;
	if( (iend	= GetPointIndex(bp_src->num, bp_src->c, 0.9f, istart)) < istart) iend = istart+1;

	bp_tgt = AllocBladeProfile();

	// copy src to target
	bp_tgt->naca  = bp_src->naca;
	bp_tgt->t_sec = bp_src->t_sec;
	for(i = 0; i < istart; i++) {
		AddProfilePoint(bp_tgt, bp_src->c[i], bp_src->t[i]);
	}
	for(i = istart; i < iend; i++) {
		for(j = 1; j < INTERPOL_PARTS; j++) {
			ratio = (float)(j)/(float)(INTERPOL_PARTS-1);
			c = ratio * bp_src->c[i] + (1.0 - ratio) * bp_src->c[i-1];
			t = ratio * bp_src->t[i] + (1.0 - ratio) * bp_src->t[i-1];
			AddProfilePoint(bp_tgt, c, t);
#ifdef DEBUG_BLADE_PROFILE
			fprintf(stderr,"i, j: %3d, %3d: ratio, c, t: %f, %f, %f\n",
					i,j,ratio, c, t);
#endif
		}
	}
	for(i = iend; i < bp_src->num; i++) {
		AddProfilePoint(bp_tgt, bp_src->c[i], bp_src->t[i]);
	}
	FreeBladeProfile(bp_src);

	return bp_tgt;
}
#undef INTERPOL_PARTS
