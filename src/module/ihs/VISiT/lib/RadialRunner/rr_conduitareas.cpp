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
#include "../General/include/curve.h"
#include "../General/include/flist.h"
#include "../General/include/points.h"

int CreateRR_ConduitAreas(struct radial *rr)
{
	int i, j;					   // loop var(s)
	float area;					   // conduit area, to be computed

	struct curve *hub;
	struct curve *shroud;

#ifdef DEBUG_AREAS
	char fn[100];
	FILE *fp;
#endif

#ifdef GAP
	(rr->be_num)++;
#endif

	// check if memory has to be freed
	for(i = 0; i < rr->be_num; i++) {
		if(rr->be[i]->area) {
			FreeFlistStruct(rr->be[i]->area);
			rr->be[i]->area = NULL;
		}
		rr->be[i]->area = AllocFlistStruct(rr->be[i]->ml->p->nump+1);
	}

#ifdef DEBUG_AREAS
	sprintf(fn,"rr_area.txt");
	if( (fp = fopen(fn,"w+")) == NULL ) {
		fprintf(stderr," error opening file '%s'\n",fn);
		exit(1);
	}
#endif

	// calculate conduit areas at meridian curve points
	hub	   = rr->be[0]->ml;
	shroud = rr->be[rr->be_num-1]->ml;
	for(i = 0; i < rr->be[0]->ml->p->nump; i++)	{   // loop over points
		area  = pow((float)(shroud->p->x[i] - hub->p->x[i]),2);
		area += pow((float)(shroud->p->y[i] - hub->p->y[i]),2);
		area += pow((float)(shroud->p->z[i] - hub->p->z[i]),2);
		area  = sqrt(area);
		area *= 2.0 * M_PI * ((hub->p->x[i] + shroud->p->x[i])/(2.0));

#ifdef DEBUG_AREAS
		fprintf(fp,"%f	%f	%f	 %f	 %f\n",hub->par[i],shroud->par[i],area, hub->p->x[i],hub->p->z[i]);
		fprintf(fp,"%f	%f	%f	 %f	 %f\n",hub->par[i],shroud->par[i],area, shroud->p->x[i],shroud->p->z[i]);
#endif

		// copy value to meridian elements
		for(j = 0; j < rr->be_num; j++) {
			Add2Flist(rr->be[j]->area, area);
		}
	}						   // end i, loop over meridian points
#ifdef GAP
	(rr->be_num)--;
#endif

#ifdef DEBUG_AREAS
	fprintf(stderr, "areas\n");
	DumpFlist(rr->be[0]->area);
	fclose(fp);
#endif
	return 0;
}
