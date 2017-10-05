#include <stdio.h>
#include <stdlib.h>
#ifdef WIN32
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#endif
#include <math.h>
#include "../General/include/geo.h"
#include "../General/include/flist.h"
#include "../General/include/curve.h"
#include "../General/include/points.h"
#include "../General/include/profile.h"
#include "../General/include/fatal.h"

#ifdef RADIAL_RUNNER
#include "include/radial.h"
#endif
#ifdef DIAGONAL_RUNNER
#include "include/diagonal.h"
#endif

#define SHIFTANGLE(alpha) ( (alpha)<(0.0) ? (alpha)+(M_PI) : (alpha) )

static int CalcRR_Angles(struct radial *rr);
static int CalcRR_PumpAngles(struct radial *rr);
static float InterpolConduitArea(struct Flist *be_area, struct curve *ml,
								 float bmpar);
static int CalcRR_InletMerVel(struct radial *rr);

int CalcRR_BladeAngles(struct radial *rr)
{
	int err = 0;

#ifdef DEBUG_ANGLES
	FILE *fp;
	char fn[100];
	int i, j;
#endif

	if(!rr->des) {
		fatal("missing design data in input file!");
		return DESIGN_DATA_ERR;
	}
	if(rr->des->revs == 0 || rr->des->dis == 0 ||  rr->des->head == 0) {
		fatal("incomplete design data set!");
		return DESIGN_DATA_ERR;
	}

#ifdef GAP
	(rr->be_num)++;
#endif

	CalcRR_InletMerVel(rr);
	if (rr->pump) {
		err = CalcRR_PumpAngles(rr);
	}
	else {
		err = CalcRR_Angles(rr);	
	}

#ifdef DEBUG_ANGLES
	for(i = 0; i < rr->be_num; i++) {
		sprintf(fn,"rr_blade_elem_%02d.txt", i);

		if( (fp = fopen(fn,"w+")) == NULL ) {
			fprintf(stderr," ERROR opening file '%s'.\n",fn);
			exit(1);
		}
		fprintf(fp," spec_revs = %8.5f\n",rr -> des -> spec_revs);
		fprintf(fp," be[%02d]->para = %f\n",i,rr->be[i]->para);
		fprintf(fp," be[%02d]->bp->num = %d\n",i,rr->be[i]->bp->num);
		for(j=0; j<2; fprintf(fp," be[%02d]->con_area[%1d]	= %8.5f\n",
							  i,j,rr->be[i]->con_area[j]),j++);
		for(j=0; j<2; fprintf(fp," be[%02d]->mer_vel[%1d]	= %8.5f\n",
							  i,j,rr->be[i]->mer_vel[j]),j++);
		for(j=0; j<2; fprintf(fp," be[%02d]->cir_vel[%1d]	= %8.5f\n",
							  i,j,rr->be[i]->cir_vel[j]),j++);
		for(j=0; j<2; fprintf(fp," be[%02d]->rot_abs[%1d]	= %8.5f\n",
							  i,j,rr->be[i]->rot_abs[j]),j++);
		for(j=0; j<2; fprintf(fp," be[%02d]->angle[%1d]		= %8.5f\n",
							  i,j,180/M_PI*rr->be[i]->angle[j]),j++);
		for(j=0; j<2; fprintf(fp," be[%02d]->mod_angle[%1d] = %8.5f\n",
							  i,j,180/M_PI*rr->be[i]->mod_angle[j]),j++);
		DumpCurve(rr->be[i]->ml,fp);

		fclose(fp);
	}

	DumpFlist(rr->be[0]->area);
#endif

#ifdef GAP
	(rr->be_num)--;
#endif

	return err;
}


// basic calculation of rad. runner inlet/outlet blade angles
static int CalcRR_Angles(struct radial *rr)
{
	int i, err = 0, errfile = 0;

	char fn[200];
	FILE *fp;

	// calc. outlet angle for each blade element
	for(i = 0; i < rr->be_num; i++) {
		// trailing edge first!
		// conduit area for current blade element
		rr->be[i]->con_area[1] = InterpolConduitArea(rr->be[i]->area, rr->be[i]->ml,
													 rr->te->bmpar->list[i]);
		// meridional, circumferential vel. and rotational part of absolute vel.
		rr->be[i]->mer_vel[1] = rr->des->dis/rr->be[i]->con_area[1];
		rr->be[i]->cir_vel[1] = float(rr->te->bmint->x[i] * M_PI * rr->des->revs/30.0);
		// outlet blade angle
		rr->be[i]->angle[1]	 = float(atan( rr->be[i]->mer_vel[1] / (rr->be[i]->cir_vel[1]-rr->be[i]->rot_abs[1]) ));
		rr->be[i]->angle[1]	 = float(SHIFTANGLE(rr->be[i]->angle[1]));
		rr->be[i]->angle[1] -= rr->be[i]->mod_angle[1];

		// now do the leading edge
		// conduit area for current blade element
		rr->be[i]->con_area[0]	 = InterpolConduitArea(rr->be[i]->area, rr->be[i]->ml,
													   rr->le->bmpar->list[i]);
		// meridional, circumferential vel. and rotational part of absolute vel.
		if(rr->des->vratio <= 0.0 || !rr->vratio_flag)
			rr->be[i]->mer_vel[0] =
				rr->des->dis/rr->be[i]->con_area[0];
		rr->be[i]->cir_vel[0]	 = float(rr->le->bmint->x[i] * M_PI*rr->des->revs/30.0);
		rr->be[i]->rot_abs[0]	 = (9.81f*rr->des->head + rr->be[i]->cir_vel[1]*rr->be[i]->rot_abs[1]) /
			(rr->be[i]->cir_vel[0]);
		// inlet blade angle
		rr->be[i]->angle[0]	 = float(atan( rr->be[i]->mer_vel[0] /( rr->be[i]->cir_vel[0]-rr->be[i]->rot_abs[0])));
		rr->be[i]->angle[0]	 = float(SHIFTANGLE(rr->be[i]->angle[0]));
		rr->be[i]->angle[0] += rr->be[i]->mod_angle[0];
		if(rr->be[i]->angle[0] < rr->be[i]->angle[1]) err = EULER_ERR;

	}											   // end i, blade elements

	if(err == EULER_ERR) {
		sprintf(fn,"euler.err");
		if ( (fp = fopen(fn,"w+")) == NULL) {
			fprintf(stderr,"Cannot open file '%s'.\n",fn);
			errfile = 0;
			fp = stderr;
		}
		else errfile = 1;
	  
		fprintf(fp,"# %s (%d) err = %d\n", __FILE__,__LINE__,err);
		for(i = 0; i < rr->be_num; i++) {
			fprintf(fp,"%4d   %.4f   %14.5f %14.5f %14.5f %14.5f %14.5f %14.5f %14.5f %14.5f",
					i,rr->be[i]->para,
					rr->be[i]->angle[0]*180./M_PI,
					rr->be[i]->angle[1]*180./M_PI,
					rr->be[i]->mer_vel[0],
					rr->be[i]->cir_vel[0],rr->be[i]->rot_abs[0],
					rr->be[i]->mer_vel[1],
					rr->be[i]->cir_vel[1],rr->be[i]->rot_abs[1]);
			if(rr->be[i]->angle[0] < rr->be[i]->angle[1])
				fprintf(fp,"\t!!! ERROR !!!\n");

			else fprintf(fp,"\t ok\n");

		}
		if(errfile) fclose(fp);
	}


#ifdef DEBUG_AREAS
	sprintf(fn,"rr_lete_areas.txt");
	if ( (fp = fopen(fn,"w+")) == NULL) {
		fprintf(stderr,"Cannot open file '%s'.\n",fn);
		exit(1);
	}

	for(i=0; i<rr->be_num; i++) {
		fprintf(fp,"%f	%f\n", rr->le->bmint->x[i], rr->be[i]->con_area[0]);
	}
	fprintf(fp,"\n");
	for(i=0; i<rr->be_num; i++) {
		fprintf(fp,"%f	%f\n", rr->te->bmint->x[i], rr->be[i]->con_area[1]);
	}
	fclose(fp);
#endif

	return err;
}

// basic calculation of rad. runner inlet/outlet blade angles for pumps
static int CalcRR_PumpAngles(struct radial *rr)
{
	int i, err = 0, errfile = 0;

	char fn[200];
	FILE *fp;

	for(i = 0; i < rr->be_num; i++) {
		// leading edge
		// conduit area for current blade element
		rr->be[i]->con_area[0] = InterpolConduitArea(rr->be[i]->area, rr->be[i]->ml,
													 rr->le->bmpar->list[i]);
		// meridional, circumferential vel. and rotational part of absolute vel.
		rr->be[i]->mer_vel[0] = rr->des->dis/rr->be[i]->con_area[0];
		rr->be[i]->cir_vel[0] = float(rr->le->bmint->x[i] * M_PI * rr->des->revs/30.0);
		rr->be[i]->rot_abs[0] = 0.0; // no swirl
		// inlet blade angle, no swirl at inlet
		rr->be[i]->angle[0]	 = float(atan( rr->be[i]->mer_vel[0] / (rr->be[i]->cir_vel[0]) ));
		rr->be[i]->angle[0]	 = float(SHIFTANGLE(rr->be[i]->angle[0]));
		rr->be[i]->angle[0] += rr->be[i]->mod_angle[0];

		// trainling edge
		// conduit area for current blade element
		rr->be[i]->con_area[1]	 = InterpolConduitArea(rr->be[i]->area, rr->be[i]->ml,
													   rr->te->bmpar->list[i]);
		// meridional, circumferential vel. and rotational part of absolute vel.
		rr->be[i]->mer_vel[1]    = rr->des->dis/rr->be[i]->con_area[1];
		rr->be[i]->cir_vel[1]	 = float(rr->te->bmint->x[i] * M_PI*rr->des->revs/30.0);
		rr->be[i]->rot_abs[1]	 = (9.81f*rr->des->head + rr->be[i]->cir_vel[0]*rr->be[i]->rot_abs[0]) /
			(rr->be[i]->cir_vel[1]);
		// inlet blade angle
		rr->be[i]->angle[1]	 = float(atan( rr->be[i]->mer_vel[1] /( rr->be[i]->cir_vel[1]-rr->be[i]->rot_abs[1])));
		rr->be[i]->angle[1]	 = float(SHIFTANGLE(rr->be[i]->angle[1]));
		rr->be[i]->angle[1] -= rr->be[i]->mod_angle[1];
		if(rr->be[i]->angle[0] < rr->be[i]->angle[1]) err = EULER_ERR;

	}											   // end i, blade elements

	if(err == EULER_ERR) {
		sprintf(fn,"euler.err");
		if ( (fp = fopen(fn,"w+")) == NULL) {
			fprintf(stderr,"Cannot open file '%s'.\n",fn);
			errfile = 0;
			fp = stderr;
		}
		else errfile = 1;
	  
		fprintf(fp,"# %s (%d) err = %d\n", __FILE__,__LINE__,err);
		fprintf(fp,"# index param %15s%15s%15s%15s%15s%15s%15s%15s\n",
				"beta_1","beta_2","v_m1","u1","v_u1","v_m2","u2","v_u2");
		for(i = 0; i < rr->be_num; i++) {
			fprintf(fp,"%4d   %.4f   %14.5f %14.5f %14.5f %14.5f %14.5f %14.5f %14.5f %14.5f",
					i,rr->be[i]->para,
					rr->be[i]->angle[0]*180./M_PI,
					rr->be[i]->angle[1]*180./M_PI,
					rr->be[i]->mer_vel[0],
					rr->be[i]->cir_vel[0],rr->be[i]->rot_abs[0],
					rr->be[i]->mer_vel[1],
					rr->be[i]->cir_vel[1],rr->be[i]->rot_abs[1]);
			if(rr->be[i]->angle[0] < rr->be[i]->angle[1])
				fprintf(fp,"\t!!! ERROR !!!\n");

			else fprintf(fp,"\t ok\n");

		}
		if(errfile) fclose(fp);
	}


#ifdef DEBUG_AREAS
	sprintf(fn,"rr_lete_areas.txt");
	if ( (fp = fopen(fn,"w+")) == NULL) {
		fprintf(stderr,"Cannot open file '%s'.\n",fn);
		exit(1);
	}

	for(i=0; i<rr->be_num; i++) {
		fprintf(fp,"%f	%f\n", rr->le->bmint->x[i], rr->be[i]->con_area[0]);
	}
	fprintf(fp,"\n");
	for(i=0; i<rr->be_num; i++) {
		fprintf(fp,"%f	%f\n", rr->te->bmint->x[i], rr->be[i]->con_area[1]);
	}
	fclose(fp);
#endif

	return err;
}


// interpolation of conduit area at current intersection point
static float InterpolConduitArea(struct Flist *be_area, struct curve *ml,
								 float bmpar)
{
	int i;

	float area;

	area = 0.0;
	for(i=1; i<be_area->num; i++) {
		if( (ml->par[i-1] <= bmpar) &&
			(ml->par[i] >= bmpar)) {
			area  = be_area->list[i-1];
			area += ((be_area->list[i] - be_area->list[i-1]) /
					 (ml->par[i] - ml->par[i-1])) * (bmpar - ml->par[i-1]);
#ifdef DEBUG_AREAS
			fprintf(stderr,"i = %d, ml->pari-1 = %f, ml->pari = %f, areai-1 = %f, areai = %f\n",
					i,ml->par[i-1],ml->par[i],be_area->list[i-1],be_area->list[i]);
			fprintf(stderr,"\t\t i = %d, bmpar = %f, area = %f\n",
					i,bmpar,area);
#endif
			return(area);
		}

	}

	return (area);
}


float CalcSpecRevs(struct design *desi)
{
	if( (desi->head)<= 1.e-8) return -99.99f;
	return float( (desi -> revs * sqrt(desi -> dis)/pow((float)desi -> head, 0.75f)) );
}


static int CalcRR_InletMerVel(struct radial *rr)
{
	int i, ifirst;
	float v0;

	if(rr->des->vratio <= 0.0 || !rr->vratio_flag) return 0;

	ifirst = NPOIN_EXT-1;
#ifdef NO_INLET_EXT
	ifirst = 0;
#endif
	// only proper if be_num odd!!
	v0 = rr->des->dis/(rr->be[rr->be_num/2]->area->list[ifirst]*
					   (1.0f+(rr->des->vratio-1.0f)/3.0f));
	for(i = 0; i < rr->be_num; i++) {
		rr->be[i]->mer_vel[0] = v0*((rr->des->vratio-1.0f)*
			                    float(pow(rr->be[i]->para,2))+1);
	}

	return 0;
}
