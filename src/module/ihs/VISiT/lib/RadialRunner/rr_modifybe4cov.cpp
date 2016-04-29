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

#ifdef RADIAL_RUNNER
#include "include/radial.h"
#endif
#ifdef DIAGONAL_RUNNER
#include "include/diagonal.h"
#endif

extern void DetermineCoefficients(float *x, float *y, float *a);
extern float EvaluateParameter(float x, float *a);

int ModifyRR_BladeElements4Covise(struct radial *rr)
{
	int i;
	float a[3];					   // coefficients for parameter distribution, 2nd order
	float x[3], y[3];				   // locations and values to function
	// location of sliders; DO NOT CHANGE !!!
	const int left	 = 0;
	const int middle = (int)(rr->be_num / 2);
	const int right	 = rr->be_num - 1;

#ifdef GAP
	rr->be[rr->be_num]->para = 1.0 + (2.0 * rr->gap) / (rr->cond[0] + rr->cond[1]);
#endif
	if(rr->be_single) return 0;

	// slider parameter locations
	x[0] = rr->be[left]->para;
	x[1] = rr->be[middle]->para;
	x[2] = rr->be[right]->para;

	// inlet angle
	y[0]  = rr->be[left]->angle[0];
	y[1]  = rr->be[middle]->angle[0];
	y[2]  = rr->be[right]->angle[0];
	DetermineCoefficients(x,y,a);
	for (i = 0; i < rr->be_num; i++)
		rr->be[i]->angle[0] = EvaluateParameter(rr->be[i]->para, &a[0]);

	// outlet angle
	y[0] = rr->be[left]->angle[1];
	y[1] = rr->be[middle]->angle[1];
	y[2] = rr->be[right]->angle[1];
	DetermineCoefficients(x,y,a);
	for (i = 0; i < rr->be_num; i++)
		rr->be[i]->angle[1] = EvaluateParameter(rr->be[i]->para, &a[0]);

	// profile thickness
	y[0] = rr->be[left]->p_thick;
	y[1] = rr->be[middle]->p_thick;
	y[2] = rr->be[right]->p_thick;
	DetermineCoefficients(x,y,a);
	for (i = 0; i < rr->be_num; i++)
		rr->be[i]->p_thick = EvaluateParameter(rr->be[i]->para, &a[0]);

	// trailing edge thickness
	y[0] = rr->be[left]->te_thick;
	y[1] = rr->be[middle]->te_thick;
	y[2] = rr->be[right]->te_thick;
	DetermineCoefficients(x,y,a);
	for (i = 0; i < rr->be_num; i++)
		rr->be[i]->te_thick = EvaluateParameter(rr->be[i]->para, &a[0]);

	// centre line camber parameter
	y[0] = rr->be[left]->camb_para;
	y[1] = rr->be[middle]->camb_para;
	y[2] = rr->be[right]->camb_para;
	DetermineCoefficients(x,y,a);
	for (i = 0; i < rr->be_num; i++)
		rr->be[i]->camb_para = EvaluateParameter(rr->be[i]->para, &a[0]);

	// centre line camber
	y[0] = rr->be[left]->camb;
	y[1] = rr->be[middle]->camb;
	y[2] = rr->be[right]->camb;
	DetermineCoefficients(x,y,a);
	for (i = 0; i < rr->be_num; i++)
		rr->be[i]->camb = EvaluateParameter(rr->be[i]->para, &a[0]);

	// trailing edge wrap
	y[0] = rr->be[left]->te_wrap;
	y[1] = rr->be[middle]->te_wrap;
	y[2] = rr->be[right]->te_wrap;
	DetermineCoefficients(x,y,a);
	for (i = 0; i < rr->be_num; i++)
		rr->be[i]->te_wrap = EvaluateParameter(rr->be[i]->para, &a[0]);

	// blade wrap
	y[0] = rr->be[left]->bl_wrap;
	y[1] = rr->be[middle]->bl_wrap;
	y[2] = rr->be[right]->bl_wrap;
	DetermineCoefficients(x,y,a);
	for (i = 0; i < rr->be_num; i++)
		rr->be[i]->bl_wrap = EvaluateParameter(rr->be[i]->para, &a[0]);

	// profile shift
	y[0] = rr->be[left]->bp_shift;
	y[1] = rr->be[middle]->bp_shift;
	y[2] = rr->be[right]->bp_shift;
	DetermineCoefficients(x,y,a);
	for (i = 0; i < rr->be_num; i++)
		rr->be[i]->bp_shift = EvaluateParameter(rr->be[i]->para, &a[0]);

	// inlet angle modification
	y[0]  = rr->be[left]->mod_angle[0];
	y[1]  = rr->be[middle]->mod_angle[0];
	y[2]  = rr->be[right]->mod_angle[0];
	DetermineCoefficients(x,y,a);
	for (i = 0; i < rr->be_num; i++)
		rr->be[i]->mod_angle[0] = EvaluateParameter(rr->be[i]->para, &a[0]);

	// outlet angle modification
	y[0] = rr->be[left]->mod_angle[1];
	y[1] = rr->be[middle]->mod_angle[1];
	y[2] = rr->be[right]->mod_angle[1];
	DetermineCoefficients(x,y,a);
	for (i = 0; i < rr->be_num; i++)
		rr->be[i]->mod_angle[1] = EvaluateParameter(rr->be[i]->para, &a[0]);

	// centre line camber position
	y[0] = rr->be[left]->camb_pos;
	y[1] = rr->be[middle]->camb_pos;
	y[2] = rr->be[right]->camb_pos;
	DetermineCoefficients(x,y,a);
	for (i = 0; i < rr->be_num; i++)
		rr->be[i]->camb_pos = EvaluateParameter(rr->be[i]->para, &a[0]);

	// remaining swirl
	y[0] = rr->be[left]->rot_abs[1];
	y[1] = rr->be[middle]->rot_abs[1];
	y[2] = rr->be[right]->rot_abs[1];
	DetermineCoefficients(x,y,a);
	for (i = 0; i < rr->be_num; i++)
		rr->be[i]->rot_abs[1] = EvaluateParameter(rr->be[i]->para, &a[0]);

	// le spline paras
	y[0] = rr->be[left]->le_para;
	y[1] = rr->be[middle]->le_para;
	y[2] = rr->be[right]->le_para;
	DetermineCoefficients(x,y,a);
	for (i = 0; i < rr->be_num; i++)
		rr->be[i]->le_para = EvaluateParameter(rr->be[i]->para, &a[0]);

	// te spline paras
	y[0] = rr->be[left]->te_para;
	y[1] = rr->be[middle]->te_para;
	y[2] = rr->be[right]->te_para;
	DetermineCoefficients(x,y,a);
	for (i = 0; i < rr->be_num; i++)
		rr->be[i]->te_para = EvaluateParameter(rr->be[i]->para, &a[0]);

	// blade length factor
	y[0] = rr->be[left]->bl_lenpara;
	y[1] = rr->be[middle]->bl_lenpara;
	y[2] = rr->be[right]->bl_lenpara;
	DetermineCoefficients(x,y,a);
	for (i = 0; i < rr->be_num; i++)
		rr->be[i]->bl_lenpara = EvaluateParameter(rr->be[i]->para, &a[0]);

	return 0;
}
