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
#include "../General/include/bias.h"
#include "../General/include/flist.h"
#include "../General/include/parameter.h"
#include "../General/include/profile.h"
#include "../General/include/common.h"
#include "../General/include/log.h"
#include "../General/include/fatal.h"

int InitRR_BladeElements(struct radial *rr)
{
	int i;
	struct Flist *bias;

	bias = CalcBladeElementBias(rr->be_num, 0.0, 1.0, rr->be_type, rr->be_bias);
#ifdef DDEBUG
	DumpFlist(bias);
#endif						   // DDEBUG

#ifdef GAP
	// one more element for gap
	rr->be_num++;
	if ((rr->be = (struct be**)calloc(rr->be_num, sizeof(struct be*))) == NULL)
		fatal("memory for (struct be *)");
#else
	if ((rr->be = (struct be**)calloc(rr->be_num, sizeof(struct be*))) == NULL)
		fatal("memory for (struct be *)");
#endif						   // GAP
	for (i = 0; i < rr->be_num; i++) {
		if ((rr->be[i] = (struct be*)calloc(1,sizeof(struct be))) == NULL)
			fatal("memory for (struct be)");
		rr->be[i]->para = bias->list[i];
#ifdef GAP
		if(i == rr->be_num-1) {
			rr->be[i]->para	  = 1.0 + (2.0 * rr->gap) / (rr->cond[0] + rr->cond[1]);
		}
#endif
		// interpolate parameter set data to blade elements
		if(rr->iang->loc)
			rr->be[i]->angle[0]		= InterpolateParameterSet(rr->iang, rr->be[i]->para, rr->extrapol);
		if(rr->mod_iang->loc)
			rr->be[i]->mod_angle[0] = InterpolateParameterSet(rr->mod_iang, rr->be[i]->para, rr->extrapol);
		if(rr->oang->loc)
			rr->be[i]->angle[1]		= InterpolateParameterSet(rr->oang, rr->be[i]->para, rr->extrapol);
		if(rr->mod_oang->loc)
			rr->be[i]->mod_angle[1] = InterpolateParameterSet(rr->mod_oang, rr->be[i]->para, rr->extrapol);
		if(rr->orot_abs->loc)
			rr->be[i]->rot_abs[1]	= InterpolateParameterSet(rr->orot_abs, rr->be[i]->para, rr->extrapol);
		rr->be[i]->p_thick	= InterpolateParameterSet(rr->t, rr->be[i]->para, rr->extrapol);
		rr->be[i]->te_thick	= InterpolateParameterSet(rr->tet, rr->be[i]->para, rr->extrapol);
		rr->be[i]->te_wrap	= InterpolateParameterSet(rr->tewr, rr->be[i]->para, rr->extrapol);
		rr->be[i]->bl_wrap	= InterpolateParameterSet(rr->blwr, rr->be[i]->para, rr->extrapol);
		rr->be[i]->bp_shift	= InterpolateParameterSet(rr->bps, rr->be[i]->para, rr->extrapol);
		if(rr->le_para->loc)
			// calculate blade profile
			rr->be[i]->le_para		= InterpolateParameterSet(rr->le_para, rr->be[i]->para, rr->extrapol);
		if(rr->te_para->loc)
			// calculate blade profile
			rr->be[i]->te_para		= InterpolateParameterSet(rr->te_para, rr->be[i]->para, rr->extrapol);
		rr->be[i]->camb		= InterpolateParameterSet(rr->camb, rr->be[i]->para, rr->extrapol);
		rr->be[i]->camb_pos	= InterpolateParameterSet(rr->camb_pos, rr->be[i]->para, rr->extrapol);
		//fprintf(stderr,"\n WARNING-fl (%s, %d):\n",__FILE__,__LINE__);
		//fprintf(stderr,"//rr->be[i]->camb_para	   = InterpolateParameterSet(...)\n");
		if(rr->cambpara)
			rr->be[i]->camb_para	   = InterpolateParameterSet(rr->cambpara, rr->be[i]->para, rr->extrapol);
		if(rr->bl_lenpara)
			rr->be[i]->bl_lenpara	   = InterpolateParameterSet(rr->bl_lenpara, rr->be[i]->para, rr->extrapol);
		rr->be[i]->bp = AllocBladeProfile();
		AssignBladeProfile(rr->bp, rr->be[i]->bp);
		//ShiftBladeProfile(rr->be[i]->bp, rr->be[i]->bp_shift);
#ifdef INTERPOL_BLADE_PROFILE
		rr->be[i]->bp = InterpolBladeProfile(rr->be[i]->bp);
#endif						// INTERPOL_BLADE_PROFILE
#ifdef DEBUG_BLADE_PROFILE
		DumpBladeProfile(rr->be[i]->bp);
#endif						// DEBUG_BLADE_PROFILE
	}                       // end i
#ifdef GAP
	rr->be_num--;
	rr->gp = rr->be[rr->be_num];
#ifdef DEBUG
	fprintf(stderr,"rr->gp->para = %f\n",rr->gp->para);
	fprintf(stderr,"rr->be[rr->be_num]->para = %f\n",rr->be[rr->be_num]->para);
#endif
#endif						   // GAP
	rr->be_single = 1;
	return 0;
}
