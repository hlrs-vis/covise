// check and copy changed parametres.

#include "RadialRunner.h"
#include <General/include/log.h>

// **************************************************

// **************************************************
int RadialRunner::CheckUserInput(const char *portname, struct geometry *, struct rr_grid *)
{
	int changed;

	if(!geo) return 0;

	int ind, ibe, i;
	char pn[256];
	// slider locations; DO NOT CHANGE !!!
	const int left	 = 0;
	const int middle = int(geo->rr->be_num / 2);
	const int right	 = geo->rr->be_num - 1;
	// min/max values of parameters
	const float min_parm	=	0.0f;
	const float max_parm	=	1.0f;
	const float min_angle	= -10.0f;
	const float max_angle	=  90.0f;
	const float min_mod		= -45.0f;
	const float max_mod		=  45.0f;
	const float min_pthick	=	0.01f;
	const float max_pthick	=	0.20f;
	const float min_tethick =	0.00f;
	const float max_tethick =	0.05f;
	const float min_camb	=	0.0f;
	const float max_camb	=	2.0f;
	const float min_tewrap	= -10.0f;
	const float max_tewrap	=  10.0f;
	const float min_blwrap	=	0.0f;
	const float max_blwrap	= 360.0f;
	const float min_shift	=	0.0f;
	const float max_shift	=  10.0f;
	const float min_swirl	= -10.0f;
	const float max_swirl	=  10.0f;
	const float min_campar	= -2.0f;
	const float max_campar	=  3.0f;
	const float min_bllen	=  0.1f;
	const float max_bllen	=  8.1f;

	// !! Angles are always changed since geo->values are RAD and
	// p_VALUES are GRAD! Should be fixed for cleanlyness, F.L.!

	changed = 0;
	dprintf(2, "RadialRunner::CheckUserInput() entered... pn=%s\n", portname);
	if (SplitPortname(portname, pn, &ind)) {
		geo->rr->be_single = 1;
		// blade element data
		if (!strcmp(M_MERIDIAN_PARAMETER, pn)) {
			changed = CheckUserFloatValue(p_MeridianParm[ind], geo->rr->be[ind]->para,
										  min_parm, max_parm, &(geo->rr->be[ind]->para));
		}
		else if (!strcmp(M_INLET_ANGLE, pn)) {
			changed = CheckUserFloatValue(p_InletAngle[ind], geo->rr->be[ind]->angle[0],
										  min_angle, max_angle, &(geo->rr->be[ind]->angle[0]));
			geo->rr->be[ind]->angle[0] = RAD(geo->rr->be[ind]->angle[0]);
		}
		else if (!strcmp(M_OUTLET_ANGLE, pn)) {
			changed = CheckUserFloatValue(p_OutletAngle[ind], geo->rr->be[ind]->angle[1],
										  min_angle, max_angle, &(geo->rr->be[ind]->angle[1]));
			geo->rr->be[ind]->angle[1] = RAD(geo->rr->be[ind]->angle[1]);
		}
		else if (!strcmp(M_INLET_ANGLE_MODIFICATION, pn)) {
			changed = CheckUserFloatValue(p_InletAngleModification[ind], geo->rr->be[ind]->mod_angle[0],
										  min_mod, max_mod, &(geo->rr->be[ind]->mod_angle[0]));
			geo->rr->be[ind]->mod_angle[0] = RAD(geo->rr->be[ind]->mod_angle[0]);
		}
		else if (!strcmp(M_OUTLET_ANGLE_MODIFICATION, pn)) {
			changed = CheckUserFloatValue(p_OutletAngleModification[ind], geo->rr->be[ind]->mod_angle[1],
										  min_mod, max_mod, &(geo->rr->be[ind]->mod_angle[1]));
			geo->rr->be[ind]->mod_angle[1] = RAD(geo->rr->be[ind]->mod_angle[1]);
		}
		else if (!strcmp(M_REMAINING_SWIRL, pn)) {
			changed = CheckUserFloatValue(p_RemainingSwirl[ind], geo->rr->be[ind]->rot_abs[1],
										  RAD(min_swirl), RAD(max_swirl), &(geo->rr->be[ind]->rot_abs[1]));
		}
		else if (!strcmp(M_PROFILE_THICKNESS, pn)) {
			changed = CheckUserFloatValue(p_ProfileThickness[ind], geo->rr->be[ind]->p_thick,
										  min_pthick, max_pthick, &(geo->rr->be[ind]->p_thick));
		}
		else if (!strcmp(M_TE_THICKNESS, pn)) {
			changed = CheckUserFloatValue(p_TrailingEdgeThickness[ind], geo->rr->be[ind]->te_thick,
										  min_tethick, max_tethick, &(geo->rr->be[ind]->te_thick));
		}
		else if (!strcmp(M_CENTRE_LINE_CAMBER, pn)) {
			changed = CheckUserFloatValue(p_CentreLineCamber[ind], geo->rr->be[ind]->camb,
										  min_camb, max_camb, &(geo->rr->be[ind]->camb));
		}
		else if (!strcmp(M_CENTRE_LINE_CAMBER_POSN, pn)) {
			changed = CheckUserFloatValue(p_CentreLineCamberPosn[ind], geo->rr->be[ind]->camb_pos,
										  min_camb, max_camb, &(geo->rr->be[ind]->camb_pos));
		}
		else if (!strcmp(M_CAMBPARA, pn)) {
			changed = CheckUserFloatValue(p_CambPara[ind], geo->rr->be[ind]->camb_para,
										  min_campar, max_campar, &(geo->rr->be[ind]->camb_para));
		}
		else if (!strcmp(M_BLADE_LENGTH_FACTOR, pn)) {
			changed = CheckUserFloatValue(p_BladeLengthFactor[ind], geo->rr->be[ind]->bl_lenpara,
										  min_bllen, max_bllen, &(geo->rr->be[ind]->bl_lenpara));
		}
		else if (!strcmp(M_TE_WRAP_ANGLE, pn)) {
			changed = CheckUserFloatValue(p_TrailingEdgeWrap[ind], geo->rr->be[ind]->te_wrap,
										  min_tewrap, max_tewrap, &(geo->rr->be[ind]->te_wrap));
			geo->rr->be[ind]->te_wrap = RAD(geo->rr->be[ind]->te_wrap);
		}
		else if (!strcmp(M_BL_WRAP_ANGLE, pn)) {
			changed = CheckUserFloatValue(p_BladeWrap[ind], geo->rr->be[ind]->bl_wrap,
										  min_blwrap, max_blwrap, &(geo->rr->be[ind]->bl_wrap));
			geo->rr->be[ind]->bl_wrap = RAD(geo->rr->be[ind]->bl_wrap);
		}
		else if (!strcmp(M_PROFILE_SHIFT, pn)) {
			changed = CheckUserFloatValue(p_ProfileShift[ind], geo->rr->be[ind]->bp_shift,
										  min_shift, max_shift, &(geo->rr->be[ind]->bp_shift));
		}
		else if (!strcmp(M_BLADE_LESPLINE_PARAS, pn)) {
			changed = CheckUserFloatValue(p_BladeLePara[ind],
										  geo->rr->be[ind]->le_para,
										  0.0,1.0,
										  &(geo->rr->be[ind]->le_para));
		}
		else if (!strcmp(M_BLADE_TESPLINE_PARAS, pn)) {
			changed = CheckUserFloatValue(p_BladeTePara[ind],
										  geo->rr->be[ind]->te_para,
										  0.0,1.0,
										  &(geo->rr->be[ind]->te_para));
		}
		else if (!strcmp(M_LEFT_POINT, pn)) {
			geo->rr->be_single=0;
			ibe = left;
			if (ind == 0) {
				changed = CheckUserFloatSliderValue(p_HubPoint[ind], geo->rr->be[ibe]->angle[0],
													min_angle, max_angle, &(geo->rr->be[ibe]->angle[0]));
				geo->rr->be[ibe]->angle[0] = RAD(geo->rr->be[ibe]->angle[0]);
			}
			if (ind == 1) {
				changed = CheckUserFloatSliderValue(p_HubPoint[ind], geo->rr->be[ibe]->angle[1],
													min_angle, max_angle, &(geo->rr->be[ibe]->angle[1]));
				geo->rr->be[ibe]->angle[1] = RAD(geo->rr->be[ibe]->angle[1]);
			}
			if (ind == 2)
				changed = CheckUserFloatSliderValue(p_HubPoint[ind], geo->rr->be[ibe]->p_thick,
													min_pthick, max_pthick, &(geo->rr->be[ibe]->p_thick));
			if (ind == 3)
				changed = CheckUserFloatSliderValue(p_HubPoint[ind], geo->rr->be[ibe]->te_thick,
													min_tethick, max_tethick, &(geo->rr->be[ibe]->te_thick));
			if (ind == 4) {
				changed = CheckUserFloatSliderValue(p_HubPoint[ind], geo->rr->be[ibe]->te_wrap,
													min_tewrap, max_tewrap, &(geo->rr->be[ibe]->te_wrap));
				geo->rr->be[ibe]->te_wrap = RAD(geo->rr->be[ibe]->te_wrap);
			}
			if (ind == 5) {
				changed = CheckUserFloatSliderValue(p_HubPoint[ind], geo->rr->be[ibe]->bl_wrap,
													min_blwrap, max_blwrap, &(geo->rr->be[ibe]->bl_wrap));
				geo->rr->be[ibe]->bl_wrap = RAD(geo->rr->be[ibe]->bl_wrap);
			}
			if (ind == 6)
				changed = CheckUserFloatSliderValue(p_HubPoint[ind], geo->rr->be[ibe]->bp_shift,
													min_shift, max_shift, &(geo->rr->be[ibe]->bp_shift));
			if (ind == 7) {
				changed = CheckUserFloatSliderValue(p_HubPoint[ind], geo->rr->be[ibe]->mod_angle[0],
													min_mod, max_mod, &(geo->rr->be[ibe]->mod_angle[0]));
				geo->rr->be[ibe]->mod_angle[0] = RAD(geo->rr->be[ibe]->mod_angle[0]);
			}
			if (ind == 8) {
				changed = CheckUserFloatSliderValue(p_HubPoint[ind], geo->rr->be[ibe]->mod_angle[1],
													min_mod, max_mod, &(geo->rr->be[ibe]->mod_angle[1]));
				geo->rr->be[ibe]->mod_angle[1] = RAD(geo->rr->be[ibe]->mod_angle[1]);
			}
			if (ind == 9)
				changed = CheckUserFloatSliderValue(p_HubPoint[ind], geo->rr->be[ibe]->rot_abs[1],
													min_swirl, max_swirl, &(geo->rr->be[ibe]->rot_abs[1]));
			if (ind == 10)
				changed = CheckUserFloatSliderValue(p_HubPoint[ind],
													geo->rr->be[ibe]->le_para,
													1.e-4f,1.0f,
													&(geo->rr->be[ibe]->le_para));
			if (ind == 11)
				changed = CheckUserFloatSliderValue(p_HubPoint[ind],
													geo->rr->be[ibe]->te_para,
													1.e-4f,1.0f,
													&(geo->rr->be[ibe]->te_para));
			if (ind == 12)
				changed = CheckUserFloatSliderValue(p_HubPoint[ind], geo->rr->be[ibe]->camb,
													min_camb, max_camb, &(geo->rr->be[ibe]->camb));
			if (ind == 13)
				changed = CheckUserFloatSliderValue(p_HubPoint[ind], geo->rr->be[ibe]->camb_pos,
													min_camb, max_camb, &(geo->rr->be[ibe]->camb_pos));
			if (ind == 14)
				changed = CheckUserFloatSliderValue(p_HubPoint[ind], geo->rr->be[ibe]->camb_para,
													min_campar, max_campar, &(geo->rr->be[ibe]->camb_para));
			if (ind == 15)
				changed = CheckUserFloatSliderValue(p_HubPoint[ind], geo->rr->be[ibe]->bl_lenpara,
													min_bllen, max_bllen, &(geo->rr->be[ibe]->bl_lenpara));
		}
		else if (!strcmp(M_MIDDLE_POINT, pn)) {
			geo->rr->be_single=0;
			ibe = middle;
			if (ind == 0) {
				changed = CheckUserFloatSliderValue(p_InnerPoint[ind], geo->rr->be[ibe]->angle[0],
													min_angle, max_angle, &(geo->rr->be[ibe]->angle[0]));
				geo->rr->be[ibe]->angle[0] = RAD(geo->rr->be[ibe]->angle[0]);
			}
			if (ind == 1) {
				changed = CheckUserFloatSliderValue(p_InnerPoint[ind], geo->rr->be[ibe]->angle[1],
													min_angle, max_angle, &(geo->rr->be[ibe]->angle[1]));
				geo->rr->be[ibe]->angle[1] = RAD(geo->rr->be[ibe]->angle[1]);
			}
			if (ind == 2)
				changed = CheckUserFloatSliderValue(p_InnerPoint[ind], geo->rr->be[ibe]->p_thick,
													min_pthick, max_pthick, &(geo->rr->be[ibe]->p_thick));
			if (ind == 3)
				changed = CheckUserFloatSliderValue(p_InnerPoint[ind], geo->rr->be[ibe]->te_thick,
													min_tethick, max_tethick, &(geo->rr->be[ibe]->te_thick));
			if (ind == 4) {
				changed = CheckUserFloatSliderValue(p_InnerPoint[ind], geo->rr->be[ibe]->te_wrap,
													min_tewrap, max_tewrap, &(geo->rr->be[ibe]->te_wrap));
				geo->rr->be[ibe]->te_wrap = RAD(geo->rr->be[ibe]->te_wrap);
			}
			if (ind == 5) {
				changed = CheckUserFloatSliderValue(p_InnerPoint[ind], geo->rr->be[ibe]->bl_wrap,
													min_blwrap, max_blwrap, &(geo->rr->be[ibe]->bl_wrap));
				geo->rr->be[ibe]->bl_wrap = RAD(geo->rr->be[ibe]->bl_wrap);
			}
			if (ind == 6)
				changed = CheckUserFloatSliderValue(p_InnerPoint[ind], geo->rr->be[ibe]->bp_shift,
													min_shift, max_shift, &(geo->rr->be[ibe]->bp_shift));
			if (ind == 7) {
				changed = CheckUserFloatSliderValue(p_InnerPoint[ind], geo->rr->be[ibe]->mod_angle[0],
													min_mod, max_mod, &(geo->rr->be[ibe]->mod_angle[0]));
				geo->rr->be[ibe]->mod_angle[0] = RAD(geo->rr->be[ibe]->mod_angle[0]);
			}
			if (ind == 8) {
				changed = CheckUserFloatSliderValue(p_InnerPoint[ind], geo->rr->be[ibe]->mod_angle[1],
													min_mod, max_mod, &(geo->rr->be[ibe]->mod_angle[1]));
				geo->rr->be[ibe]->mod_angle[1] = RAD(geo->rr->be[ibe]->mod_angle[1]);
			}
			if (ind == 9)
				changed = CheckUserFloatSliderValue(p_InnerPoint[ind], geo->rr->be[ibe]->rot_abs[1],
													min_swirl, max_swirl, &(geo->rr->be[ibe]->rot_abs[1]));
			if (ind == 10)
				changed = CheckUserFloatSliderValue(p_InnerPoint[ind],
													geo->rr->be[ibe]->le_para,
													1.e-4f,1.0f,
													&(geo->rr->be[ibe]->le_para));
			if (ind == 11)
				changed = CheckUserFloatSliderValue(p_InnerPoint[ind],
													geo->rr->be[ibe]->te_para,
													1.e-4f,1.0f,
													&(geo->rr->be[ibe]->te_para));
			if (ind == 12)
				changed = CheckUserFloatSliderValue(p_InnerPoint[ind], geo->rr->be[ibe]->camb,
													min_camb, max_camb, &(geo->rr->be[ibe]->camb));
			if (ind == 13)
				changed = CheckUserFloatSliderValue(p_InnerPoint[ind], geo->rr->be[ibe]->camb_pos,
													min_camb, max_camb, &(geo->rr->be[ibe]->camb_pos));
			if (ind == 14)
				changed = CheckUserFloatSliderValue(p_InnerPoint[ind], geo->rr->be[ibe]->camb_para,
													min_campar, max_campar, &(geo->rr->be[ibe]->camb_para));
			if (ind == 15)
				changed = CheckUserFloatSliderValue(p_InnerPoint[ind], geo->rr->be[ibe]->bl_lenpara,
													min_bllen, max_bllen, &(geo->rr->be[ibe]->bl_lenpara));
		}
		else if (!strcmp(M_RIGHT_POINT, pn)) {
			geo->rr->be_single=0;
			ibe = right;
			if (ind == 0) {
				changed = CheckUserFloatSliderValue(p_ShroudPoint[ind], geo->rr->be[ibe]->angle[0],
													min_angle, max_angle, &(geo->rr->be[ibe]->angle[0]));
				geo->rr->be[ibe]->angle[0] = RAD(geo->rr->be[ibe]->angle[0]);
			}
			if (ind == 1) {
				changed = CheckUserFloatSliderValue(p_ShroudPoint[ind], geo->rr->be[ibe]->angle[1],
													min_angle, max_angle, &(geo->rr->be[ibe]->angle[1]));
				geo->rr->be[ibe]->angle[1] = RAD(geo->rr->be[ibe]->angle[1]);
			}
			if (ind == 2)
				changed = CheckUserFloatSliderValue(p_ShroudPoint[ind], geo->rr->be[ibe]->p_thick,
													min_pthick, max_pthick, &(geo->rr->be[ibe]->p_thick));
			if (ind == 3)
				changed = CheckUserFloatSliderValue(p_ShroudPoint[ind], geo->rr->be[ibe]->te_thick,
													min_tethick, max_tethick, &(geo->rr->be[ibe]->te_thick));
			if (ind == 4) {
				changed = CheckUserFloatSliderValue(p_ShroudPoint[ind], geo->rr->be[ibe]->te_wrap,
													min_tewrap, max_tewrap, &(geo->rr->be[ibe]->te_wrap));
				geo->rr->be[ibe]->te_wrap = RAD(geo->rr->be[ibe]->te_wrap);
			}
			if (ind == 5) {
				changed = CheckUserFloatSliderValue(p_ShroudPoint[ind], geo->rr->be[ibe]->bl_wrap,
													min_blwrap, max_blwrap, &(geo->rr->be[ibe]->bl_wrap));
				geo->rr->be[ibe]->bl_wrap = RAD(geo->rr->be[ibe]->bl_wrap);
			}
			if (ind == 6)
				changed = CheckUserFloatSliderValue(p_ShroudPoint[ind], geo->rr->be[ibe]->bp_shift,
													min_shift, max_shift, &(geo->rr->be[ibe]->bp_shift));
			if (ind == 7) {
				changed = CheckUserFloatSliderValue(p_ShroudPoint[ind], geo->rr->be[ibe]->mod_angle[0],
													min_mod, max_mod, &(geo->rr->be[ibe]->mod_angle[0]));
				geo->rr->be[ibe]->mod_angle[0] = RAD(geo->rr->be[ibe]->mod_angle[0]);
			}
			if (ind == 8) {
				changed = CheckUserFloatSliderValue(p_ShroudPoint[ind], geo->rr->be[ibe]->mod_angle[1],
													min_mod, max_mod, &(geo->rr->be[ibe]->mod_angle[1]));
				geo->rr->be[ibe]->mod_angle[1] = RAD(geo->rr->be[ibe]->mod_angle[1]);
			}
			if (ind == 9)
				changed = CheckUserFloatSliderValue(p_ShroudPoint[ind], geo->rr->be[ibe]->rot_abs[1],
													min_swirl, max_swirl, &(geo->rr->be[ibe]->rot_abs[1]));
			if (ind == 10)
				changed = CheckUserFloatSliderValue(p_ShroudPoint[ind],
													geo->rr->be[ibe]->le_para,
													1.e-4f,1.0f,
													&(geo->rr->be[ibe]->le_para));
			if (ind == 11)
				changed = CheckUserFloatSliderValue(p_ShroudPoint[ind],
													geo->rr->be[ibe]->te_para,
													1.e-4f,1.0f,
													&(geo->rr->be[ibe]->te_para));
			if (ind == 12)
				changed = CheckUserFloatSliderValue(p_ShroudPoint[ind], geo->rr->be[ibe]->camb,
													min_camb, max_camb, &(geo->rr->be[ibe]->camb));
			if (ind == 13)
				changed = CheckUserFloatSliderValue(p_ShroudPoint[ind], geo->rr->be[ibe]->camb_pos,
													min_camb, max_camb, &(geo->rr->be[ibe]->camb_pos));
			if (ind == 14)
				changed = CheckUserFloatSliderValue(p_ShroudPoint[ind], geo->rr->be[ibe]->camb_para,
													min_campar, max_campar, &(geo->rr->be[ibe]->camb_para));
			if (ind == 15)
				changed = CheckUserFloatSliderValue(p_ShroudPoint[ind], geo->rr->be[ibe]->bl_lenpara,
													min_bllen, max_bllen, &(geo->rr->be[ibe]->bl_lenpara));
		}
		else if ( (!strcmp(M_2DPORT, pn)) || (!strcmp(M_2DPLOT, pn))) {
			// do nothing.
		}
		else {
			dprintf(2, "Sorry, no function for %s implemented\n", pn);
			changed = 0;
		}
	}
	else {
		// global runner data
		if (!strcmp(M_OUTLET_DIAMETER_ABS, pn)) {
			changed = CheckUserFloatValue(p_OutletDiameterAbs, geo->rr->ref,
										  0.1f, 10.0f, &(geo->rr->ref));
		}
		else if (!strcmp(M_NUMBER_OF_BLADES, pn)) {
			changed = CheckUserIntValue(p_NumberOfBlades, geo->rr->nob,
										2,21, &(geo->rr->nob));
		}
		else if (!strcmp(M_INLET_DIAMETER_REL, pn)) {
			changed = CheckUserFloatValue(p_InletDiameterRel, geo->rr->diam[0],
										  0.2f,10.0f, &(geo->rr->diam[0]));
		}
		else if (!strcmp(M_SHROUD_HEIGHT_DIFF, pn)) {
			changed = CheckUserFloatValue(p_ShroudHeightDiff, geo->rr->height,
										  0.0f,1.0f, &(geo->rr->height));
		}
#ifdef GAP
		else if (!strcmp(M_GAP_WIDTH, pn)) {
			changed = CheckUserFloatValue(p_GapWidth, geo->rr->gap,
										  0.0001,0.5, &(geo->rr->gap));
		}
#endif
		else if (!strcmp(M_CONDUIT_WIDTH, pn)) {
			changed = CheckUserFloatVectorValue(p_ConduitWidth,
												geo->rr->cond,
												0.0f,0.5f,
												geo->rr->cond,2);
		}

		else if (!strcmp(M_CONTOUR_ANGLES, pn)) {
			if((changed = CheckUserFloatVectorValue(p_ContourAngles,
													geo->rr->angle,
													-90.0f,270.0f,
													geo->rr->angle,2))) {
				geo->rr->angle[0] =
					RAD(p_ContourAngles->getValue(0));
				geo->rr->angle[1] =
					RAD(p_ContourAngles->getValue(1));
			}
			else {
				p_ContourAngles->setValue(0,GRAD(geo->rr->angle[0]));
				p_ContourAngles->setValue(1,GRAD(geo->rr->angle[1]));
			}
		}
		else if (!strcmp(M_INLET_OPEN_ANGLES, pn)) {
			if((changed = CheckUserFloatVectorValue(p_InletOpenAngles,
													geo->rr->iop_angle,
													-90.0f,180.0f,
													geo->rr->iop_angle,2))) {
				geo->rr->iop_angle[0] =
					RAD(p_InletOpenAngles->getValue(0));
				geo->rr->iop_angle[1] =
					RAD(p_InletOpenAngles->getValue(1));
			}
			else {
				p_InletOpenAngles->setValue(0,GRAD(geo->rr->iop_angle[0]));
				p_InletOpenAngles->setValue(1,GRAD(geo->rr->iop_angle[1]));
			}
		}
		else if (!strcmp(M_OUTLET_OPEN_ANGLES, pn)) {
			if((changed = CheckUserFloatVectorValue(p_OutletOpenAngles,
													geo->rr->oop_angle,
													-90.0f,180.0f,
													geo->rr->oop_angle,2))) {
				geo->rr->oop_angle[0] =
					RAD(p_OutletOpenAngles->getValue(0));
				geo->rr->oop_angle[1] =
					RAD(p_OutletOpenAngles->getValue(1));
			}
			else {
				p_OutletOpenAngles->setValue(0,GRAD(geo->rr->oop_angle[0]));
				p_OutletOpenAngles->setValue(1,GRAD(geo->rr->oop_angle[1]));
			}
		}
		else if (!strcmp(M_HUB_CURVE_PARAMETERS, pn)) {
			changed = CheckUserFloatVectorValue(p_HubCurveParameters, geo->rr->hspara,
												0.000f, 1.0f,geo->rr->hspara,2);
		}
		else if (!strcmp(M_SHROUD_CURVE_PARAMETERS, pn)) {
			changed = CheckUserFloatVectorValue(p_ShroudCurveParameters, geo->rr->sspara,
												0.000f, 1.0f,geo->rr->sspara,2);
		}
		else if (!strcmp(M_HUB_STRAIGHT_PARAMETERS, pn)) {
			changed = CheckUserFloatVectorValue(p_HubStraightParameters, geo->rr->hstparam,
												0.000f, 1.0f,geo->rr->hstparam,2);
		}
		else if (!strcmp(M_SHROUD_STRAIGHT_PARAMETERS, pn)) {
			changed = CheckUserFloatVectorValue(p_ShroudStraightParameters, geo->rr->sstparam,
												0.000f, 1.0f,geo->rr->sstparam,2);
		}

		// extended meridian contour data
		else if (!strcmp(M_INLET_ANGLE_EXT, pn)) {
			if((changed = CheckUserFloatValue(p_InletAngleExt, geo->rr->ext_iangle,-90.0,180.0, &(geo->rr->ext_iangle))))
				geo->rr->ext_iangle = RAD(geo->rr->ext_iangle);
			else p_InletAngleExt->setValue(GRAD(geo->rr->ext_iangle));
		}
		else if (!strcmp(M_HEIGHT_EXT, pn)) {
			changed = CheckUserFloatVectorValue(p_HeightExt,
												geo->rr->ext_height,
												-0.1f,4.0f,
												geo->rr->ext_height,2);
		}
		else if (!strcmp(M_DIAM_EXT, pn)) {
			changed = CheckUserFloatVectorValue(p_DiamExt,
												geo->rr->ext_diam,
												-0.1f,4.0f,
												geo->rr->ext_diam,2);
		}
		else if (!strcmp(M_WIDTH_EXT, pn)) {
			changed = CheckUserFloatVectorValue(p_WidthExt,
												geo->rr->ext_cond,
												-0.1f,4.0f,
												geo->rr->ext_cond,2);
		}
		else if (!strcmp(M_HUB_CURVE_PEXT, pn)) {
			changed = CheckUserFloatVectorValue(p_HubCurveParaExt, geo->rr->hspara_inext,
												-0.0001f, 1.0f,geo->rr->hspara_inext,2);
		}
		else if (!strcmp(M_SHROUD_CURVE_PEXT, pn)) {
			changed = CheckUserFloatVectorValue(p_ShroudCurveParaExt, geo->rr->sspara_inext,
												-0.0001f, 1.0f,geo->rr->sspara_inext,2);
		}

		// design data
		else if (!strcmp(M_DESIGN_Q,pn)) {
			changed = CheckUserFloatValue(p_DDischarge,
										  geo->rr->des->dis,
										  0.0,1.0e+6,
										  &(geo->rr->des->dis));
			if(changed && rrg) {
				rrg->inbc->bcQ = geo->rr->des->dis;
				p_bcQ->setValue(geo->rr->des->dis);
			}
		}
		else if (!strcmp(M_DESIGN_H,pn)) {
			changed = CheckUserFloatValue(p_DHead,
										  geo->rr->des->head,
										  0.0,1.0e+6,
										  &(geo->rr->des->head));
			if(changed && rrg) {
				rrg->inbc->bcH = geo->rr->des->head;
				p_bcH->setValue(geo->rr->des->head);
			}
		}
		else if (!strcmp(M_DESIGN_N,pn)) {
			changed = CheckUserFloatValue(p_DRevolut,
										  geo->rr->des->revs,
										  0.0,1.0e+6,
										  &(geo->rr->des->revs));
			if(changed && rrg) {
				rrg->inbc->bcN = geo->rr->des->revs;
			}
		}
		else if (!strcmp(M_INLET_VRATIO,pn))
			changed = CheckUserFloatValue(p_DVRatio,
										  geo->rr->des->vratio,
										  0.0,1.0e+3,
										  &(geo->rr->des->vratio));
		else if(!strcmp(M_DEFINE_VRATIO,pn)) {
			changed = 1;
			geo->rr->vratio_flag =
				(int)(p_DDefineVRatio->getValue());
		}

		// blade edge data, le
		else if (!strcmp(M_LE_HUB_PARM, pn)) {
			changed = CheckUserFloatValue(p_LeHubParm, geo->rr->le->para[0],
										  min_parm, max_parm, &(geo->rr->le->para[0]));
		}
		else if (!strcmp(M_LE_SHROUD_PARM, pn)) {
			changed = CheckUserFloatValue(p_LeShroudParm, geo->rr->le->para[1],
										  min_parm, max_parm, &(geo->rr->le->para[1]));
		}
		else if (!strcmp(M_LE_HUB_ANGLE, pn)) {
			if((changed = CheckUserFloatValue(p_LeHubAngle, geo->rr->le->angle[0],-90.0, 180.0, &(geo->rr->le->angle[0]))))
				geo->rr->le->angle[0] = RAD(geo->rr->le->angle[0]);
			else p_LeHubAngle->setValue(GRAD(geo->rr->le->angle[0]));
		}
		else if (!strcmp(M_LE_SHROUD_ANGLE, pn)) {
			if((changed = CheckUserFloatValue(p_LeShroudAngle, geo->rr->le->angle[1],-90.0, 180.0, &(geo->rr->le->angle[1]))))
				geo->rr->le->angle[1] = RAD(geo->rr->le->angle[1]);
			else p_LeShroudAngle->setValue(GRAD(geo->rr->le->angle[1]));
		}
		else if (!strcmp(M_LE_CURVE_PARAMETER, pn)) {
			changed = CheckUserFloatVectorValue(p_LeCurveParam, geo->rr->le->spara,
												0.0001f, 0.999f,geo->rr->le->spara,2);
		}
		// te
		else if (!strcmp(M_TE_HUB_PARM, pn)) {
			changed = CheckUserFloatValue(p_TeHubParm, geo->rr->te->para[0],
										  min_parm, max_parm, &(geo->rr->te->para[0]));
		}
		else if (!strcmp(M_TE_SHROUD_PARM, pn)) {
			changed = CheckUserFloatValue(p_TeShroudParm, geo->rr->te->para[1],
										  min_parm, max_parm, &(geo->rr->te->para[1]));
		}
		else if (!strcmp(M_TE_HUB_ANGLE, pn)) {
			if((changed = CheckUserFloatValue(p_TeHubAngle, geo->rr->te->angle[0],-90.0, 180.0, &(geo->rr->te->angle[0]))))
				geo->rr->te->angle[0] = RAD(geo->rr->te->angle[0]);
			else p_TeHubAngle->setValue(GRAD(geo->rr->te->angle[0]));
		}
		else if (!strcmp(M_TE_SHROUD_ANGLE, pn)) {
			if((changed = CheckUserFloatValue(p_TeShroudAngle, geo->rr->te->angle[1],-90.0, 180.0, &(geo->rr->te->angle[1]))))
				geo->rr->te->angle[1] = RAD(geo->rr->te->angle[1]);
			else p_TeShroudAngle->setValue(GRAD(geo->rr->te->angle[1]));
		}
		else if (!strcmp(M_TE_CURVE_PARAMETER, pn)) {
			changed = CheckUserFloatVectorValue(p_TeCurveParam, geo->rr->te->spara,
												0.0001f, 0.999f,geo->rr->te->spara,2);
		}
		// blade element specifications
#ifdef MODIFY_NOB
		// first make sure that all bes are cleanly deallocated and
		// thoroughly re-created in CreateRR_BladeElements (rr_comp.c)
		// NOT THE CASE YET!!! Still has to be done!!!! F.L. 9/2003.
		dprintf(2,"MODIFY_NOB is activated in the Makefile\n");
		dprintf(2,"first make sure that all bes are cleanly\n");
		dprintf(2,"deallocated and thoroughly re-created in\n");
		dprintf(2,"CreateRR_BladeElements (rr_comp.c)\n");
		dprintf(2,"Do NOT forget a clean interpolation to\n");
		dprintf(2,"from existing to new blade elements\n");
		dprintf(2,"src: %s (%d)\n",__FILE__,__LINE__);
		sendError("Check Source Code first!! See 'stderr'.");
		exit(1);
		else if (!strcmp(M_NUMBER_OF_BLADE_ELEMENTS, pn)) {
			changed = CheckUserIntValue(p_NumberOfBladeElements, geo->rr->be_num,
										5, MAX_ELEMENTS, &(geo->rr->be_num));
		}
#endif										// MODIFY_NOB
		else if (!strcmp(M_BLADE_BIAS_FACTOR, pn)) {
			changed = CheckUserFloatValue(p_BladeElementBiasFactor, geo->rr->be_bias,
										  -20.0, 20.0, &(geo->rr->be_bias));
		}
		else if (!strcmp(M_BLADE_BIAS_TYPE, pn)) {
			changed = CheckUserIntValue(p_BladeElementBiasType, geo->rr->be_type,
										0, 2, &(geo->rr->be_type));
		}
		else if(!strcmp(M_EULER_EQN,pn)) {
			changed = 1;
			geo->rr->euler = (int)(p_EulerEqn->getValue());
		}
		else if(!strcmp(M_PUMP,pn)) {
			changed = 1;
			geo->rr->pump = (int)(p_Pump->getValue());
		}
		else if(!strcmp(M_ROTATE_CLOCKWISE,pn)) {
			changed = 1;
			geo->rr->rot_clockwise =
				(int)(p_RotateClockwise->getValue());
		}
		else if(!strcmp(M_CAMBER_POS,pn)) {
			changed = 1;
			geo->rr->camb_flag = (int)(p_CamberPos->getValue());
		}
		else if(!strcmp(M_CAMB2SURF,pn)) {
			changed = 1;
			geo->rr->camb2surf = p_Camb2Surf->getValue() + 1;
		}
		else if(!strcmp(M_SHOW_EXTENSION,pn)) {
			changed = 1;
			geo->rr->showExt = (int)(p_ShowExtensions->getValue());
		}
		else if(!strcmp(M_EXTENDED_MENU,pn)) {
			changed = 0;
		}
		else if(!strcmp(M_STRAIGHT_HUB,pn)) {
			changed = 1;
			geo->rr->straight_cont[0] =
				(int)(p_StraightHub->getValue());
		}
		else if(!strcmp(M_STRAIGHT_SHRD,pn)) {
			changed = 1;
			geo->rr->straight_cont[1] =
				(int)(p_StraightShrd->getValue());
		}
		// ***** Grid parameters, only if rrg exists!!!
		else if(!strcmp("makeGrid",pn));
		else if(!strcmp("writeGrid",pn) && rrg)
			rrg->write_grid = (int)p_writeGrid->getValue();
		else if(!strcmp(M_CREATE_BC,pn) && rrg)
			rrg->create_inbc = (int)p_createBC->getValue();
		else if(!strcmp(M_MESH_EXT,pn) && rrg) {
			rrg->mesh_ext = (int)p_meshExt->getValue();
			dprintf(2,"rrg->mesh_ext = %d\n",rrg->mesh_ext);
		}
		else if(!strcmp(M_ROT_EXT,pn) && rrg)
			rrg->rot_ext = (int)p_rotExt->getValue();
		else if(!strcmp(M_USE_Q,pn) && rrg)
			rrg->inbc->useQ = (int)p_useQ->getValue();
		else if(!strcmp(M_USE_ALPHA,pn) && rrg)
			rrg->inbc->useAlpha = (int)p_useAlpha->getValue();
		else if(!strcmp(M_BCQ,pn) && rrg)
			changed = CheckUserFloatValue(p_bcQ,
										  rrg->inbc->bcQ,0.0,
										  1.0e+8,
										  &(rrg->inbc->bcQ));
		else if(!strcmp(M_BCH,pn) && rrg)
			changed = CheckUserFloatValue(p_bcH,
										  rrg->inbc->bcH,0.0,
										  1.0e+8,
										  &(rrg->inbc->bcH));
		else if(!strcmp(M_BCALPHA,pn) && rrg) {
			if((changed = CheckUserFloatValue(p_bcAlpha,
											  GRAD(rrg->inbc->bcAlpha),0.0,
											  180.0,
											  &(rrg->inbc->bcAlpha))))
				rrg->inbc->bcAlpha =
					RAD(p_bcAlpha->getValue());
		}
		else if(!strcmp(M_SHOW_COMPLETE_GRID,pn)) changed = 1;
		else if(!strcmp(M_LAYERS2SHOW,pn) && rrg) {
			if(p_GridLayers->getValue(1) < p_GridLayers->getValue(0)) {
				sendError(" max value < min value!");
				p_GridLayers->setValue(1,p_GridLayers->getValue(0));
				if(p_GridLayers->getValue(1) > rrg->ge_num-1) {
					sendError(" max layer-to-show value  > than number of layers!");
					p_GridLayers->setValue(0, rrg->ge_num-1);
					p_GridLayers->setValue(1, rrg->ge_num-1);
				}
			}
			else if(p_GridLayers->getValue(1) > rrg->ge_num-1) {
				sendError(" max layer-to-show value  > than number of layers!");
				p_GridLayers->setValue(1, rrg->ge_num-1);
			}
			else if(p_GridLayers->getValue(0) < 1) {
				sendError(" min layer-to-show value < than ZERO!");
				p_GridLayers->setValue(0,1);
			}
			changed = 1;
		}
		else if(!strcmp(M_GRID_MERIDS,pn) && rrg)
			changed = CheckDiscretization(p_GridMerids, &rrg->ge_num,
										  &rrg->ge_bias, &rrg->ge_type);
		else if(!strcmp(M_CIRCUMF_DIS,pn) && rrg) {
			changed = CheckDiscretization(p_CircumfDis, &rrg->cdis,
										  &rrg->cbias, &rrg->cbias_type);
		}
		else if(!strcmp(M_CIRCUMF_DIS_LE,pn) && rrg) {
			changed = CheckDiscretization(p_CircumfDisLe, &rrg->cledis,
										  &rrg->clebias, &rrg->clebias_type);
		}
		else if(!strcmp(M_MERID_INLET,pn) && rrg)
			changed = CheckDiscretization(p_MeridInletDis, &rrg->ssmdis,
										  &rrg->ssmbias, &rrg->ssmbias_type);
		else if(!strcmp(M_PS_DIS,pn) && rrg) {
			if(p_PSDis->getValue(0) >= rrg->ssdis) {
				sendError(" Value for '%s(0)' must be smaller than '%s(0)'! Last change ignored!",
						   M_PS_DIS,M_SS_DIS);
				p_PSDis->setValue(0,rrg->psdis);
			}
			changed = CheckDiscretization(p_PSDis, &rrg->psdis,
										  &rrg->psbias, &rrg->psbias_type);
		}
		else if(!strcmp(M_SS_DIS,pn) && rrg) {
			if(p_SSDis->getValue(0) <= rrg->psdis) {
				sendError(" Value for '%s(0)' must be bigger than '%s(0)'! Last change ignored!",
						   M_SS_DIS,M_PS_DIS);
				p_SSDis->setValue(0,rrg->ssdis);
			}
			changed = CheckDiscretization(p_SSDis, &rrg->ssdis,
										  &rrg->ssbias, &rrg->ssbias_type);
		}
		else if(!strcmp(M_BL_DIS,pn) && rrg)
			changed = CheckDiscretization(p_BLDis, &rrg->psedis,
										  &rrg->psebias, &rrg->psebias_type);
		else if(!strcmp(M_MERID_OUTLET,pn) && rrg)
			changed = CheckDiscretization(p_MeridOutletDis, &rrg->lowdis,
										  &rrg->lowbias, &rrg->lowbias_type);
#ifndef NO_INLET_EXT
		else if(!strcmp(M_MERID_INEXT,pn) && rrg)
			changed = CheckDiscretization(p_MeridInExtDis, &rrg->extdis,
										  &rrg->extbias, &rrg->extbias_type);
#endif
		else if(!strcmp(M_SKEW_INLET,pn) && rrg)
			rrg->skew_runin = (int)p_SkewInlet->getValue();
		else if(!strcmp(M_PHI_SCALE,pn) && rrg) {
			changed = CheckUserFloatVectorValue(p_PhiScale, rrg->phi_scale,
												-2.0,2.0,rrg->phi_scale,2);
		}

		else if(!strcmp(M_PHI_SKEW,pn) && rrg) {
			changed = CheckUserFloatVectorValue(p_PhiSkew, rrg->phi_skew,
												-10.0,10.0,rrg->phi_skew,2);
		}
		else if(!strcmp(M_PHI_SKEWOUT,pn) && rrg) {
			changed = CheckUserFloatVectorValue(p_PhiSkewOut, rrg->phi_skewout,
												-10.0,10.0,rrg->phi_skewout,2);
		}
		else if(!strcmp(M_BOUND_LAY_RATIO,pn) && rrg) {
			rrg->bl_scale[0]  = 1.0-p_BoundLayRatio->getValue(0);
			rrg->bl_scale[1]  = 1.0-p_BoundLayRatio->getValue(1);
		}
		else if(!strcmp(M_V14_ANGLE,pn) && rrg) {
			for(i = 0; i < 2; i++) {
				if((changed = CheckUserFloatVectorValue2(p_V14Angle, rrg->v14_angle[i],
														 0.0,90.0,&rrg->v14_angle[i],i)))
					rrg->v14_angle[i] = RAD(rrg->v14_angle[i]);
				else p_V14Angle->setValue(i,GRAD(rrg->v14_angle[i]));
			}
		}
		else if(!strcmp(M_BLV14_PART,pn) && rrg) {
			changed = CheckUserFloatVectorValue(p_BlV14Part, rrg->bl_v14_part,
												0.0f,1.0f,rrg->bl_v14_part,2);
		}
		else if(!strcmp(M_SS_PART,pn) && rrg) {
			changed = CheckUserFloatVectorValue(p_SSPart, rrg->ss_part,
												0.01f,0.99f,rrg->ss_part,2);
		}
		else if(!strcmp(M_PS_PART,pn) && rrg) {
			changed = CheckUserFloatVectorValue(p_PSPart, rrg->ps_part,
												0.01f,0.99f,rrg->ps_part,2);
		}
		else if(!strcmp(M_SSLE_PART,pn) && rrg) {
			changed = CheckUserFloatVectorValue(p_SSLePart,
												rrg->ssle_part,
												0.00001f,0.1f,
												rrg->ssle_part,2);
		}
		else if(!strcmp(M_PSLE_PART,pn) && rrg) {
			changed = CheckUserFloatVectorValue(p_PSLePart,
												rrg->psle_part,
												0.00001f,0.1f,
												rrg->psle_part,2);
		}
		else if(!strcmp(M_OUT_PART,pn) && rrg) {
			if((changed = CheckUserFloatVectorValue2(p_OutPart, rrg->out_part[0],
													 0.01f,0.49f,&rrg->out_part[0],0))) {
				rrg->out_part[1] = 1.0f - rrg->out_part[0];
				p_OutPart->setValue(1,rrg->out_part[1]);
			}
			else {
				if((changed = CheckUserFloatVectorValue2(p_OutPart, rrg->out_part[1],
														 0.51f,0.99f,&rrg->out_part[1],1))) {
					rrg->out_part[0] = 1.0f - rrg->out_part[1];
					p_OutPart->setValue(0,rrg->out_part[0]);
				}
			}
		}
		else if(!strcmp(M_GRIDTYPE,pn) && rrg) {
			if(rrg) {
				changed = CheckUserChoiceValue(p_GridTypeChoice, 2);
				dprintf(2," p_GridTypeChoice = %d\n", p_GridTypeChoice->getValue());
				rrg->type = p_GridTypeChoice->getValue() + 1;
			}
			else
				sendWarning(" Grid does not exist yet! Selection has no effect!");
		}
#ifndef NO_INLET_EXT
		else if(!strcmp(M_PHI_EXT,pn) && rrg)
			changed = CheckUserFloatValue(p_PhiExtScale, rrg->phi0_ext,
										  -2.0f,2.0f,&rrg->phi0_ext);
		else if(!strcmp(M_RRIN_ANGLE,pn) && rrg) {
			for(i = 0; i < 2; i++) {
				if((changed = CheckUserFloatVectorValue2(p_RRInAngle, rrg->angle_ext[i],
														 0.0f,90.0f,&rrg->angle_ext[i],i)))
					rrg->angle_ext[i] = RAD(rrg->angle_ext[i]);
				else p_RRInAngle->setValue(i,GRAD(rrg->angle_ext[i]));
			}
		}
#endif
		else {
			dprintf(2, "Sorry, no function for '%s' implemented\n",
					pn);
			changed = 0;
		}
		// really ugly to put this here. But if put to CreateMenuRunnerData() the extended
		// parameters are shown after the initialization of the menu even if hidden!
		if(m_types->getValue() == 1)
			if(p_DDefineVRatio->getValue()) p_DVRatio->show();
			else p_DVRatio->hide();
		else p_DVRatio->hide();
		dprintf(2," CheckUserInput(): m_types = %d\n",
				m_types->getValue());

		if(m_types->getValue() == 3)
			RadialRunner::ShowHideExtended(p_ExtendedMenu->getValue());
		else
			RadialRunner::ShowHideExtended(0);
		if(m_types->getValue() == 7) {
			if(m_paraset->getValue() == 2)
				RadialRunner::ShowHideModifiedOptions(p_GridTypeChoice->getValue() + 1);
			else
				RadialRunner::ShowHideModifiedOptions(0);
			if(p_ShowComplete->getValue()) p_GridLayers->hide();
			else p_GridLayers->show();
		}
		else {
			RadialRunner::ShowHideModifiedOptions(0);
			p_GridLayers->hide();
		}
	}
	return changed;
}

