// create the menues

//#include <config/CoviseConfig.h>
//#include <stdlib.h>
//#include <stdio.h>
//#include <string.h>
//#include <errno.h>
#include "RadialRunner.h"

void RadialRunner::CreateUserMenu(void)
{
	dprintf(2,"CreateUserMenu():numReducedMenuPoints=%d\n",numReducedMenuPoints);
	m_types = paraSwitch("types", "Select_type_of_parameters");
	paraCase(M_DESIGN_DATA);
	RadialRunner::CreateMenuDesignData();
	paraEndCase();								   // end of M_DESIGN_DATA

	paraCase(M_SPECIALS);
	RadialRunner::CreateMenuSpecials();
	paraEndCase();								   // end of M_SPECIALS

	paraCase(M_RUNNER_DATA);
	RadialRunner::CreateMenuRunnerData();
	paraEndCase();								   // end of M_RUNNER_DATA

#ifdef CREATE_PROFILE_MENU
	paraCase(M_BLADE_PROFILE_DATA);
	RadialRunner::CreateMenuProfileData();
	paraEndCase();								   // end of M_BLADE_PROFILE_DATA
#endif										   // CREATE_PROFILE_MENU

	paraCase(M_LEADING_EDGE_DATA);
	RadialRunner::CreateMenuLeadingEdge();
	paraEndCase();								   // end of M_LEADING_EDGE

	paraCase(M_TRAILING_EDGE_DATA);
	RadialRunner::CreateMenuTrailingEdge();
	paraEndCase();								   // end of M_TRAILING_EDGE

	paraCase(M_BLADE_DATA);
	RadialRunner::CreateMenuBladeData();
	paraEndCase();								   // end of M_BLADE_DATA

	paraCase(M_GRID_DATA);
	RadialRunner::CreateMenuGridData();
	paraEndCase();								   // end of M_GRID_DATA
	paraEndSwitch();							   // end of "types"
}


void RadialRunner::CreatePortMenu(void)
{
	int i;
	char name[200];

	paraSwitch(M_2DPORT, "Select_plot_port");
	for(i = 1; i <= NUM_PLOT_PORTS; i++) {
		sprintf(name,"%s%d","port_no._",i);
		paraCase(name);
		RadialRunner::CreatePlotPortMenu(i);
		dprintf(3,"name: %s\n",name);
		paraEndCase();								// end of M_MERIDIAN_CONTOUR_PLOT
	}
	paraEndSwitch();							   // end of "2DplotPort1"

}


void RadialRunner::CreatePlotPortMenu(int p)
{
	char name[110];

	sprintf(name,"%s_%d",M_2DPLOT,p);
	m_2DplotChoice[p-1] = paraSwitch(name, "Select_plot_data");
	paraCase(M_MERIDIAN_CONTOUR_PLOT);
	paraEndCase();								   // end of M_MERIDIAN_CONTOUR_PLOT
	paraCase(M_MERIDIAN_CONTOUR_EXT);
	paraEndCase();								   // end of M_MERIDIAN_CONTOUR_PLOT
	paraCase(M_CONFORMAL_VIEW);
	RadialRunner::CreateMenuConformalView(p);
	paraEndCase();								   // end of M_CONFORMAL_VIEW
	paraCase(M_CAMBER);
	RadialRunner::CreateMenuCamber(p);
	paraEndCase();
	paraCase(M_NORMCAMBER);
	RadialRunner::CreateMenuNormCamber(p);
	paraEndCase();
	paraCase(M_THICKNESS);
	paraEndCase();
	paraCase(M_OVERLAP);
	paraEndCase();
	paraCase(M_BLADE_ANGLES);
	paraEndCase();
	paraCase(M_EULER_ANGLES);
	paraEndCase();
	paraCase(M_MERIDIAN_VELOCITY);
	paraEndCase();
	paraCase(M_CIRCUMF_VELOCITY);
	paraEndCase();

	paraEndSwitch();							   // end of "2DplotPort"

}


void RadialRunner::CreateMenuConformalView(int p)
{
	int i;
	char buf[200];

	for(i = 0; i < MAX_ELEMENTS; i++) {
		sprintf(buf,"%s_%d_%d",M_SHOW_CONFORMAL,p,i+1);
		dprintf(3,"buf = %s\n",buf);
		p_ShowConformal[i][p-1] = addBooleanParam(buf,buf);
		p_ShowConformal[i][p-1]->setValue(0);
	}
}


void RadialRunner::CreateMenuCamber(int p)
{
	int i;
	char buf[200];

	for(i = 0; i < MAX_ELEMENTS; i++) {
		sprintf(buf,"%s_%d_%d",M_SHOW_CAMBER,p,i+1);
		dprintf(3,"buf = %s\n",buf);
		p_ShowCamber[i][p-1] = addBooleanParam(buf,buf);
		p_ShowCamber[i][p-1]->setValue(0);
	}
}


void RadialRunner::CreateMenuNormCamber(int p)
{
	int i;
	char buf[200];

	for(i = 0; i < MAX_ELEMENTS; i++) {
		sprintf(buf,"%s_%d_%d",M_SHOW_NORMCAMBER,p,i+1);
		dprintf(2,"buf = %s\n",buf);
		p_ShowNormCamber[i][p-1] = addBooleanParam(buf,buf);
		p_ShowNormCamber[i][p-1]->setValue(0);
	}
}


#ifdef CREATE_PROFILE_MENU
void RadialRunner::CreateMenuProfileData(void)
{
}
#endif											  // CREATE_PROFILE_MENU

void RadialRunner::CreateMenuDesignData(void)
{
	p_DDischarge = addFloatParam(M_DESIGN_Q,M_DESIGN_Q);
	p_DDischarge->setValue(0.0);

	p_DHead		 = addFloatParam(M_DESIGN_H,M_DESIGN_H);
	p_DHead->setValue(0.0);

	p_DRevolut	 = addFloatParam(M_DESIGN_N,M_DESIGN_N);
	p_DRevolut->setValue(0.0);

	p_DDefineVRatio = addBooleanParam(M_DEFINE_VRATIO,M_DEFINE_VRATIO);
	p_DDefineVRatio->setValue(0);

	p_DVRatio	= addFloatParam(M_INLET_VRATIO,M_INLET_VRATIO);
	p_DVRatio->setValue(0.0);
	p_DVRatio->hide();
}


void RadialRunner::CreateMenuSpecials(void)
{
	const char *cambfuncs[NUM_CAMBFUNCS] = {M_CAMB_RESULT,M_CAMB_TEFIX,
									  M_CAMB_LEFIX,M_CAMB_CLLEN};

	p_EulerEqn = addBooleanParam(M_EULER_EQN,M_EULER_EQN);
	p_EulerEqn->setValue(0);

	p_Pump = addBooleanParam(M_PUMP,M_PUMP);
	p_Pump->setValue(0);

	p_WriteBladeData = addBooleanParam(M_WRITE_BLADEDATA,
									   M_WRITE_BLADEDATA);
	p_WriteBladeData->setValue(0);

	p_RotateClockwise = addBooleanParam(M_ROTATE_CLOCKWISE,
										M_ROTATE_CLOCKWISE);
	p_RotateClockwise->setValue(0);

	p_CamberPos = addBooleanParam(M_CAMBER_POS,M_CAMBER_POS);
	p_CamberPos->setValue(0);

	p_ShowExtensions = addBooleanParam(M_SHOW_EXTENSION,M_SHOW_EXTENSION);
	p_ShowExtensions->setValue(0);

	p_Camb2Surf = addChoiceParam(M_CAMB2SURF,M_CAMB2SURF);
	p_Camb2Surf->setValue(NUM_CAMBFUNCS,cambfuncs,0);

}

void RadialRunner::CreateMenuRunnerData(void)
{
	float init_val[] = {0,0};

	p_ExtendedMenu	= addBooleanParam(M_EXTENDED_MENU,M_EXTENDED_MENU);
	p_ExtendedMenu->setValue(0);

	p_StraightHub  = addBooleanParam(M_STRAIGHT_HUB,M_STRAIGHT_HUB);
	p_StraightHub->setValue(0);

	p_StraightShrd	= addBooleanParam(M_STRAIGHT_SHRD,M_STRAIGHT_SHRD);
	p_StraightShrd->setValue(0);

	p_NumberOfBlades = addInt32Param(M_NUMBER_OF_BLADES,M_NUMBER_OF_BLADES);
	p_NumberOfBlades->setValue(0);

	p_OutletDiameterAbs = addFloatParam(M_OUTLET_DIAMETER_ABS,M_OUTLET_DIAMETER_ABS);
	p_OutletDiameterAbs->setValue(0.0);

	p_InletDiameterRel = addFloatParam(M_INLET_DIAMETER_REL,M_INLET_DIAMETER_REL);
	p_InletDiameterRel->setValue(0.0);

	p_ShroudHeightDiff = addFloatParam(M_SHROUD_HEIGHT_DIFF,M_SHROUD_HEIGHT_DIFF);
	p_ShroudHeightDiff->setValue(0.0);

#ifdef GAP
	p_GapWidth = addFloatParam(M_GAP_WIDTH,M_GAP_WIDTH);
	p_GapWidth->setValue(0.0);
#endif

	p_ConduitWidth = addFloatVectorParam(M_CONDUIT_WIDTH,M_CONDUIT_WIDTH);
	p_ConduitWidth->setValue(2,init_val);

	p_ContourAngles = addFloatVectorParam(M_CONTOUR_ANGLES,M_CONTOUR_ANGLES);
	p_ContourAngles->setValue(2,init_val);

	p_InletOpenAngles = addFloatVectorParam(M_INLET_OPEN_ANGLES,M_INLET_OPEN_ANGLES);
	p_InletOpenAngles->setValue(2,init_val);

	p_OutletOpenAngles = addFloatVectorParam(M_OUTLET_OPEN_ANGLES,M_OUTLET_OPEN_ANGLES);
	p_OutletOpenAngles->setValue(2,init_val);

	p_HubCurveParameters = addFloatVectorParam(M_HUB_CURVE_PARAMETERS, M_HUB_CURVE_PARAMETERS);
	p_HubCurveParameters->setValue(2, init_val);

	p_ShroudCurveParameters = addFloatVectorParam(M_SHROUD_CURVE_PARAMETERS, M_SHROUD_CURVE_PARAMETERS);
	p_ShroudCurveParameters->setValue(2, init_val);

	p_HubStraightParameters = addFloatVectorParam(M_HUB_STRAIGHT_PARAMETERS, M_HUB_STRAIGHT_PARAMETERS);
	p_HubStraightParameters->setValue(2, init_val);

	p_ShroudStraightParameters = addFloatVectorParam(M_SHROUD_STRAIGHT_PARAMETERS, M_SHROUD_STRAIGHT_PARAMETERS);
	p_ShroudStraightParameters->setValue(2, init_val);

	// extended menu
	p_InletAngleExt = addFloatParam(M_INLET_ANGLE_EXT,M_INLET_ANGLE_EXT);
	p_InletAngleExt->setValue(0.0);

	p_HeightExt = addFloatVectorParam(M_HEIGHT_EXT,M_HEIGHT_EXT);
	p_HeightExt->setValue(2,init_val);

	p_DiamExt = addFloatVectorParam(M_DIAM_EXT,M_DIAM_EXT);
	p_DiamExt->setValue(2,init_val);

	p_WidthExt = addFloatVectorParam(M_WIDTH_EXT,M_WIDTH_EXT);
	p_WidthExt->setValue(2,init_val);

	p_HubCurveParaExt = addFloatVectorParam(M_HUB_CURVE_PEXT,M_HUB_CURVE_PEXT);
	p_HubCurveParaExt->setValue(2, init_val);

	p_ShroudCurveParaExt = addFloatVectorParam(M_SHROUD_CURVE_PEXT,M_SHROUD_CURVE_PEXT);
	p_ShroudCurveParaExt->setValue(2, init_val);
}


void RadialRunner::CreateMenuLeadingEdge(void)
{
	float init_val[] = {0,0};

	p_LeHubParm = addFloatParam(M_LE_HUB_PARM, M_LE_HUB_PARM);
	p_LeHubParm->setValue(0.0);

	p_LeHubAngle = addFloatParam(M_LE_HUB_ANGLE, M_LE_HUB_ANGLE);
	p_LeHubAngle->setValue(0.0);

	p_LeShroudParm = addFloatParam(M_LE_SHROUD_PARM, M_LE_SHROUD_PARM);
	p_LeShroudParm->setValue(0.0);

	p_LeShroudAngle = addFloatParam(M_LE_SHROUD_ANGLE, M_LE_SHROUD_ANGLE);
	p_LeShroudAngle->setValue(0.0);

	p_LeCurveParam = addFloatVectorParam(M_LE_CURVE_PARAMETER, M_LE_CURVE_PARAMETER);
	p_LeCurveParam->setValue(2,init_val);
}


void RadialRunner::CreateMenuTrailingEdge(void)
{
	float init_val[] = {0,0};

	p_TeHubParm = addFloatParam(M_TE_HUB_PARM, M_TE_HUB_PARM);
	p_TeHubParm->setValue(0.0);

	p_TeHubAngle = addFloatParam(M_TE_HUB_ANGLE, M_TE_HUB_ANGLE);
	p_TeHubAngle->setValue(0.0);

	p_TeShroudParm = addFloatParam(M_TE_SHROUD_PARM, M_TE_SHROUD_PARM);
	p_TeShroudParm->setValue(0.0);

	p_TeShroudAngle = addFloatParam(M_TE_SHROUD_ANGLE, M_TE_SHROUD_ANGLE);
	p_TeShroudAngle->setValue(0.0);

	p_TeCurveParam = addFloatVectorParam(M_TE_CURVE_PARAMETER, M_TE_CURVE_PARAMETER);
	p_TeCurveParam->setValue(2,init_val);
}


void RadialRunner::ReducedModifyMenu(void)
{
	int i;
	char *buf;

	paraSwitch(M_REDUCED_MODIFY, M_REDUCED_MODIFY);
	for (i = 0; i < numReducedMenuPoints; i++) {
		paraCase(ReducedModifyMenuPoints[i]);

		buf = IndexedParameterName(M_LEFT_POINT, i);
		p_HubPoint[i] = addFloatSliderParam(buf, buf);
		p_HubPoint[i]->setValue(0.0);
		p_HubPoint[i]->disable();
		free(buf);

		buf = IndexedParameterName(M_MIDDLE_POINT, i);
		p_InnerPoint[i] = addFloatSliderParam(buf, buf);
		p_InnerPoint[i]->setValue(0.0);
		p_InnerPoint[i]->disable();
		free(buf);

		buf = IndexedParameterName(M_RIGHT_POINT, i);
		p_ShroudPoint[i] = addFloatSliderParam(buf, buf);
		p_ShroudPoint[i]->setValue(0.0);
		p_ShroudPoint[i]->disable();
		free(buf);

		paraEndCase();
	}
	paraEndSwitch();
}


void RadialRunner::CreateMenuBladeData(void)
{
	int i;
	char *buf;

	p_NumberOfBladeElements = addInt32Param(M_NUMBER_OF_BLADE_ELEMENTS,
												M_NUMBER_OF_BLADE_ELEMENTS);
	p_NumberOfBladeElements->setValue(0);

	p_BladeElementBiasFactor = addFloatParam(M_BLADE_BIAS_FACTOR,
												   M_BLADE_BIAS_FACTOR);
	p_BladeElementBiasFactor->setValue(0.0);

	p_BladeElementBiasType = addInt32Param(M_BLADE_BIAS_TYPE,
											   M_BLADE_BIAS_TYPE);
	p_BladeElementBiasType->setValue(0);

	paraSwitch(M_BLADE_ELEMENT_DATA, M_BLADE_ELEMENT_DATA);
	for (i = 0; i < MAX_ELEMENTS; i++) {
		paraCase(IndexedParameterName(M_BLADE_ELEMENT_DATA, i));

		buf = IndexedParameterName(M_MERIDIAN_PARAMETER, i);
		p_MeridianParm[i] = addFloatParam(buf, buf);
		p_MeridianParm[i]->setValue(0.0);
		p_MeridianParm[i]->disable();
		free(buf);

		buf = IndexedParameterName(M_INLET_ANGLE, i);
		p_InletAngle[i] = addFloatParam(buf, buf);
		p_InletAngle[i]->setValue(0.0);
		p_InletAngle[i]->disable();
		free(buf);

		buf = IndexedParameterName(M_OUTLET_ANGLE, i);
		p_OutletAngle[i] = addFloatParam(buf, buf);
		p_OutletAngle[i]->setValue(0.0);
		p_OutletAngle[i]->disable();
		free(buf);

		buf = IndexedParameterName(M_PROFILE_THICKNESS, i);
		p_ProfileThickness[i] = addFloatParam(buf, buf);
		p_ProfileThickness[i]->setValue(0.0);
		p_ProfileThickness[i]->disable();
		free(buf);

		buf = IndexedParameterName(M_TE_THICKNESS, i);
		p_TrailingEdgeThickness[i] = addFloatParam(buf, buf);
		p_TrailingEdgeThickness[i]->setValue(0.0);
		p_TrailingEdgeThickness[i]->disable();
		free(buf);

		buf = IndexedParameterName(M_TE_WRAP_ANGLE, i);
		p_TrailingEdgeWrap[i] = addFloatParam(buf, buf);
		p_TrailingEdgeWrap[i]->setValue(0.0);
		p_TrailingEdgeWrap[i]->disable();
		free(buf);

		buf = IndexedParameterName(M_BL_WRAP_ANGLE, i);
		p_BladeWrap[i] = addFloatParam(buf, buf);
		p_BladeWrap[i]->setValue(0.0);
		p_BladeWrap[i]->disable();
		free(buf);

		buf = IndexedParameterName(M_PROFILE_SHIFT, i);
		p_ProfileShift[i] = addFloatParam(buf, buf);
		p_ProfileShift[i]->setValue(0.0);
		p_ProfileShift[i]->disable();
		free(buf);

		buf = IndexedParameterName(M_INLET_ANGLE_MODIFICATION, i);
		p_InletAngleModification[i] = addFloatParam(buf, buf);
		p_InletAngleModification[i]->setValue(0.0);
		p_InletAngleModification[i]->disable();
		free(buf);

		buf = IndexedParameterName(M_OUTLET_ANGLE_MODIFICATION, i);
		p_OutletAngleModification[i] = addFloatParam(buf, buf);
		p_OutletAngleModification[i]->setValue(0.0);
		p_OutletAngleModification[i]->disable();
		free(buf);

		buf = IndexedParameterName(M_REMAINING_SWIRL, i);
		p_RemainingSwirl[i] = addFloatParam(buf, buf);
		p_RemainingSwirl[i]->setValue(0.0);
		p_RemainingSwirl[i]->disable();
		free(buf);

		buf = IndexedParameterName(M_BLADE_LESPLINE_PARAS, i);
		p_BladeLePara[i] = addFloatParam(buf, buf);
		p_BladeLePara[i]->setValue(0.0);
		p_BladeLePara[i]->disable();
		free(buf);

		buf = IndexedParameterName(M_BLADE_TESPLINE_PARAS, i);
		p_BladeTePara[i] = addFloatParam(buf, buf);
		p_BladeTePara[i]->setValue(0.0);
		p_BladeTePara[i]->disable();
		free(buf);

		buf = IndexedParameterName(M_CENTRE_LINE_CAMBER, i);
		p_CentreLineCamber[i] = addFloatParam(buf, buf);
		p_CentreLineCamber[i]->setValue(0.0);
		p_CentreLineCamber[i]->disable();
		free(buf);

		buf = IndexedParameterName(M_CENTRE_LINE_CAMBER_POSN, i);
		p_CentreLineCamberPosn[i] = addFloatParam(buf, buf);
		p_CentreLineCamberPosn[i]->setValue(0.0);
		p_CentreLineCamberPosn[i]->disable();
		free(buf);

		buf = IndexedParameterName(M_CAMBPARA, i);
		p_CambPara[i] = addFloatParam(buf, buf);
		p_CambPara[i]->setValue(0.0);
		p_CambPara[i]->disable();
		free(buf);

		buf = IndexedParameterName(M_BLADE_LENGTH_FACTOR, i);
		p_BladeLengthFactor[i] = addFloatParam(buf, buf);
		p_BladeLengthFactor[i]->setValue(0.0);
		p_BladeLengthFactor[i]->disable();
		free(buf);

		paraEndCase();
	}
	paraEndSwitch();
	ReducedModifyMenu();
}


void RadialRunner::CreateMenuGridData(void)
{
	const char *gridtypes[2] = {M_B2B_CLASSIC,M_B2B_MODIFIED};
#ifndef YAC
	long int init_val[] = {1,1};
#else
	int init_val[] = {1,1};
#endif

	// Grid will be created, when button is pushed ...
	p_makeGrid = addBooleanParam("makeGrid","Make_a_grid_now");
	p_makeGrid->setValue(0);

	p_GridTypeChoice = addChoiceParam(M_GRIDTYPE,M_GRIDTYPE);
	p_GridTypeChoice->setValue(2,gridtypes,0);

	p_ShowComplete = addBooleanParam(M_SHOW_COMPLETE_GRID,M_SHOW_COMPLETE_GRID);
	p_ShowComplete->setValue(1);

	p_GridLayers = addInt32VectorParam(M_LAYERS2SHOW,M_LAYERS2SHOW);
	p_GridLayers->setValue(2,init_val);
	p_GridLayers->hide();

	m_paraset = paraSwitch(M_GRID_PARA_SELECT, M_GRID_PARA_SELECT);
	paraCase(M_BASIC_GRID);
	RadialRunner::CreateMenuBasicGrid();
	paraEndCase();
	paraCase(M_GRID_TOPOLOGY);
	RadialRunner::CreateMenuGridTopo();
	paraEndCase();
	paraCase(M_BC_AND_CFD);
	RadialRunner::CreateMenuBCandCFD();
	paraEndCase();
	paraEndSwitch();

}


void RadialRunner::CreateMenuBasicGrid(void)
{
	p_GridMerids = addFloatVectorParam(M_GRID_MERIDS, M_GRID_MERIDS);
	p_GridMerids->setValue(0,0,0);

	p_CircumfDis = addFloatVectorParam(M_CIRCUMF_DIS,M_CIRCUMF_DIS);
	p_CircumfDis->setValue(0,0,0);

	p_CircumfDisLe = addFloatVectorParam(M_CIRCUMF_DIS_LE,M_CIRCUMF_DIS_LE);
	p_CircumfDisLe->setValue(0,0,0);

	p_MeridInletDis = addFloatVectorParam(M_MERID_INLET,M_MERID_INLET);
	p_MeridInletDis->setValue(0,0,0);

	p_PSDis = addFloatVectorParam(M_PS_DIS,M_PS_DIS);
	p_PSDis->setValue(0,0,0);

	p_SSDis = addFloatVectorParam(M_SS_DIS,M_SS_DIS);
	p_SSDis->setValue(0,0,0);

	p_BLDis = addFloatVectorParam(M_BL_DIS,M_BL_DIS);
	p_BLDis->setValue(0,0,0);

	p_MeridOutletDis = addFloatVectorParam(M_MERID_OUTLET,M_MERID_OUTLET);
	p_MeridOutletDis->setValue(0,0,0);

#ifndef NO_INLET_EXT
	p_MeridInExtDis = addFloatVectorParam(M_MERID_INEXT,M_MERID_INEXT);
	p_MeridInExtDis->setValue(0,0,0);
#endif

}


void RadialRunner::CreateMenuBCandCFD(void)
{
	p_writeGrid = addBooleanParam("writeGrid",
								  "Write_grid_and_boco_to_file");
	p_writeGrid->setValue(0);

	p_createBC = addBooleanParam(M_CREATE_BC,M_CREATE_BC);
	p_createBC->setValue(0);

	p_meshExt = addBooleanParam(M_MESH_EXT,M_MESH_EXT);
	p_meshExt->setValue(0);

	p_rotExt = addBooleanParam(M_ROT_EXT,M_ROT_EXT);
	p_rotExt->setValue(0);

	p_bcQ = addFloatParam(M_BCQ,M_BCQ);
	p_bcQ->setValue(0.0);

	p_useQ = addBooleanParam(M_USE_Q,M_USE_Q);
	p_useQ->setValue(0);

	p_bcH = addFloatParam(M_BCH,M_BCH);
	p_bcH->setValue(0.0);

	p_bcAlpha = addFloatParam(M_BCALPHA,M_BCALPHA);
	p_bcAlpha->setValue(0.0);

	p_useAlpha = addBooleanParam(M_USE_ALPHA,M_USE_ALPHA);
	p_useAlpha->setValue(0);
}


void RadialRunner::CreateMenuGridTopo(void)
{
	float init_val[] = {0,0};

	p_SkewInlet = addBooleanParam(M_SKEW_INLET,M_SKEW_INLET);
	p_SkewInlet->setValue(0);

	p_PhiScale = addFloatVectorParam(M_PHI_SCALE, M_PHI_SCALE);
	p_PhiScale->setValue(2,init_val);

	p_PhiSkew = addFloatVectorParam(M_PHI_SKEW, M_PHI_SKEW);
	p_PhiSkew->setValue(2,init_val);

	p_PhiSkewOut = addFloatVectorParam(M_PHI_SKEWOUT, M_PHI_SKEWOUT);
	p_PhiSkewOut->setValue(2,init_val);

	p_BoundLayRatio = addFloatVectorParam(M_BOUND_LAY_RATIO,M_BOUND_LAY_RATIO);
	p_BoundLayRatio->setValue(2,init_val);

	p_V14Angle = addFloatVectorParam(M_V14_ANGLE, M_V14_ANGLE);
	p_V14Angle->setValue(2,init_val);

	p_BlV14Part = addFloatVectorParam(M_BLV14_PART, M_BLV14_PART);
	p_BlV14Part->setValue(2,init_val);

	p_SSPart = addFloatVectorParam(M_SS_PART, M_SS_PART);
	p_SSPart->setValue(2,init_val);

	p_PSPart = addFloatVectorParam(M_PS_PART, M_PS_PART);
	p_PSPart->setValue(2,init_val);

	p_SSLePart = addFloatVectorParam(M_SSLE_PART, M_SSLE_PART);
	p_SSLePart->setValue(2,init_val);

	p_PSLePart = addFloatVectorParam(M_PSLE_PART, M_PSLE_PART);
	p_PSLePart->setValue(2,init_val);

	p_OutPart = addFloatVectorParam(M_OUT_PART, M_OUT_PART);
	p_OutPart->setValue(2,init_val);

	RadialRunner::ShowHideModifiedOptions(CLASSIC);

#ifndef NO_INLET_EXT
	p_PhiExtScale = addFloatParam(M_PHI_EXT, M_PHI_EXT);
	p_PhiExtScale->setValue(0);

	p_RRInAngle = addFloatVectorParam(M_RRIN_ANGLE, M_RRIN_ANGLE);
	p_RRInAngle->setValue(2,init_val);
#endif

}

// **************************************************
void RadialRunner::ShowHideModifiedOptions(int type)
{
	dprintf(2," RadialRunner::ShowHideExtended(%d) ...\n",type);
	switch(type) {
		case 0:
			p_BlV14Part->hide();
			p_SSLePart->hide();
			p_PSLePart->hide();
			break;
		case CLASSIC:
			p_BlV14Part->hide();
			p_SSLePart->hide();
			p_PSLePart->hide();
			break;
		case MODIFIED:
			p_BlV14Part->show();
			p_SSLePart->show();
			p_PSLePart->show();
			break;
		default:
			dprintf(2," ShowHideExtended(): invalid grid type = %d\n",
					type);
			dprintf(2," This should NOT happen!!\n\n");
			break;
	}
}

// **************************************************
void RadialRunner::ShowHideExtended(int flag)
{
	dprintf(2," RadialRunner::ShowHideExtended(%d) ...\n",flag);

	if(flag) {
		p_InletAngleExt->show();
		p_HeightExt->show();
		p_DiamExt->show();
		p_WidthExt->show();
		p_HubCurveParaExt->show();
		p_ShroudCurveParaExt->show();
	}
	else {
		p_InletAngleExt->hide();
		p_HeightExt->hide();
		p_DiamExt->hide();
		p_WidthExt->hide();
		p_HubCurveParaExt->hide();
		p_ShroudCurveParaExt->hide();
	}
}

