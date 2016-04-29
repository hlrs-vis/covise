// copy data to control panel or v/v

#include "RadialRunner.h"
#include <General/include/log.h>

void RadialRunner::Struct2CtrlPanel(void)
{

	dprintf(2, "RadialRunner::Struct2CtrlPanel() entering ...");
	dprintf(2, "euler = %d\n",geo->rr->euler);
	p_DDischarge->setValue(geo->rr->des->dis);
	p_DHead->setValue(geo->rr->des->head);
	p_DRevolut->setValue(geo->rr->des->revs);
	p_DVRatio->setValue(geo->rr->des->vratio);
   p_EulerEqn->setValue(geo->rr->euler != 0);
	p_Camb2Surf->setValue(geo->rr->camb2surf - 1);
	p_StraightHub->setValue(geo->rr->straight_cont[0] != 0);
	p_StraightShrd->setValue(geo->rr->straight_cont[1] != 0);
	p_CamberPos->setValue(geo->rr->camb_flag != 0);
	dprintf(2," p_EulerEqn = %d\n",p_EulerEqn->getValue());
	p_ShowExtensions->setValue(geo->rr->showExt != 0);
	p_NumberOfBlades->setValue(geo->rr->nob);
	p_OutletDiameterAbs->setValue(geo->rr->ref);
	p_InletDiameterRel->setValue(geo->rr->diam[0]);
	p_ShroudHeightDiff->setValue(geo->rr->height);
#ifdef GAP
	p_GapWidth->setValue(geo->rr->gap);
#endif
	p_ConduitWidth->setValue(0,geo->rr->cond[0]);
	p_ConduitWidth->setValue(1,geo->rr->cond[1]);
	p_ContourAngles->setValue(0,GRAD(geo->rr->angle[0]));
	p_ContourAngles->setValue(1,GRAD(geo->rr->angle[1]));
	p_InletOpenAngles->setValue(0,GRAD(geo->rr->iop_angle[0]));
	p_InletOpenAngles->setValue(1,GRAD(geo->rr->iop_angle[1]));
	p_OutletOpenAngles->setValue(0,GRAD(geo->rr->oop_angle[0]));
	p_OutletOpenAngles->setValue(1,GRAD(geo->rr->oop_angle[1]));
	p_HubCurveParameters->setValue(0,geo->rr->hspara[0]);
	p_HubCurveParameters->setValue(1,geo->rr->hspara[1]);
	p_ShroudCurveParameters->setValue(0,geo->rr->sspara[0]);
	p_ShroudCurveParameters->setValue(1,geo->rr->sspara[1]);
	p_ShroudStraightParameters->setValue(0,geo->rr->sstparam[0]);
	p_ShroudStraightParameters->setValue(1,geo->rr->sstparam[1]);
	p_HubStraightParameters->setValue(0,geo->rr->hstparam[0]);
	p_HubStraightParameters->setValue(1,geo->rr->hstparam[1]);

	p_InletAngleExt->setValue(GRAD(geo->rr->ext_iangle));
	p_HeightExt->setValue(0,geo->rr->ext_height[0]);
	p_HeightExt->setValue(1,geo->rr->ext_height[1]);
	p_DiamExt->setValue(0,geo->rr->ext_diam[0]);
	p_DiamExt->setValue(1,geo->rr->ext_diam[1]);
	p_WidthExt->setValue(0,geo->rr->ext_cond[0]);
	p_WidthExt->setValue(1,geo->rr->ext_cond[1]);
	p_HubCurveParaExt->setValue(0,geo->rr->hspara_inext[0]);
	p_HubCurveParaExt->setValue(1,geo->rr->hspara_inext[1]);
	p_ShroudCurveParaExt->setValue(0,geo->rr->sspara_inext[0]);
	p_ShroudCurveParaExt->setValue(1,geo->rr->sspara_inext[1]);

	p_LeHubParm->setValue(geo->rr->le->para[0]);
	p_LeHubAngle->setValue(GRAD(geo->rr->le->angle[0]));
	p_LeShroudParm->setValue(geo->rr->le->para[1]);
	p_LeShroudAngle->setValue(GRAD(geo->rr->le->angle[1]));
	p_LeCurveParam->setValue(0,geo->rr->le->spara[0]);
	p_LeCurveParam->setValue(1,geo->rr->le->spara[1]);

	p_TeHubParm->setValue(geo->rr->te->para[0]);
	p_TeHubAngle->setValue(GRAD(geo->rr->te->angle[0]));
	p_TeShroudParm->setValue(geo->rr->te->para[1]);
	p_TeShroudAngle->setValue(GRAD(geo->rr->te->angle[1]));
	p_TeCurveParam->setValue(0,geo->rr->te->spara[0]);
	p_TeCurveParam->setValue(1,geo->rr->te->spara[1]);

	p_NumberOfBladeElements->setValue(geo->rr->be_num);
	p_BladeElementBiasFactor->setValue(geo->rr->be_bias);
	p_BladeElementBiasType->setValue(geo->rr->be_type);

	RadialRunner::BladeElements2CtrlPanel();
	RadialRunner::BladeElements2Reduced();

	dprintf(2, "RadialRunner::Struct2CtrlPanel() ... done \n");
}


void RadialRunner::BladeElements2CtrlPanel(void)
{
	for (int i = 0; i < geo->rr->be_num; i++) {
		p_MeridianParm[i]->setValue(geo->rr->be[i]->para);
		p_InletAngle[i]->setValue(GRAD(geo->rr->be[i]->angle[0]));
		p_OutletAngle[i]->setValue(GRAD(geo->rr->be[i]->angle[1]));
		p_ProfileThickness[i]->setValue(geo->rr->be[i]->p_thick);
		p_TrailingEdgeThickness[i]->setValue(geo->rr->be[i]->te_thick);
		p_TrailingEdgeWrap[i]->setValue(GRAD(geo->rr->be[i]->te_wrap));
		p_BladeWrap[i]->setValue(GRAD(geo->rr->be[i]->bl_wrap));
		p_ProfileShift[i]->setValue(geo->rr->be[i]->bp_shift);
		p_InletAngleModification[i]->setValue(GRAD(geo->rr->be[i]->mod_angle[0]));
		p_OutletAngleModification[i]->setValue(GRAD(geo->rr->be[i]->mod_angle[1]));
		p_RemainingSwirl[i]->setValue(geo->rr->be[i]->rot_abs[1]);
		p_BladeLePara[i]->setValue(geo->rr->be[i]->le_para);
		p_BladeTePara[i]->setValue(geo->rr->be[i]->te_para);
		p_CentreLineCamber[i]->setValue(geo->rr->be[i]->camb);
		p_CentreLineCamberPosn[i]->setValue(geo->rr->be[i]->camb_pos);
		p_CambPara[i]->setValue(geo->rr->be[i]->camb_para);
		p_BladeLengthFactor[i]->setValue(geo->rr->be[i]->bl_lenpara);
	}
}


void RadialRunner::BladeElements2Reduced(void)
{
	// DON NOT CHANGE !!!
	const int left	 = 0;
	const int middle = (int)(geo->rr->be_num / 2);
	const int right	 = geo->rr->be_num - 1;
	// min/max values of parameters
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
	const float min_blwrap	=	5.0f;
	const float max_blwrap	= 120.0f;
	const float min_shift	=	0.0f;
	const float max_shift	=  10.0f;
	const float min_swirl	= -10.0f;
	const float max_swirl	=  10.0f;
	const float min_campar	= -2.0f;
	const float max_campar	=  3.0f;
	const float min_bllen	=  0.1f;
	const float max_bllen	=  8.1f;

	p_HubPoint[0]->setValue(min_angle, max_angle, GRAD(geo->rr->be[left]->angle[0]));
	p_InnerPoint[0]->setValue(min_angle, max_angle, GRAD(geo->rr->be[middle]->angle[0]));
	p_ShroudPoint[0]->setValue(min_angle, max_angle, GRAD(geo->rr->be[right]->angle[0]));
	p_HubPoint[1]->setValue(min_angle, max_angle, GRAD(geo->rr->be[left]->angle[1]));
	p_InnerPoint[1]->setValue(min_angle, max_angle, GRAD(geo->rr->be[middle]->angle[1]));
	p_ShroudPoint[1]->setValue(min_angle, max_angle, GRAD(geo->rr->be[right]->angle[1]));
	p_HubPoint[2]->setValue(min_pthick, max_pthick, geo->rr->be[left]->p_thick);
	p_InnerPoint[2]->setValue(min_pthick, max_pthick, geo->rr->be[middle]->p_thick);
	p_ShroudPoint[2]->setValue(min_pthick, max_pthick, geo->rr->be[right]->p_thick);
	p_HubPoint[3]->setValue(min_tethick, max_tethick, geo->rr->be[left]->te_thick);
	p_InnerPoint[3]->setValue(min_tethick, max_tethick, geo->rr->be[middle]->te_thick);
	p_ShroudPoint[3]->setValue(min_tethick, max_tethick, geo->rr->be[right]->te_thick);
	p_HubPoint[4]->setValue(min_tewrap, max_tewrap, GRAD(geo->rr->be[left]->te_wrap));
	p_InnerPoint[4]->setValue(min_tewrap, max_tewrap, GRAD(geo->rr->be[middle]->te_wrap));
	p_ShroudPoint[4]->setValue(min_tewrap, max_tewrap, GRAD(geo->rr->be[right]->te_wrap));
	p_HubPoint[5]->setValue(min_blwrap, max_blwrap, GRAD(geo->rr->be[left]->bl_wrap));
	p_InnerPoint[5]->setValue(min_blwrap, max_blwrap, GRAD(geo->rr->be[middle]->bl_wrap));
	p_ShroudPoint[5]->setValue(min_blwrap, max_blwrap, GRAD(geo->rr->be[right]->bl_wrap));
	p_HubPoint[6]->setValue(min_shift, max_shift, geo->rr->be[left]->bp_shift);
	p_InnerPoint[6]->setValue(min_shift, max_shift, geo->rr->be[middle]->bp_shift);
	p_ShroudPoint[6]->setValue(min_shift, max_shift, geo->rr->be[right]->bp_shift);
	p_HubPoint[7]->setValue(min_mod, max_mod, GRAD(geo->rr->be[left]->mod_angle[0]));
	p_InnerPoint[7]->setValue(min_mod, max_mod, GRAD(geo->rr->be[middle]->mod_angle[0]));
	p_ShroudPoint[7]->setValue(min_mod, max_mod, GRAD(geo->rr->be[right]->mod_angle[0]));
	p_HubPoint[8]->setValue(min_mod, max_mod, GRAD(geo->rr->be[left]->mod_angle[1]));
	p_InnerPoint[8]->setValue(min_mod, max_mod, GRAD(geo->rr->be[middle]->mod_angle[1]));
	p_ShroudPoint[8]->setValue(min_mod, max_mod, GRAD(geo->rr->be[right]->mod_angle[1]));
	p_HubPoint[9]->setValue(min_swirl, max_swirl, geo->rr->be[left]->rot_abs[1]);
	p_InnerPoint[9]->setValue(min_swirl, max_swirl, geo->rr->be[middle]->rot_abs[1]);
	p_ShroudPoint[9]->setValue(min_swirl, max_swirl, geo->rr->be[right]->rot_abs[1]);
	p_HubPoint[10]->setValue(0.0,1.0, geo->rr->be[left]->le_para);
	p_InnerPoint[10]->setValue(0.0,1.0, geo->rr->be[middle]->le_para);
	p_ShroudPoint[10]->setValue(0.0,1.0, geo->rr->be[right]->le_para);
	p_HubPoint[11]->setValue(0.0,1.0, geo->rr->be[left]->te_para);
	p_InnerPoint[11]->setValue(0.0,1.0, geo->rr->be[middle]->te_para);
	p_ShroudPoint[11]->setValue(0.0,1.0, geo->rr->be[right]->te_para);
	p_HubPoint[12]->setValue(min_camb, max_camb, geo->rr->be[left]->camb);
	p_InnerPoint[12]->setValue(min_camb, max_camb, geo->rr->be[middle]->camb);
	p_ShroudPoint[12]->setValue(min_camb, max_camb, geo->rr->be[right]->camb);
	p_HubPoint[13]->setValue(min_camb, max_camb, geo->rr->be[left]->camb_pos);
	p_InnerPoint[13]->setValue(min_camb, max_camb, geo->rr->be[middle]->camb_pos);
	p_ShroudPoint[13]->setValue(min_camb, max_camb, geo->rr->be[right]->camb_pos);
	p_HubPoint[14]->setValue(min_campar, max_campar, geo->rr->be[left]->camb_para);
	p_InnerPoint[14]->setValue(min_campar, max_campar, geo->rr->be[middle]->camb_para);
	p_ShroudPoint[14]->setValue(min_campar, max_campar, geo->rr->be[right]->camb_para);
	p_HubPoint[15]->setValue(min_bllen, max_bllen, geo->rr->be[left]->bl_lenpara);
	p_InnerPoint[15]->setValue(min_bllen, max_bllen, geo->rr->be[middle]->bl_lenpara);
	p_ShroudPoint[15]->setValue(min_bllen, max_bllen, geo->rr->be[right]->bl_lenpara);
}


void RadialRunner::Grid2CtrlPanel(void)
{
	dprintf(2," Grid2CtrlPanel() ...");
	p_GridMerids->setValue((float)rrg->ge_num, rrg->ge_bias, (float)rrg->ge_type);
	p_CircumfDis->setValue((float)rrg->cdis, rrg->cbias, (float)rrg->cbias_type);
	p_CircumfDisLe->setValue((float)rrg->cledis, rrg->clebias, (float)rrg->clebias_type);
	p_MeridInletDis->setValue((float)rrg->ssmdis, rrg->ssmbias, (float)rrg->ssmbias_type);
	p_PSDis->setValue((float)rrg->psdis, rrg->psbias, (float)rrg->psbias_type);
	p_SSDis->setValue((float)rrg->ssdis, rrg->ssbias, (float)rrg->ssbias_type);
	p_BLDis->setValue((float)rrg->psedis, rrg->psebias, (float)rrg->psebias_type);
	p_MeridOutletDis->setValue((float)rrg->lowdis, rrg->lowbias, (float)rrg->lowbias_type);
#ifndef NO_INLET_EXT
	p_MeridInExtDis->setValue((float)rrg->extdis, rrg->extbias, (float)rrg->extbias_type);
#endif
	p_PhiScale->setValue(0,rrg->phi_scale[0]);
	p_PhiScale->setValue(1,rrg->phi_scale[1]);
	p_V14Angle->setValue(0,GRAD(rrg->v14_angle[0]));
	p_V14Angle->setValue(1,GRAD(rrg->v14_angle[1]));
	p_BlV14Part->setValue(0,rrg->bl_v14_part[0]);
	p_BlV14Part->setValue(1,rrg->bl_v14_part[1]);
	p_SSPart->setValue(0,rrg->ss_part[0]);
	p_SSPart->setValue(1,rrg->ss_part[1]);
	p_PSPart->setValue(0,rrg->ps_part[0]);
	p_PSPart->setValue(1,rrg->ps_part[1]);
	p_SSLePart->setValue(0,rrg->ssle_part[0]);
	p_SSLePart->setValue(1,rrg->ssle_part[1]);
	p_PSLePart->setValue(0,rrg->psle_part[0]);
	p_PSLePart->setValue(1,rrg->psle_part[1]);
	p_OutPart->setValue(0,rrg->out_part[0]);
	p_OutPart->setValue(1,rrg->out_part[1]);
#ifndef NO_INLET_EXT
	p_PhiExtScale->setValue(rrg->phi0_ext);
	p_RRInAngle->setValue(0,GRAD(rrg->angle_ext[0]));
	p_RRInAngle->setValue(1,GRAD(rrg->angle_ext[1]));
#endif

	p_BoundLayRatio->setValue(0,(1.0-rrg->bl_scale[0]));
	p_BoundLayRatio->setValue(1,(1.0-rrg->bl_scale[1]));

	// bc-values
	rrg->inbc->bcQ	  = geo->rr->des->dis;
	rrg->inbc->bcH	  = geo->rr->des->head;
	rrg->inbc->bcN	  = geo->rr->des->revs;
	rrg->inbc->vratio = geo->rr->des->vratio;
	p_bcQ->setValue(rrg->inbc->bcQ);
	p_bcH->setValue(rrg->inbc->bcH);

	// switches
	p_writeGrid->setValue(rrg->write_grid != 0);
	p_createBC->setValue(rrg->create_inbc != 0);
	p_meshExt->setValue(rrg->mesh_ext != 0);
	

	dprintf(2," done!\n");
}

