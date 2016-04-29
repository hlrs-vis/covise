// copy data to control panel or v/v

#include "AxialRunner.h"
#include <General/include/log.h>

void AxialRunner::Struct2CtrlPanel(void)
{
   float Dref;

#ifdef ABSOLUTE_VALUES
   Dref      = geo->ar->ref;
#else
   Dref = 1.0;
#endif

   dprintf(1, "AxialRunner::Struct2CtrlPanel(): ...\n");

   // design data
   p_DDischarge->setValue(geo->ar->des->dis);
   p_DHead->setValue(geo->ar->des->head);
   p_DRevolut->setValue(geo->ar->des->revs);
   p_DVRatio->setValue(geo->ar->des->vratio);
   // global runner data
   p_EulerEqn->setValue(geo->ar->euler != 0 );
   p_ModelInlet->setValue(geo->ar->mod->inl != 0 );
   p_ModelBend->setValue(geo->ar->mod->bend != 0 );
   p_ModelOutlet->setValue(geo->ar->mod->outl != 0 );
   p_ModelArb->setValue(geo->ar->mod->arbitrary != 0 );
   p_NumberOfBlades->setValue(geo->ar->nob);
   p_OuterDiameter->setValue(geo->ar->ref);
   dprintf(6, "geo->ar->enlace=%f\n", geo->ar->enlace);
   p_BladeEnlacement->setValue(geo->ar->enlace);
   dprintf(6, "geo->ar->piv=%f\n", geo->ar->piv);
   p_PivotLocation->setValue(geo->ar->piv);
   dprintf(6, "geo->ar->bangle=%f\n", geo->ar->bangle);
   p_BladeAngle->setValue(geo->ar->bangle);

   dprintf(2, "global runner data set\n");

   // machine dimension data
   p_InletExtHeight->setValue(REL2ABS(geo->ar->h_inl_ext,Dref));
   p_InletExtDiameter->setValue(REL2ABS(geo->ar->d_inl_ext,Dref));
   p_InletPitch->setValue(geo->ar->arb_angle);
   p_ArbPart->setValue(0,geo->ar->arb_part[0]);
   p_ArbPart->setValue(1,geo->ar->arb_part[1]);
   p_HubDiameter->setValue(REL2ABS(geo->ar->diam[0], Dref));
   p_ShroudRadius->setValue(0,REL2ABS(geo->ar->r_shroud[0],Dref));
   p_ShroudRadius->setValue(1,REL2ABS(geo->ar->r_shroud[1],Dref));
   p_ShroudAngle->setValue(geo->ar->ang_shroud);
   p_HubCornerA->setValue(REL2ABS(geo->ar->a_hub,Dref));
   p_HubCornerB->setValue(REL2ABS(geo->ar->b_hub,Dref));
   p_HubBendNOS->setValue(geo->ar->hub_nos);
   p_RunnerHeight->setValue(REL2ABS(geo->ar->h_run,Dref));
   p_ShroudSphereDiameter->setValue(REL2ABS(geo->ar->d_shroud_sphere,
      Dref));
   p_ShroudCounter->setValue(geo->ar->shroud_counter_rad);
   p_ShroudCounterNOS->setValue(geo->ar->counter_nos);
   p_ShroudHemisphere->setValue(geo->ar->shroud_hemi);
   p_HubSphereDiameter->setValue(REL2ABS(geo->ar->d_hub_sphere,Dref));
   p_DraftHeight->setValue(REL2ABS(geo->ar->h_draft,Dref));
   p_DraftDiameter->setValue(REL2ABS(geo->ar->d_draft,Dref));
   p_DraftAngle->setValue(geo->ar->ang_draft);

   dprintf(2, "machine dimension data set\n");

   // blade edge data
   p_TEShroudConstriction->setValue(geo->ar->te->con[1]);
   p_TEHubConstriction->setValue(geo->ar->te->con[0]);
   p_TENoConstriction->setValue(geo->ar->te->nocon);
   p_LEShroudConstriction->setValue(geo->ar->le->con[1]);
   p_LEHubConstriction->setValue(geo->ar->le->con[0]);
   p_LENoConstriction->setValue(geo->ar->le->nocon);

   dprintf(2, "blade edge data set\n");

   // blade elements, bias data
   p_NumberOfBladeElements->setValue(geo->ar->be_num);
   p_BladeElementBiasFactor->setValue(geo->ar->be_bias);
   p_BladeElementBiasType->setValue(geo->ar->be_type);
   p_ForceCamb->setValue(geo->ar->clspline != 0 );
   p_LeSplineParameters->setValue(geo->ar->le_part[0],
      geo->ar->le_part[1],
      geo->ar->le_part[2]);
   SetFloatDoubleVector(p_TeSplineParameters,geo->ar->te_part[0],geo->ar->te_part[1]);
   dprintf(2, "blade element bias data set\n");
   if(!p_ForceCamb->getValue()) p_TeSplineParameters->hide();

   BladeElements2CtrlPanel();
   BladeElements2Reduced();

   dprintf(1, "AxialRunner::Struct2CtrlPanel()... done\n");
}


void AxialRunner::BladeElements2CtrlPanel(void)
{
   int i;
   // blade element array data
   for(i = 0; i < geo->ar->be_num; i++)
   {
      p_InletAngle[i]->setValue(geo->ar->be[i]->angle[0]);
      p_OutletAngle[i]->setValue(geo->ar->be[i]->angle[1]);
      p_InletAngleModification[i]->setValue(geo->ar->be[i]->mod_angle[0]);
      p_OutletAngleModification[i]->setValue(geo->ar->be[i]->mod_angle[1]);
      p_ProfileThickness[i]->setValue(geo->ar->be[i]->p_thick);
      p_TrailingEdgeThickness[i]->setValue(geo->ar->be[i]->te_thick);
      p_MaximumCamber[i]->setValue(geo->ar->be[i]->camb);
      p_CamberPosition[i]->setValue(geo->ar->be[i]->camb_pos);
      p_ProfileShift[i]->setValue(geo->ar->be[i]->bp_shift);
   }

   for (i = geo->ar->be_num; i < MAX_ELEMENTS; i++)
   {
      p_InletAngle[i]->disable();
      p_OutletAngle[i]->disable();
      p_InletAngleModification[i]->disable();
      p_OutletAngleModification[i]->disable();
      p_ProfileThickness[i]->disable();
      p_TrailingEdgeThickness[i]->disable();
      p_MaximumCamber[i]->disable();
      p_CamberPosition[i]->disable();
      p_ProfileShift[i]->disable();
   }
}


void AxialRunner::BladeElements2Reduced(void)
{
   const int left  = 0;
   const int middle = (int)(geo->ar->be_num / 2);
   const int right    = geo->ar->be_num - 1;
   // min/max values of parameters
   const float min_angle       = -20.0f;
   const float max_angle       = 120.0f;
   const float max_angle_mod   =  60.0f;
   const float min_camb     = 0.1f;
   const float max_camb     = 0.95f;
   const float min_shift       = 0.05f;
   const float max_shift       = 10.0f;
   const float rel_min_pthick  = 0.0001f;
   const float rel_max_pthick  = 0.0020f;
   const float rel_min_tethick = 0.00001f;
   const float rel_max_tethick = 0.001f;
   float min_pthick;
   float max_pthick;
   float min_tethick;
   float max_tethick;
   min_pthick  = rel_min_pthick*geo->minmax[0];
   max_pthick  = rel_max_pthick*geo->minmax[1];
   min_tethick = rel_min_tethick*geo->minmax[0];
   max_tethick = rel_max_tethick*geo->minmax[1];

   dprintf(2, "blade element array data set\n");

   p_HubPoint[0]->setValue(min_angle, max_angle,
      geo->ar->be[left]->angle[0]);
   p_InnerPoint[0]->setValue(min_angle, max_angle,
      geo->ar->be[middle]->angle[0]);
   p_ShroudPoint[0]->setValue(min_angle, max_angle,
      geo->ar->be[right]->angle[0]);
   p_HubPoint[1]->setValue(min_angle, max_angle,
      geo->ar->be[left]->angle[1]);
   p_InnerPoint[1]->setValue(min_angle, max_angle,
      geo->ar->be[middle]->angle[1]);
   p_ShroudPoint[1]->setValue(min_angle, max_angle,
      geo->ar->be[right]->angle[1]);
   p_HubPoint[2]->setValue(min_angle, max_angle_mod,
      geo->ar->be[left]->mod_angle[0]);
   p_InnerPoint[2]->setValue(min_angle, max_angle_mod,
      geo->ar->be[middle]->mod_angle[0]);
   p_ShroudPoint[2]->setValue(min_angle, max_angle_mod,
      geo->ar->be[right]->mod_angle[0]);
   p_HubPoint[3]->setValue(min_angle, max_angle_mod,
      geo->ar->be[left]->mod_angle[1]);
   p_InnerPoint[3]->setValue(min_angle, max_angle_mod,
      geo->ar->be[middle]->mod_angle[1]);
   p_ShroudPoint[3]->setValue(min_angle, max_angle_mod,
      geo->ar->be[right]->mod_angle[1]);
   p_HubPoint[4]->setValue(min_pthick, max_pthick,
      geo->ar->be[left]->p_thick);
   p_InnerPoint[4]->setValue(min_pthick, max_pthick,
      geo->ar->be[middle]->p_thick);
   p_ShroudPoint[4]->setValue(min_pthick, max_pthick,
      geo->ar->be[right]->p_thick);
   p_HubPoint[5]->setValue(min_tethick, max_tethick,
      geo->ar->be[left]->te_thick);
   p_InnerPoint[5]->setValue(min_tethick, max_tethick,
      geo->ar->be[middle]->te_thick);
   p_ShroudPoint[5]->setValue(min_tethick, max_tethick,
      geo->ar->be[right]->te_thick);
   p_HubPoint[6]->setValue(min_camb, max_camb, geo->ar->be[left]->camb);
   p_InnerPoint[6]->setValue(min_camb, max_camb,
      geo->ar->be[middle]->camb);
   p_ShroudPoint[6]->setValue(min_camb, max_camb,
      geo->ar->be[right]->camb);
   p_HubPoint[7]->setValue(min_camb, max_camb,
      geo->ar->be[left]->camb_pos);
   p_InnerPoint[7]->setValue(min_camb, max_camb,
      geo->ar->be[middle]->camb_pos);
   p_ShroudPoint[7]->setValue(min_camb, max_camb,
      geo->ar->be[right]->camb_pos);
   p_HubPoint[8]->setValue(min_shift, max_shift,
      geo->ar->be[left]->bp_shift);
   p_InnerPoint[8]->setValue(min_shift, max_shift,
      geo->ar->be[middle]->bp_shift);
   p_ShroudPoint[8]->setValue(min_shift, max_shift,
      geo->ar->be[right]->bp_shift);

}


void AxialRunner::Grid2CtrlPanel(void)
{
   dprintf(2," Grid2CtrlPanel() ...");fflush(stderr);
   p_GridMerids->setValue((float)rrg->ge_num, rrg->ge_bias, (float)rrg->ge_type);
   p_CircumfDis->setValue((float)rrg->cdis, rrg->cbias, (float)rrg->cbias_type);
   p_CircumfDisLe->setValue((float)rrg->cledis, rrg->clebias, (float)rrg->clebias_type);
   p_MeridInletDis->setValue((float)rrg->ssmdis, rrg->ssmbias, (float)rrg->ssmbias_type);
   p_PSDis->setValue((float)rrg->psdis, rrg->psbias, (float)rrg->psbias_type);
   p_SSDis->setValue((float)rrg->ssdis, rrg->ssbias, (float)rrg->ssbias_type);
   p_BLDis->setValue((float)rrg->psedis, rrg->psebias, (float)rrg->psebias_type);
   p_MeridOutletDis->setValue((float)rrg->lowdis, rrg->lowbias, (float)rrg->lowbias_type);
   p_OutletCoreDis->setValue((float)rrg->lowindis, rrg->lowinbias, (float)rrg->lowin_type);
#ifndef NO_INLET_EXT
   p_MeridInExtDis->setValue((float)rrg->extdis, rrg->extbias, (float)rrg->extbias_type);
#endif
   p_PhiScale->setValue(0,rrg->phi_scale[0]);
   p_PhiScale->setValue(1,rrg->phi_scale[1]);
   p_PhiSkew->setValue(0,rrg->phi_skew[0]);
   p_PhiSkew->setValue(1,rrg->phi_skewout[1]);
   p_PhiSkewOut->setValue(0,rrg->phi_skewout[0]);
   p_PhiSkewOut->setValue(1,rrg->phi_skewout[1]);
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
   rrg->inbc->bcQ   = geo->ar->des->dis;
   rrg->inbc->bcH   = geo->ar->des->head;
   rrg->inbc->bcN   = geo->ar->des->revs;
   rrg->inbc->vratio = geo->ar->des->vratio;
   p_bcQ->setValue(rrg->inbc->bcQ);
   p_bcH->setValue(rrg->inbc->bcH);

   // switches
   p_writeGrid->setValue(rrg->write_grid != 0);
   p_createBC->setValue(rrg->create_inbc != 0);
   p_meshExt->setValue(rrg->mesh_ext != 0 );

   dprintf(2," done!\n");
}

