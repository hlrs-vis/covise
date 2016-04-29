// create the menues

#include "AxialRunner.h"
#include <General/include/log.h>

void AxialRunner::CreateUserMenu(void)
{
   dprintf(1, "Entering CreateUserMenu()\n");
   p_TypeSelected = paraSwitch("types", "Select_type_of_parameters");
#ifdef   VATECH
   paraCase(M_VAT_CFDSWITCHES);
   AxialRunner::CreateMenuVATCFDSwitches();
   paraEndCase();
#endif                                         // VATECH
   paraCase(M_DESIGN_DATA);
   AxialRunner::CreateMenuDesignData();
   paraEndCase();                                 // end of M_DESIGN_DATA
   paraCase(M_RUNNER_DATA);
   AxialRunner::CreateMenuRunnerData();
   paraEndCase();                                 // end of M_RUNNER_DATA

   paraCase(M_MACHINE_DIMS);
   AxialRunner::CreateMenuMachineDimensions();
   paraEndCase();                                 // end of M_MACHINE_DIMS

   paraCase(M_BLADE_EDGES);
   AxialRunner::CreateMenuBladeEdges();
   paraEndCase();                                 // end of M_BLADE_EDGES

#ifdef CREATE_PROFILE_MENU
   paraCase(M_BLADE_PROFILE_DATA);
   AxialRunner::CreateMenuProfileData();
   paraEndCase();                                 // end of M_BLADE_PROFILE_DATA
#endif                                         // CREATE_PROFILE_MENU

   paraCase(M_BLADE_DATA);
   AxialRunner::CreateMenuBladeData();
   paraEndCase();                                 // end of M_BLADE_DATA

   paraCase(M_GRID_DATA);
   AxialRunner::CreateMenuGridData();
   paraEndCase();                                 // end of M_GRID_DATA
   paraEndSwitch();                               // end of "types"
}


void AxialRunner::CreatePortMenu(void)
{
   int i;
   char name[200];

   paraSwitch(M_2DPORT, "Select_plot_port");
   for(i = 1; i <= NUM_PLOT_PORTS; i++)
   {
      sprintf(name,"%s%d","port_no._",i);
      paraCase(name);
      AxialRunner::CreatePlotPortMenu(i);
#ifdef DEBUG_PLOT_SELECT
      fprintf(stderr,"name: %s\n",name);
#endif
      paraEndCase();                              // end of M_MERIDIAN_CONTOUR_PLOT
   }
   paraEndSwitch();                               // end of "2DplotPort1"

}


void AxialRunner::CreatePlotPortMenu(int p)
{
   char name[110];

   sprintf(name,"%s_%d",M_2DPLOT,p);
   m_2DplotChoice[p-1] = paraSwitch(name, "Select_plot_data");
   paraCase(M_MERIDIAN_CONTOUR_PLOT);
   paraEndCase();                                 // end of M_MERIDIAN_CONTOUR_PLOT
   paraCase(M_CONFORMAL_VIEW);
   AxialRunner::CreateMenuConformalView(p);
   paraEndCase();                                 // end of M_CONFORMAL_VIEW
   paraCase(M_CAMBER);
   AxialRunner::CreateMenuCamber(p);
   paraEndCase();
   paraCase(M_NORMCAMBER);
   AxialRunner::CreateMenuNormCamber(p);
   paraEndCase();
   paraCase(M_THICKNESS);
   paraEndCase();
   paraCase(M_MAXTHICK);
   paraEndCase();
   paraCase(M_OVERLAP);
   paraEndCase();
   paraCase(M_BLADE_ANGLES);
   paraEndCase();
   paraCase(M_EULER_ANGLES);
   paraEndCase();
   paraCase(M_CHORD_ANGLE);
   paraEndCase();
   paraCase(M_PARAM_SLIDER);
   paraEndCase();
   paraEndSwitch();                               // end of "2DplotPort"

}


void AxialRunner::CreateMenuConformalView(int p)
{
   int i;
   char buf[200];

   for(i = 0; i < MAX_ELEMENTS; i++)
   {
      sprintf(buf,"%s_%d_%d",M_SHOW_CONFORMAL,p,i+1);
      p_ShowConformal[i][p-1] = addBooleanParam(buf,buf);
      p_ShowConformal[i][p-1]->setValue(0);
   }
}


void AxialRunner::CreateMenuCamber(int p)
{
   int i;
   char buf[200];

   for(i = 0; i < MAX_ELEMENTS; i++)
   {
      sprintf(buf,"%s_%d_%d",M_SHOW_CAMBER,p,i+1);
      p_ShowCamber[i][p-1] = addBooleanParam(buf,buf);
      p_ShowCamber[i][p-1]->setValue(0);
   }
}


void AxialRunner::CreateMenuNormCamber(int p)
{
   int i;
   char buf[200];

   for(i = 0; i < MAX_ELEMENTS; i++)
   {
      sprintf(buf,"%s_%d_%d",M_SHOW_NORMCAMBER,p,i+1);
      p_ShowNormCamber[i][p-1] = addBooleanParam(buf,buf);
      p_ShowNormCamber[i][p-1]->setValue(0);
   }
}


void AxialRunner::CreateMenuDesignData(void)
{
   p_DDischarge = addFloatParam(M_DESIGN_Q,M_DESIGN_Q);
   p_DDischarge->setValue(0.0);

   p_DHead       = addFloatParam(M_DESIGN_H,M_DESIGN_H);
   p_DHead->setValue(0.0);

   p_DRevolut   = addFloatParam(M_DESIGN_N,M_DESIGN_N);
   p_DRevolut->setValue(0.0);

   p_DDefineVRatio = addBooleanParam(M_DEFINE_VRATIO,M_DEFINE_VRATIO);
   p_DDefineVRatio->setValue(0);

   p_DVRatio   = addFloatParam(M_INLET_VRATIO,M_INLET_VRATIO);
   p_DVRatio->setValue(0.0);
   p_DVRatio->hide();
}


void AxialRunner::CreateMenuRunnerData(void)
{
   p_EulerEqn = addBooleanParam(M_EULER_EQN,M_EULER_EQN);
   p_EulerEqn->setValue(0);

   p_WriteBladeData = addBooleanParam(M_WRITE_BLADEDATA,
      M_WRITE_BLADEDATA);
   p_WriteBladeData->setValue(0);

   p_RotateClockwise = addBooleanParam(M_ROTATE_CLOCKWISE,
      M_ROTATE_CLOCKWISE);
   p_RotateClockwise->setValue(0);

   p_ModelInlet = addBooleanParam(M_MODEL_INLET,M_MODEL_INLET);
   p_ModelInlet->setValue(0);

   p_ModelBend = addBooleanParam(M_MODEL_BEND,M_MODEL_BEND);
   p_ModelBend->setValue(0);

   p_ModelArb = addBooleanParam(M_MODEL_ARB,M_MODEL_ARB);
   p_ModelArb->setValue(0);

   p_ModelOutlet = addBooleanParam(M_MODEL_OUTLET,M_MODEL_OUTLET);
   p_ModelOutlet->setValue(0);

   p_NumberOfBlades = addInt32Param(M_NO_BLADES, M_NO_BLADES);
   p_NumberOfBlades->setValue(0);

   p_OuterDiameter = addFloatParam(M_OUTER_DIAMETER,
      M_OUTER_DIAMETER);
   p_OuterDiameter->setValue(0.0);

   p_BladeEnlacement = addFloatParam(M_BLADE_ENLACEMENT,
      M_BLADE_ENLACEMENT);
   p_BladeEnlacement->setValue(0.0);

   p_PivotLocation = addFloatParam(M_PIVOT_LOCATION,
      M_PIVOT_LOCATION);
   p_PivotLocation->setValue(0.0);

   p_BladeAngle = addFloatSliderParam(M_BLADE_ANGLE, M_BLADE_ANGLE);
   p_BladeAngle->setValue(0.0);

   p_BladeAngle->setMin(-10);
   p_BladeAngle->setMax(5);

}


void AxialRunner::CreateMenuMachineDimensions(void)
{
   float init_val[] = {0,0};

   paraSwitch(M_GEO_MANIPULATION, "Select_part_of_runner");
   paraCase(M_INLET);
   p_InletExtHeight = addFloatParam(M_INLET_HEIGHT, M_INLET_HEIGHT);
   p_InletExtHeight->setValue(0.0);

   p_InletExtDiameter = addFloatParam(M_INLET_DIAMETER,
      M_INLET_DIAMETER);
   p_InletExtDiameter->setValue(0.0);

   p_InletPitch = addFloatParam(M_INLET_PITCH, M_INLET_PITCH);
   p_InletPitch->setValue(0.0);

   p_ArbPart = addFloatVectorParam(M_ARB_PART, M_ARB_PART);
   p_ArbPart->setValue(2,init_val);
   paraEndCase();

   paraCase(M_BEND);
   p_BendSelection = paraSwitch(M_HUB_SHROUD_BEND,M_HUB_SHROUD);
   paraCase(M_HUB_CORNER);
   AxialRunner::CreateHubCornerMenu();
   paraEndCase();                                 // M_HUB_CORNER

   paraCase(M_SHROUD_CORNER);
   p_ShroudRadius = addFloatVectorParam(M_SBEND_RAD, M_SBEND_RAD);
   p_ShroudRadius->setValue(2,init_val);

   p_ShroudAngle = addFloatParam(M_SBEND_ANGLE, M_SBEND_ANGLE);
   p_ShroudAngle->setValue(0.0);
   paraEndCase();                                 // M_SHROUD
   paraEndSwitch();                               // M_HUB_SHROUD_CORNER
   paraEndCase();                                 // M_BEND

   paraCase(M_RUNNER);
   p_RunnerHeight = addFloatParam(M_RUNNER_HEIGHT, M_RUNNER_HEIGHT);
   p_RunnerHeight->setValue(0.0);

   paraSwitch(M_HUB_SHROUD_RUNNER,M_HUB_SHROUD);
   paraCase(M_HUB);
   p_HubDiameter = addFloatParam(M_HUB_DIAMETER, M_HUB_DIAMETER);
   p_HubDiameter->setValue(0.0);

   p_HubSphereDiameter = addFloatParam(M_HSPHERE_DIAMETER,
      M_HSPHERE_DIAMETER);
   p_HubSphereDiameter->setValue(0.0);

   AxialRunner::CreateHubCapMenu();
   paraEndCase();                                 // M_HUB

   paraCase(M_SHROUD);
   p_ShroudSphereDiameter = addFloatParam(M_SSPHERE_DIAMETER,
      M_SSPHERE_DIAMETER);
   p_ShroudSphereDiameter->setValue(0.0);

   p_ShroudHemisphere = addInt32Param(M_SSPHERE_HEMI,
      M_SSPHERE_HEMI);
   p_ShroudHemisphere->setValue(0);

   p_ShroudCounter = addInt32Param(M_SCOUNTER_ARC, M_SCOUNTER_ARC);
   p_ShroudCounter->setValue(0);

   p_ShroudCounterNOS = addInt32Param(M_SCOUNTER_NOS,
      M_SCOUNTER_NOS);
   p_ShroudCounterNOS->setValue(0);
   paraEndCase();                                 // M_SHROUD
   paraEndSwitch();                               // M_HUB_SHROUD
   paraEndCase();                                 // M_RUNNER

   paraCase(M_OUTLET);
   p_DraftHeight = addFloatParam(M_DRAFT_HEIGHT, M_DRAFT_HEIGHT);
   p_DraftHeight->setValue(0.0);

   p_DraftDiameter = addFloatParam(M_DRAFT_DIAMETER,
      M_DRAFT_DIAMETER);
   p_DraftDiameter->setValue(0.0);

   p_DraftAngle = addFloatParam(M_DRAFT_ANGLE, M_DRAFT_ANGLE);
   p_DraftAngle->setValue(0.0);
   paraEndCase();
   paraEndSwitch();                               // M_GEO_MANIPULATION
}


void AxialRunner::CreateHubCornerMenu(void)
{
   int i;
   float init_val[] = {0,0};
   char buf[200];
   char *pselect[MAX_POINTS+1];

   dprintf(1,"AxialRunner::CreateHubCornerMenu() ...\n");
   for(i = 0; i < MAX_POINTS; i++)
   {
      sprintf(buf,"point_%02d",i+1);
      dprintf(3,"   buf = %s (i=%d)\n",buf,i);
      pselect[i] = strdup(buf);
   }

   p_HubCornerA = addFloatParam(M_HBEND_CORNER_A, M_HBEND_CORNER_A);
   p_HubCornerA->setValue(0.0);

   p_HubCornerB = addFloatParam(M_HBEND_CORNER_B, M_HBEND_CORNER_B);
   p_HubCornerB->setValue(0.0);

   p_HubBendNOS = addInt32Param(M_HBEND_NOS, M_HBEND_NOS);
   p_HubBendNOS->setValue(0);

   p_HubBendModifyPoints = addBooleanParam(M_HBEND_MODIFY,M_HBEND_MODIFY);
   p_HubBendModifyPoints->setValue(0);

   p_HubPointSelected = addChoiceParam(M_POINT_DATA, M_POINT_DATA);
   p_HubPointSelected->setValue(MAX_POINTS,pselect,0);
   p_HubPointSelected->hide();

   p_HubPointValues = addFloatVectorParam(M_HPOINT_VAL,M_HPOINT_VAL);
   p_HubPointValues->setValue(2,init_val);
   p_HubPointValues->hide();
}


void AxialRunner::CreateHubCapMenu(void)
{
   int i;
   float init_val[] = {0,0};
   char buf[200];
   char *pselect[MAX_POINTS+1];

   dprintf(1,"AxialRunner::CreateHubCapMenu() ...\n");
   for(i = 0; i < MAX_POINTS; i++)
   {
      sprintf(buf,"point_%02d",i+1);
      pselect[i] = strdup(buf);
   }

   p_HubCPointSelected = addChoiceParam(M_CPOINT_DATA, M_CPOINT_DATA);
   p_HubCPointSelected->setValue(MAX_POINTS,pselect,0);
   p_HubCPointSelected->hide();

   p_HubCPointValues = addFloatVectorParam(M_HCPOINT_VAL,M_HCPOINT_VAL);
   p_HubCPointValues->setValue(2,init_val);
   p_HubCPointValues->hide();
}


void AxialRunner::ReducedModifyMenu(void)
{
   int i;
   char *buf;

   m_sliderChoice = paraSwitch(M_REDUCED_MODIFY, M_REDUCED_MODIFY);
   for (i = 0; i < numReducedMenuPoints; i++)
   {
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


void AxialRunner::CreateMenuBladeData(void)
{
   int i;
   char *buf;
   float init_val[] = {0,0};

   p_NumberOfBladeElements = addInt32Param(M_NO_BLADE_ELEMENTS, M_NO_BLADE_ELEMENTS);
   p_NumberOfBladeElements->setValue(0);

   p_BladeElementBiasFactor = addFloatParam(M_BLADE_BIAS_FACTOR, M_BLADE_BIAS_FACTOR);
   p_BladeElementBiasFactor->setValue(0.0);

   p_BladeElementBiasType = addInt32Param(M_BLADE_BIAS_TYPE, M_BLADE_BIAS_TYPE);
   p_BladeElementBiasType->setValue(0);

   p_ForceCamb = addBooleanParam(M_FORCE_CAMB, M_FORCE_CAMB);
   p_ForceCamb->setValue(0);

   p_LeSplineParameters = addFloatVectorParam(M_LESPLINE_PARAMETERS,M_LESPLINE_PARAMETERS);
   p_LeSplineParameters->setValue(0,0,0);

   p_TeSplineParameters = addFloatVectorParam(M_TESPLINE_PARAMETERS,M_TESPLINE_PARAMETERS);
   p_TeSplineParameters->setValue(2,init_val);

   paraSwitch(M_BLADE_ELEMENT_DATA, M_BLADE_ELEMENT_DATA);
   for (i = 0; i < MAX_ELEMENTS; i++)
   {
      paraCase(IndexedParameterName(M_BLADE_ELEMENT_DATA, i));

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

      buf = IndexedParameterName(M_MAXIMUM_CAMBER, i);
      p_MaximumCamber[i] = addFloatParam(buf, buf);
      p_MaximumCamber[i]->setValue(0.0);
      p_MaximumCamber[i]->disable();
      free(buf);

      buf = IndexedParameterName(M_CAMBER_POSITION, i);
      p_CamberPosition[i] = addFloatParam(buf, buf);
      p_CamberPosition[i]->setValue(0.0);
      p_CamberPosition[i]->disable();
      free(buf);

      buf = IndexedParameterName(M_PROFILE_SHIFT, i);
      p_ProfileShift[i] = addFloatParam(buf, buf);
      p_ProfileShift[i]->setValue(0.0);
      p_ProfileShift[i]->disable();
      free(buf);

      paraEndCase();
   }
   paraEndSwitch();

   i = LOCK_PTHICK;
   p_Lock[i] = addBooleanParam(M_LOCK_PTHICK,M_LOCK_PTHICK);
   p_Lock[i]->setValue(0);

   ReducedModifyMenu();
}


void AxialRunner::CreateMenuBladeEdges()
{
   p_TEShroudConstriction = addFloatParam(M_TE_SHROUD_CON, M_TE_SHROUD_CON);
   p_TEShroudConstriction->setValue(0.0);

   p_LEShroudConstriction = addFloatParam(M_LE_SHROUD_CON, M_LE_SHROUD_CON);
   p_LEShroudConstriction->setValue(0.0);

   p_TEHubConstriction = addFloatParam(M_TE_HUB_CON, M_TE_HUB_CON);
   p_TEHubConstriction->setValue(0.0);

   p_LEHubConstriction = addFloatParam(M_LE_HUB_CON, M_LE_HUB_CON);
   p_LEHubConstriction->setValue(0.0);

   p_TENoConstriction = addFloatParam(M_TE_NO_CON, M_TE_NO_CON);
   p_TENoConstriction->setValue(0.0);

   p_LENoConstriction = addFloatParam(M_LE_NO_CON, M_LE_NO_CON);
   p_LENoConstriction->setValue(0.0);
}


void AxialRunner::CreateMenuGridData(void)
{
   const char *gridtypes[2] = {M_B2B_CLASSIC,M_B2B_MODIFIED};
   long int init_val[] = {1,1};

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
   AxialRunner::CreateMenuBasicGrid();
   paraEndCase();
   paraCase(M_GRID_TOPOLOGY);
   AxialRunner::CreateMenuGridTopo();
   paraEndCase();
   paraCase(M_BC_AND_CFD);
   AxialRunner::CreateMenuBCandCFD();
   paraEndCase();
   paraEndSwitch();

}


void AxialRunner::CreateMenuBasicGrid(void)
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

   p_OutletCoreDis = addFloatVectorParam(M_OUTLET_CORE,M_OUTLET_CORE);
   p_OutletCoreDis->setValue(0,0,0);

#ifndef NO_INLET_EXT
   p_MeridInExtDis = addFloatVectorParam(M_MERID_INEXT,M_MERID_INEXT);
   p_MeridInExtDis->setValue(0,0,0);
#endif

}


void AxialRunner::CreateMenuBCandCFD(void)
{
   p_writeGrid = addBooleanParam("writeGrid",
      "Write_grid_and_boco_to_file");
   p_writeGrid->setValue(0);

   p_createBC = addBooleanParam(M_CREATE_BC,M_CREATE_BC);
   p_createBC->setValue(0);

   p_constAlpha = addBooleanParam(M_CONST_ALPHA,M_CONST_ALPHA);
   p_constAlpha->setValue(0);

   p_turbProf = addBooleanParam(M_TURB_PROF,M_TURB_PROF);
   p_turbProf->setValue(0);

   p_meshExt = addBooleanParam(M_MESH_EXT,M_MESH_EXT);
   p_meshExt->setValue(0);

   p_rotExt = addBooleanParam(M_ROT_EXT,M_ROT_EXT);
   p_rotExt->setValue(0);

   p_RunFENFLOSS = addBooleanParam(M_RUN_FEN,"runFENFLOSS_CFD");
   p_RunFENFLOSS->setValue(0);

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


void AxialRunner::CreateMenuGridTopo(void)
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

   AxialRunner::ShowHideModifiedOptions(CLASSIC);

#ifndef NO_INLET_EXT
   p_PhiExtScale = addFloatParam(M_PHI_EXT, M_PHI_EXT);
   p_PhiExtScale->setValue(0);

   p_RRInAngle = addFloatVectorParam(M_RRIN_ANGLE, M_RRIN_ANGLE);
   p_RRInAngle->setValue(2,init_val);
#endif

}


void AxialRunner::ShowHideModifiedOptions(int type)
{
   dprintf(2," AxialRunner::ShowHideModifiedOptions(%d) ...\n",type);
   switch(type)
   {
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
         sendError(" ShowHideExtended(): invalid grid type = %d\n", type);
         break;
   }
}
