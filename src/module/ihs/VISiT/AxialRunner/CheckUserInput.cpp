// check and copy changed parametres.

#include "AxialRunner.h"
#include <General/include/log.h>

// **************************************************

// **************************************************

int AxialRunner::CheckUserInput(const char *portname, struct geometry *, struct rr_grid *)
{
   int changed;
   // slider locations; DO NOT CHANGE !!!
   if (!geo)  return 0;

   int ind, ibe;
   char pn[256];
   const int left  = 0;
   const int middle = int(geo->ar->be_num / 2);
   const int right    = geo->ar->be_num - 1;
   // min/max values of parameters
   const int   min_nos      = 0;
   const int   max_nos      = 10;
   const float min_camb     =   0.0f;
   const float max_camb     =    1.0f;
   const float min_shift       =   0.0f;
   const float max_shift       =  10.0f;
   const float min_con      =  -1.0f;
   const float max_con      =   1.0f;
   const float min_angle       = -20.0f;
   const float max_angle       = 120.0f;
   const float min_mod      = -20.0f;
   const float max_mod      =  60.0f;
   const float rel_min_diam    =   0.0f;
   const float rel_max_diam    =   3.0f;
   const float rel_min_ext_diam   =   1.0f;
   const float rel_max_ext_diam   =   5.0f;
   const float rel_min_ellipse    =   0.01f;
   const float rel_max_ellipse    =   10.0f;
   const float rel_min_height  =   0.0f;
   const float rel_max_height  =   2.0f;
   const float rel_min_ssphere_diam =   0.0f;
   const float rel_max_ssphere_diam =   2.0f;
   const float rel_min_hsphere_diam =   0.0f;
   const float rel_max_hsphere_diam =   0.8f;
   const float rel_min_pthick  =   0.0001f;
   const float rel_max_pthick  =   0.20f;
   const float rel_min_tethick    =   0.00f;
   const float rel_max_tethick    =   0.01f;
   float Dref;
   float min_diam, max_diam, min_ext_diam, max_ext_diam;
   float min_ellipse, max_ellipse, min_height, max_height;
   float min_ssphere_diam, max_ssphere_diam;
   float min_hsphere_diam, max_hsphere_diam;
   float min_pthick, max_pthick, min_tethick, max_tethick;
#ifdef ABSOLUTE_VALUES
   Dref       = geo->ar->ref;
#else
   Dref = 1.0;
#endif
   min_diam     = rel_min_diam*geo->minmax[0];
   max_diam     = rel_max_diam*geo->minmax[1];
   min_ext_diam = rel_min_ext_diam*geo->minmax[0];
   max_ext_diam = rel_max_ext_diam*geo->minmax[1];
   min_ellipse  = rel_min_ellipse*geo->minmax[0];
   max_ellipse  = rel_max_ellipse*geo->minmax[1];
   min_height   = rel_min_height*geo->minmax[0];
   max_height   = rel_max_height*geo->minmax[1];
   min_ssphere_diam = rel_min_ssphere_diam*geo->minmax[0];
   max_ssphere_diam = rel_max_ssphere_diam*geo->minmax[1];
   min_hsphere_diam = rel_min_hsphere_diam*geo->minmax[0];
   max_hsphere_diam = rel_max_hsphere_diam*geo->minmax[1];
   min_pthick   = rel_min_pthick*geo->minmax[0];
   max_pthick   = rel_max_pthick*geo->minmax[1];
   min_tethick  = rel_min_tethick*geo->minmax[0];
   max_tethick  = rel_max_tethick*geo->minmax[1];

   dprintf(1, "AxialRunner::CheckUserInput() entering ..., pn=%s\n",
      portname);
   changed = 0;

   if (SplitPortname(portname, pn, &ind))
   {
      if(ind < geo->ar->be_num)
      {
      geo->ar->be_single = 1;
      // blade element data
      if (!strcmp(M_INLET_ANGLE, pn))
      {
         changed = CheckUserFloatValue(p_InletAngle[ind], geo->ar->be[ind]->angle[0],
            min_angle, max_angle, &(geo->ar->be[ind]->angle[0]));
      }
      if (!strcmp(M_OUTLET_ANGLE, pn))
      {
         changed = CheckUserFloatValue(p_OutletAngle[ind], geo->ar->be[ind]->angle[1],
            min_angle, max_angle, &(geo->ar->be[ind]->angle[1]));
      }
      if (!strcmp(M_INLET_ANGLE_MODIFICATION, pn))
      {
         changed = CheckUserFloatValue(p_InletAngleModification[ind], geo->ar->be[ind]->mod_angle[0],
            min_mod, max_mod, &(geo->ar->be[ind]->mod_angle[0]));
      }
      if (!strcmp(M_OUTLET_ANGLE_MODIFICATION, pn))
      {
         changed = CheckUserFloatValue(p_OutletAngleModification[ind], geo->ar->be[ind]->mod_angle[1],
            min_mod, max_mod, &(geo->ar->be[ind]->mod_angle[1]));
      }
      if (!strcmp(M_PROFILE_THICKNESS, pn))
      {
         changed = CheckUserFloatValue(p_ProfileThickness[ind], geo->ar->be[ind]->p_thick,
            min_pthick, max_pthick, &(geo->ar->be[ind]->p_thick));
      }
      if (!strcmp(M_TE_THICKNESS, pn))
      {
         changed = CheckUserFloatValue(p_TrailingEdgeThickness[ind], geo->ar->be[ind]->te_thick,
            min_tethick, max_tethick, &(geo->ar->be[ind]->te_thick));
      }
      if (!strcmp(M_MAXIMUM_CAMBER, pn))
      {
         changed = CheckUserFloatValue(p_MaximumCamber[ind], geo->ar->be[ind]->camb,
            min_camb, max_camb, &(geo->ar->be[ind]->camb));
      }
      if (!strcmp(M_CAMBER_POSITION, pn))
      {
         changed = CheckUserFloatValue(p_CamberPosition[ind], geo->ar->be[ind]->camb_pos,
            min_camb, max_camb, &(geo->ar->be[ind]->camb_pos));
      }
      if (!strcmp(M_PROFILE_SHIFT, pn))
      {
         changed = CheckUserFloatValue(p_ProfileShift[ind], geo->ar->be[ind]->bp_shift,
            min_shift, max_shift, &(geo->ar->be[ind]->bp_shift));
      }
      else if (!strcmp(M_LEFT_POINT, pn))
      {
         ibe = left; geo->ar->be_single = 0;
         if (ind == 0)
         {
            changed = CheckUserFloatSliderValue(p_HubPoint[ind], geo->ar->be[ibe]->angle[0],
               min_angle, max_angle, &(geo->ar->be[ibe]->angle[0]));
         }
         if (ind == 1)
         {
            changed = CheckUserFloatSliderValue(p_HubPoint[ind], geo->ar->be[ibe]->angle[1],
               min_angle, max_angle, &(geo->ar->be[ibe]->angle[1]));
         }
         if (ind == 2)
         {
            changed = CheckUserFloatSliderValue(p_HubPoint[ind], geo->ar->be[ibe]->mod_angle[0],
               min_angle, max_angle, &(geo->ar->be[ibe]->mod_angle[0]));
         }
         if (ind == 3)
         {
            changed = CheckUserFloatSliderValue(p_HubPoint[ind], geo->ar->be[ibe]->mod_angle[1],
               min_angle, max_angle, &(geo->ar->be[ibe]->mod_angle[1]));
         }
         if (ind == 4)
         {
            changed = CheckUserFloatSliderValue(p_HubPoint[ind], geo->ar->be[ibe]->p_thick,
               min_pthick, max_pthick, &(geo->ar->be[ibe]->p_thick));
         }
         if (ind == 5)
         {
            changed = CheckUserFloatSliderValue(p_HubPoint[ind], geo->ar->be[ibe]->te_thick,
               min_tethick, max_tethick, &(geo->ar->be[ibe]->te_thick));
         }
         if (ind == 6)
         {
            changed = CheckUserFloatSliderValue(p_HubPoint[ind], geo->ar->be[ibe]->camb,
               min_camb, max_camb, &(geo->ar->be[ibe]->camb));
         }
         if (ind == 7)
         {
            changed = CheckUserFloatSliderValue(p_HubPoint[ind], geo->ar->be[ibe]->camb_pos,
               min_camb, max_camb, &(geo->ar->be[ibe]->camb_pos));
         }
         if (ind == 8)
         {
            changed = CheckUserFloatSliderValue(p_HubPoint[ind], geo->ar->be[ibe]->bp_shift,
               min_shift, max_shift, &(geo->ar->be[ibe]->bp_shift));
         }
      }
      else if (!strcmp(M_MIDDLE_POINT, pn))
      {
         ibe = middle; geo->ar->be_single = 0;
         if (ind == 0)
         {
            changed = CheckUserFloatSliderValue(p_InnerPoint[ind], geo->ar->be[ibe]->angle[0],
               min_angle, max_angle, &(geo->ar->be[ibe]->angle[0]));
         }
         if (ind == 1)
         {
            changed = CheckUserFloatSliderValue(p_InnerPoint[ind], geo->ar->be[ibe]->angle[1],
               min_angle, max_angle, &(geo->ar->be[ibe]->angle[1]));
         }
         if (ind == 2)
         {
            changed = CheckUserFloatSliderValue(p_InnerPoint[ind], geo->ar->be[ibe]->mod_angle[0],
               min_angle, max_angle, &(geo->ar->be[ibe]->mod_angle[0]));
         }
         if (ind == 3)
         {
            changed = CheckUserFloatSliderValue(p_InnerPoint[ind], geo->ar->be[ibe]->mod_angle[1],
               min_angle, max_angle, &(geo->ar->be[ibe]->mod_angle[1]));
         }
         if (ind == 4)
         {
            changed = CheckUserFloatSliderValue(p_InnerPoint[ind], geo->ar->be[ibe]->p_thick,
               min_pthick, max_pthick, &(geo->ar->be[ibe]->p_thick));
         }
         if (ind == 5)
         {
            changed = CheckUserFloatSliderValue(p_InnerPoint[ind], geo->ar->be[ibe]->te_thick,
               min_tethick, max_tethick, &(geo->ar->be[ibe]->te_thick));
         }
         if (ind == 6)
         {
            changed = CheckUserFloatSliderValue(p_InnerPoint[ind], geo->ar->be[ibe]->camb,
               min_camb, max_camb, &(geo->ar->be[ibe]->camb));
         }
         if (ind == 7)
         {
            changed = CheckUserFloatSliderValue(p_InnerPoint[ind], geo->ar->be[ibe]->camb_pos,
               min_camb, max_camb, &(geo->ar->be[ibe]->camb_pos));
         }
         if (ind == 8)
         {
            changed = CheckUserFloatSliderValue(p_InnerPoint[ind], geo->ar->be[ibe]->bp_shift,
               min_shift, max_shift, &(geo->ar->be[ibe]->bp_shift));
         }
      }
      else if (!strcmp(M_RIGHT_POINT, pn))
      {
         ibe = right; geo->ar->be_single = 0;
         if (ind == 0)
         {
            changed = CheckUserFloatSliderValue(p_ShroudPoint[ind], geo->ar->be[ibe]->angle[0],
               min_angle, max_angle, &(geo->ar->be[ibe]->angle[0]));
         }
         if (ind == 1)
         {
            changed = CheckUserFloatSliderValue(p_ShroudPoint[ind], geo->ar->be[ibe]->angle[1],
               min_angle, max_angle, &(geo->ar->be[ibe]->angle[1]));
         }
         if (ind == 2)
         {
            changed = CheckUserFloatSliderValue(p_ShroudPoint[ind], geo->ar->be[ibe]->mod_angle[0],
               min_angle, max_angle, &(geo->ar->be[ibe]->mod_angle[0]));
         }
         if (ind == 3)
         {
            changed = CheckUserFloatSliderValue(p_ShroudPoint[ind], geo->ar->be[ibe]->mod_angle[1],
               min_angle, max_angle, &(geo->ar->be[ibe]->mod_angle[1]));
         }
         if (ind == 4)
         {
            changed = CheckUserFloatSliderValue(p_ShroudPoint[ind], geo->ar->be[ibe]->p_thick,
               min_pthick, max_pthick, &(geo->ar->be[ibe]->p_thick));
         }
         if (ind == 5)
         {
            changed = CheckUserFloatSliderValue(p_ShroudPoint[ind], geo->ar->be[ibe]->te_thick,
               min_tethick, max_tethick, &(geo->ar->be[ibe]->te_thick));
         }
         if (ind == 6)
         {
            changed = CheckUserFloatSliderValue(p_ShroudPoint[ind], geo->ar->be[ibe]->camb,
               min_camb, max_camb, &(geo->ar->be[ibe]->camb));
         }
         if (ind == 7)
         {
            changed = CheckUserFloatSliderValue(p_ShroudPoint[ind], geo->ar->be[ibe]->camb_pos,
               min_camb, max_camb, &(geo->ar->be[ibe]->camb_pos));
         }
         if (ind == 8)
         {
            changed = CheckUserFloatSliderValue(p_ShroudPoint[ind], geo->ar->be[ibe]->bp_shift,
               min_shift, max_shift, &(geo->ar->be[ibe]->bp_shift));
         }
      }
      else
      {
         dprintf(2, "Sorry, no function for %s implemented\n", pn);
         changed = 0;
      }
      }
      else
      {
         dprintf(2, "index > ar->be_num\n", pn);
         changed = 0;
      }
   }
   else
   {
      dprintf(2, "Entering part with NO indices ...\n");
      // design data
      if (!strcmp(M_DESIGN_Q,pn))
         changed = CheckUserFloatValue(p_DDischarge,
            geo->ar->des->dis,
            0.0,1.0e+6,
            &(geo->ar->des->dis));
      else if (!strcmp(M_DESIGN_H,pn))
         changed = CheckUserFloatValue(p_DHead,
               geo->ar->des->head,
               0.0,1.0e+6,
               &(geo->ar->des->head));
      else if (!strcmp(M_DESIGN_N,pn))
         changed = CheckUserFloatValue(p_DRevolut,
               geo->ar->des->revs,
               0.0,1.0e+6,
               &(geo->ar->des->revs));
      else if (!strcmp(M_INLET_VRATIO,pn))
         changed = CheckUserFloatValue(p_DVRatio,
               geo->ar->des->vratio,
               0.0,1.0e+3,
               &(geo->ar->des->vratio));
      else if(!strcmp(M_DEFINE_VRATIO,pn))
      {
         changed = 1;
         geo->ar->vratio_flag =
            (int)(p_DDefineVRatio->getValue());
      }
      // global runner data
      else if (!strcmp(M_NO_BLADES, pn))
      {
         changed = CheckUserIntValue(p_NumberOfBlades, geo->ar->nob,
            2, 10, &(geo->ar->nob));
      }
      else if (!strcmp(M_OUTER_DIAMETER, pn))
      {
         if((changed = CheckUserFloatValue(p_OuterDiameter, geo->ar->ref,
            min_diam, max_diam, &(geo->ar->ref))) )
         {
            AxialRunner::Struct2CtrlPanel();
            dprintf(4,"FL: ref = %f, diam[1] = %f\n",geo->ar->ref,geo->ar->diam[1]);
         }
      }
      else if (!strcmp(M_BLADE_ENLACEMENT, pn))
      {
         changed = CheckUserFloatValue(p_BladeEnlacement, geo->ar->enlace,
            0.0, (float)(geo->ar->nob), &(geo->ar->enlace));
      }
      else if (!strcmp(M_PIVOT_LOCATION, pn))
      {
         changed = CheckUserFloatValue(p_PivotLocation, geo->ar->piv,
            0.0, 1.0, &(geo->ar->piv));
      }
      else if (!strcmp(M_BLADE_ANGLE, pn))
      {
         changed = CheckUserFloatSliderValue(p_BladeAngle,
            geo->ar->bangle,
            -90.0, 90.0,
            &(geo->ar->bangle));
      }
      else if(!strcmp(M_EULER_EQN,pn))
      {
         changed = 1;
         geo->ar->euler = (int)(p_EulerEqn->getValue());
      }
      else if(!strcmp(M_ROTATE_CLOCKWISE,pn))
      {
         changed = 1;
         geo->ar->rot_clockwise =
            (int)(p_RotateClockwise->getValue());
      }
      else if(!strcmp(M_FORCE_CAMB,pn))
      {
         changed = 1;
         geo->ar->clspline =
            (int)(p_ForceCamb->getValue());
      }
      else if(!strcmp(M_MODEL_INLET,pn))
      {
         changed = 1;
         geo->ar->mod->inl = (int)(p_ModelInlet->getValue());
         if(geo->ar->mod->bend == 0 && geo->ar->mod->inl == 1)
         {
            sendError("Modelling inlet but no bend makes no sense!");
            geo->ar->mod->bend = 1;
            p_ModelBend->setValue(geo->ar->mod->bend != 0 );
         }
      }
      else if(!strcmp(M_MODEL_BEND,pn))
      {
         changed = 1;
         geo->ar->mod->bend = (int)(p_ModelBend->getValue());
         if(geo->ar->mod->bend == 0 && geo->ar->mod->inl == 1)
         {
            sendError("Modelling inlet but no bend makes no sense!");
            geo->ar->mod->bend = 1;
            p_ModelBend->setValue(geo->ar->mod->bend != 0 );
         }
      }
      else if(!strcmp(M_MODEL_OUTLET,pn))
      {
         changed = 1;
         geo->ar->mod->outl = (int)(p_ModelOutlet->getValue());
      }
      else if(!strcmp(M_MODEL_ARB,pn))
      {
         changed = 1;
         geo->ar->mod->arbitrary = (int)(p_ModelArb->getValue());
         geo->ar->mod->bend = (int)(p_ModelArb->getValue());
         p_ModelBend->setValue(geo->ar->mod->bend != 0 );
      }
#ifdef   VATECH
      else if (!strcmp(M_DIMLESS_N, pn))
         changed = CheckUserFloatValue(p_nED,geo->ar->nED,-1.E+6,
               1.E+6, &(geo->ar->nED));
      else if (!strcmp(M_DIMLESS_Q, pn))
         changed = CheckUserFloatValue(p_QED,geo->ar->QED,-1.E+6,
               1.E+6, &(geo->ar->QED));
      else if (!strcmp(M_HEAD, pn))
         changed = CheckUserFloatValue(p_Head,geo->ar->H,-1.E+6,
               1.E+6, &(geo->ar->H));
      else if (!strcmp(M_FLOW, pn))
         changed = CheckUserFloatValue(p_Discharge,geo->ar->Q,-1.E+6,
               1.E+6, &(geo->ar->Q));
      else if (!strcmp(M_DIAMETER, pn))
         changed = CheckUserFloatValue(p_ProtoDiam,geo->ar->D1,-1.E+6,
               1.E+6, &(geo->ar->D1));
      else if (!strcmp(M_SPEED, pn))
         changed = CheckUserFloatValue(p_ProtoSpeed,geo->ar->n,
               -1.E+6,1.E+6, &(geo->ar->n));
      else if (!strcmp(M_ALPHA, pn))
         changed = CheckUserFloatValue(p_alpha,geo->ar->alpha,
               -1.E+6,1.E+6, &(geo->ar->alpha));
      else if (!strcmp(M_RUN_VATEULER,pn))
      {
         changed = (int)(p_RunVATEuler->getValue());
      }
      else if (!strcmp(M_OMEGA,pn))
      {
         int i = 0;
         while (multieu && multieu[i])
         {
            dprintf(2, "Rotiere Velocity: i=%d\n");
            CalcRelVelocity(multieu[i++], omega->getValue());
         }
         if (eu)  CalcRelVelocity(eu, omega->getValue());
         changed = 1;
      }
      else if (!strcmp(M_ROTATE_GRID,pn))
      {
         // der andere Fall ist implizit in compute abgehandelt ...
         if (!p_RotateGrid->getValue())
         {
            FreeMultiEu();
            dprintf(3, "%s: %d mltieu geloescht\n", M_ROTATE_GRID, p_RotateGrid->getValue());
         }
         dprintf(3, "%s: %d multieu eingeschaltet\n", M_ROTATE_GRID, p_RotateGrid->getValue());
         changed = 1;
      }
#endif                                      // VATECH

      // blade element specifications
#ifdef MODIFY_NOB
      else if (!strcmp(M_NO_BLADE_ELEMENTS, pn))
      {
         changed = CheckUserIntValue(p_NumberOfBladeElements, geo->ar->nob,
            5, MAX_ELEMENTS, &(geo->ar->nob));
      }
#endif                                      // MODIFY_NOB
      else if (!strcmp(M_BLADE_BIAS_TYPE, pn))
      {
         changed = CheckUserIntValue(p_BladeElementBiasType, geo->ar->be_type,
            0, 2, &(geo->ar->be_type));
      }
      else if (!strcmp(M_BLADE_BIAS_FACTOR, pn))
      {
         changed = CheckUserFloatValue(p_BladeElementBiasFactor, geo->ar->be_bias,
            0.0, 10.0, &(geo->ar->be_bias));
      }
      else if (!strcmp(M_LESPLINE_PARAMETERS, pn))
      {
         changed = CheckUserFloatVectorValue(p_LeSplineParameters,geo->ar->le_part,1.f,1.e-4f,0.9999f,geo->ar->le_part,3);
      }
      else if (!strcmp(M_TESPLINE_PARAMETERS, pn))
      {
         changed = CheckUserFloatVectorValue(p_TeSplineParameters,geo->ar->te_part,1.f,1.e-4f,0.9999f,geo->ar->te_part,2);
      }
      else if(!strcmp(M_LOCK_PTHICK,pn))
      {
         changed = 1; int i = LOCK_PTHICK;
         geo->ar->parlock[i] = (int)p_Lock[i]->getValue();
      }

      // machine dimensions data
      else if (!strcmp(M_HUB_DIAMETER, pn))
      {
         changed = CheckUserFloatAbsValue(p_HubDiameter, geo->ar->diam[0], Dref,
            min_diam, max_diam, &(geo->ar->diam[0]));
      }
      else if(!strcmp(M_INLET_HEIGHT, pn))
      {
         changed = CheckUserFloatAbsValue(p_InletExtHeight, geo->ar->h_inl_ext, Dref,
            min_height, max_height, &(geo->ar->h_inl_ext));
      }
      else if(!strcmp(M_INLET_DIAMETER, pn))
      {
         changed = CheckUserFloatAbsValue(p_InletExtDiameter, geo->ar->d_inl_ext, Dref,
            min_ext_diam, max_ext_diam, &(geo->ar->d_inl_ext));
      }
      else if(!strcmp(M_INLET_PITCH, pn))
      {
         changed = CheckUserFloatValue(p_InletPitch, geo->ar->arb_angle,
            0.0f, 89.99f, &(geo->ar->arb_angle));
      }
      else if(!strcmp(M_ARB_PART, pn))
      {
         changed = CheckUserFloatVectorValue(p_ArbPart, geo->ar->arb_part,
            1.0f,
            0.0f,1.0f,geo->ar->arb_part,2);
      }
      else if(!strcmp(M_SBEND_RAD, pn))
      {
         changed = CheckUserFloatVectorValue(p_ShroudRadius, geo->ar->r_shroud, Dref,
            min_ellipse,max_ellipse,geo->ar->r_shroud,2);
      }
      else if(!strcmp(M_SBEND_ANGLE, pn))
      {
         changed = CheckUserFloatValue(p_ShroudAngle, geo->ar->ang_shroud,
            min_angle, max_angle, &(geo->ar->ang_shroud));
      }
      else if(!strcmp(M_HBEND_CORNER_A, pn))
      {
         changed = CheckUserFloatAbsValue(p_HubCornerA, geo->ar->a_hub, Dref,
            min_ellipse, max_ellipse, &(geo->ar->a_hub));
      }
      else if(!strcmp(M_HBEND_CORNER_B, pn))
      {
         changed = CheckUserFloatAbsValue(p_HubCornerB, geo->ar->b_hub, Dref,
            min_ellipse, max_ellipse, &(geo->ar->b_hub));
      }
      // hub menu
      // FL: since s.o. changed the menu structure ...
      else if ( (!strcmp(M_GEO_MANIPULATION,pn) || !strcmp(M_HUB_SHROUD_BEND,pn))
         && p_BendSelection->getValue() == 1 )
      {
         if(p_HubBendModifyPoints->getValue())
         {
            p_HubPointSelected->show();
            p_HubPointValues->show();
         }
         else
         {
            p_HubPointSelected->hide();
            p_HubPointValues->hide();
         }
         SetFloatDoubleVector(p_HubCPointValues,
            REL2ABS(geo->ar->p_hubcap->x[p_HubCPointSelected->getValue()],Dref),
            REL2ABS(geo->ar->p_hubcap->z[p_HubCPointSelected->getValue()],Dref));
      }
      else if (!strcmp(M_HBEND_NOS, pn))
      {
         changed = CheckUserIntValue(p_HubBendNOS, geo->ar->hub_nos,
            min_nos, max_nos, &(geo->ar->hub_nos));
      }
      else if (!strcmp(M_HBEND_MODIFY, pn))
      {
         // leave changed unchanged :)
         if((geo->ar->hub_bmodpoints =
            p_HubBendModifyPoints->getValue()))
         {
            int index = p_HubPointSelected->getValue();
            p_HubPointSelected->show();
            p_HubPointValues->show();
            if(geo->ar->p_hbpoints->x)
            {
               SetFloatDoubleVector(p_HubPointValues,REL2ABS(geo->ar->p_hbpoints->x[index],Dref),
                  REL2ABS(geo->ar->p_hbpoints->z[index],Dref));
               dprintf(4," CheckUserInput ... getting values from corner point no %d\n",
                  p_HubPointSelected->getValue());
            }
         }
         else
         {
            p_HubPointSelected->hide();
            p_HubPointValues->hide();
         }
      }
      else if (!strcmp(M_POINT_DATA,pn))
      {
         if(CheckUserChoiceValue(p_HubPointSelected,p_HubBendNOS->getValue()))
         {
            int index = p_HubPointSelected->getValue();
            if(geo->ar->p_hbpoints && geo->ar->p_hbpoints->x)
            {
               SetFloatDoubleVector(p_HubPointValues,REL2ABS(geo->ar->p_hbpoints->x[index],Dref),
                  REL2ABS(geo->ar->p_hbpoints->z[index], Dref));
               dprintf(4," CheckUserInput ... getting values from corner point no %d\n",
                  p_HubPointSelected->getValue());
            }
         }
      }
      else if (!strcmp(M_HPOINT_VAL,pn))
      {
         int index = p_HubPointSelected->getValue();
         if(geo->ar->p_hbpoints)
         {
         float v[] = {geo->ar->p_hbpoints->x[index], geo->ar->p_hbpoints->z[index]};
         changed = CheckUserFloatVectorValue(p_HubPointValues,v,Dref,-max_diam,max_diam,v,2);
         geo->ar->p_hbpoints->x[index]=v[0];geo->ar->p_hbpoints->z[index]=v[1];
         dprintf(4," CheckUserInput ... setting values of corner point no %d(%f,%f)\n",
            p_HubPointSelected->getValue(),p_HubPointValues->getValue(0),p_HubPointValues->getValue(1));
         }
      }
      // cap
      else if (!strcmp(M_CPOINT_DATA,pn))
      {
         if(CheckUserChoiceValue(p_HubCPointSelected,geo->ar->cap_nop))
         {
            int index = p_HubCPointSelected->getValue();

            SetFloatDoubleVector(p_HubCPointValues,REL2ABS(geo->ar->p_hubcap->x[index],Dref),
               REL2ABS(geo->ar->p_hubcap->z[index], Dref));
         }
      }
      else if (!strcmp(M_HCPOINT_VAL,pn))
      {
         int index;
         float v[2];
         index = p_HubCPointSelected->getValue();
         v[0] = geo->ar->p_hubcap->x[index]; v[1] = geo->ar->p_hubcap->z[index];
         dprintf(4,"Entering -> CheckUserFloatVectorValue(), index = %d\n", index);
         //changed = CheckUserFloatVectorValue(p_HubCPointValues,v,Dref,min_diam,max_diam,v,2);
         changed = CheckUserFloatVectorValue(p_HubCPointValues,v,Dref,-3000,3000,v,2);
         geo->ar->p_hubcap->x[index] = v[0];
         geo->ar->p_hubcap->z[index] = v[1];
         dprintf(4," CheckUserInput ... setting values of corner point no %d(%f,%f)\n",
            p_HubCPointSelected->getValue(),
            p_HubCPointValues->getValue(0),
            p_HubCPointValues->getValue(1));
         dprintf(4," CheckUserInput ... v=[%f %f], x, z: %f, %f\n", v[0], v[1],
            geo->ar->p_hubcap->x[index], geo->ar->p_hubcap->z[index]);
      }
      else if (!strcmp(M_RUNNER_HEIGHT, pn))
      {
         changed = CheckUserFloatAbsValue(p_RunnerHeight, geo->ar->h_run, Dref,
            min_height, max_height, &(geo->ar->h_run));
      }
      else if (!strcmp(M_SSPHERE_DIAMETER, pn))
      {
         changed = CheckUserFloatAbsValue(p_ShroudSphereDiameter, geo->ar->d_shroud_sphere, Dref,
            min_ssphere_diam, max_ssphere_diam, &(geo->ar->d_shroud_sphere));
      }
      else if (!strcmp(M_SCOUNTER_ARC, pn))
      {
         changed = CheckUserIntValue(p_ShroudCounter, geo->ar->shroud_counter_rad,
            0, 1, &(geo->ar->shroud_counter_rad));
      }
      else if (!strcmp(M_SCOUNTER_NOS, pn))
      {
         changed = CheckUserIntValue(p_ShroudCounterNOS, geo->ar->counter_nos,
            min_nos, max_nos, &(geo->ar->counter_nos));
      }
      else if (!strcmp(M_SSPHERE_HEMI, pn))
      {
         changed = CheckUserIntValue(p_ShroudHemisphere, geo->ar->shroud_hemi,
            0, 1, &(geo->ar->shroud_hemi));
      }
      else if (!strcmp(M_HSPHERE_DIAMETER, pn))
      {
         changed = CheckUserFloatAbsValue(p_HubSphereDiameter, geo->ar->d_hub_sphere, Dref,
            min_hsphere_diam, max_hsphere_diam, &(geo->ar->d_hub_sphere));
      }
      else if (!strcmp(M_DRAFT_HEIGHT, pn))
      {
         changed = CheckUserFloatAbsValue(p_DraftHeight, geo->ar->h_draft, Dref,
            min_height, max_height, &(geo->ar->h_draft));
      }
      else if (!strcmp(M_DRAFT_DIAMETER, pn))
      {
         changed = CheckUserFloatAbsValue(p_DraftDiameter, geo->ar->d_draft, Dref,
            geo->ar->diam[1], max_ssphere_diam, &(geo->ar->d_draft));
      }
      else if (!strcmp(M_DRAFT_ANGLE, pn))
      {
         changed = CheckUserFloatValue(p_DraftAngle, geo->ar->ang_draft,
            min_angle, max_angle, &(geo->ar->ang_draft));
      }
      // blade edge data
      else if (!strcmp(M_LE_SHROUD_CON, pn))
      {
         changed = CheckUserFloatValue(p_LEShroudConstriction, geo->ar->le->con[1],
            min_con, max_con, &(geo->ar->le->con[1]));
      }
      else if (!strcmp(M_LE_HUB_CON, pn))
      {
         changed = CheckUserFloatValue(p_LEHubConstriction, geo->ar->le->con[0],
            min_con, max_con, &(geo->ar->le->con[0]));
      }
      else if (!strcmp(M_TE_SHROUD_CON, pn))
      {
         changed = CheckUserFloatValue(p_TEShroudConstriction, geo->ar->te->con[1],
            min_con, max_con, &(geo->ar->te->con[1]));
      }
      else if (!strcmp(M_TE_HUB_CON, pn))
      {
         changed = CheckUserFloatValue(p_TEHubConstriction, geo->ar->te->con[0],
            min_con, max_con, &(geo->ar->te->con[0]));
      }
      else if (!strcmp(M_LE_NO_CON, pn))
      {
         changed = CheckUserFloatValue(p_LENoConstriction, geo->ar->le->nocon,
            0.0, 1.0, &(geo->ar->le->nocon));
      }
      else if (!strcmp(M_TE_NO_CON, pn))
      {
         changed = CheckUserFloatValue(p_TENoConstriction, geo->ar->te->nocon,
            0.0, 1.0, &(geo->ar->te->nocon));
      }
      // ***** Grid parameters, only if rrg exists!!!
      else if(!strcmp("makeGrid",pn));
      else if(!strcmp("writeGrid",pn) && rrg)
         rrg->write_grid = (int)p_writeGrid->getValue();
      else if(!strcmp(M_CREATE_BC,pn) && rrg)
         rrg->create_inbc = (int)p_createBC->getValue();
      else if(!strcmp(M_MESH_EXT,pn) && rrg)
         rrg->mesh_ext = (int)p_meshExt->getValue();
      else if(!strcmp(M_ROT_EXT,pn) && rrg)
         rrg->rot_ext = (int)p_rotExt->getValue();
      else if(!strcmp(M_USE_Q,pn) && rrg)
         rrg->inbc->useQ = (int)p_useQ->getValue();
      else if(!strcmp(M_USE_ALPHA,pn) && rrg)
         rrg->inbc->useAlpha = (int)p_useAlpha->getValue();
      else if(!strcmp(M_CONST_ALPHA,pn) && rrg)
         rrg->alpha_const = (int)p_constAlpha->getValue();
      else if(!strcmp(M_TURB_PROF,pn) && rrg)
         rrg->turb_prof = (int)p_turbProf->getValue();
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
      else if(!strcmp(M_BCALPHA,pn) && rrg)
      {
         if((changed = CheckUserFloatValue(p_bcAlpha,
            GRAD(rrg->inbc->bcAlpha),0.0,
            180.0,
            &(rrg->inbc->bcAlpha))))
            rrg->inbc->bcAlpha =
               RAD(p_bcAlpha->getValue());
      }
      else if(!strcmp(M_RUN_FEN,pn));
      else if(!strcmp(M_SHOW_COMPLETE_GRID,pn)) changed = 1;
      else if(!strcmp(M_LAYERS2SHOW,pn) && rrg)
      {
         if(p_GridLayers->getValue(1) < p_GridLayers->getValue(0))
         {
            sendError(" max value < min value!");
            p_GridLayers->setValue(1,p_GridLayers->getValue(0));
            if(p_GridLayers->getValue(1) > rrg->ge_num-1)
            {
               sendError(" max layer-to-show value  > than number of layers!");
               p_GridLayers->setValue(0, rrg->ge_num-1);
               p_GridLayers->setValue(1, rrg->ge_num-1);
            }
         }
         else if(p_GridLayers->getValue(1) > rrg->ge_num-1)
         {
            sendError(" max layer-to-show value  > than number of layers!");
            p_GridLayers->setValue(1, rrg->ge_num-1);
         }
         else if(p_GridLayers->getValue(0) < 1)
         {
            sendError(" min layer-to-show value < than ZERO!");
            p_GridLayers->setValue(0,1);
         }
         changed = 1;
      }
      else if(!strcmp(M_GRID_MERIDS,pn) && rrg)
         changed = CheckDiscretization(p_GridMerids, &rrg->ge_num,
               &rrg->ge_bias, &rrg->ge_type);
      else if(!strcmp(M_CIRCUMF_DIS,pn) && rrg)
      {
         changed = CheckDiscretization(p_CircumfDis, &rrg->cdis,
            &rrg->cbias, &rrg->cbias_type);
      }
      else if(!strcmp(M_CIRCUMF_DIS_LE,pn) && rrg)
      {
         changed = CheckDiscretization(p_CircumfDisLe, &rrg->cledis,
            &rrg->clebias, &rrg->clebias_type);
      }
      else if(!strcmp(M_MERID_INLET,pn) && rrg)
         changed = CheckDiscretization(p_MeridInletDis, &rrg->ssmdis,
               &rrg->ssmbias, &rrg->ssmbias_type);
      else if(!strcmp(M_PS_DIS,pn) && rrg)
      {
         if(p_PSDis->getValue(0) >= rrg->ssdis)
         {
            sendError(" Value for '%s(0)' must be smaller than '%s(0)'! Last change ignored!",
               M_PS_DIS,M_SS_DIS);
            p_PSDis->setValue(0,float(rrg->psdis));
         }
         changed = CheckDiscretization(p_PSDis, &rrg->psdis,
            &rrg->psbias, &rrg->psbias_type);
      }
      else if(!strcmp(M_SS_DIS,pn) && rrg)
      {
         if(p_SSDis->getValue(0) <= rrg->psdis)
         {
            sendError(" Value for '%s(0)' must be bigger than '%s(0)'! Last change ignored!",
               M_SS_DIS,M_PS_DIS);
            p_SSDis->setValue(0, float(rrg->ssdis));
         }
         changed = CheckDiscretization(p_SSDis, &rrg->ssdis,
            &rrg->ssbias, &rrg->ssbias_type);
      }
      else if(!strcmp(M_BL_DIS,pn) && rrg)
         changed = CheckDiscretization(p_BLDis, &rrg->psedis,
               &rrg->psebias, &rrg->psebias_type);
      else if(!strcmp(M_MERID_OUTLET,pn) && rrg)
      {
         changed = CheckDiscretization(p_MeridOutletDis, &rrg->lowdis,
            &rrg->lowbias, &rrg->lowbias_type);
         rrg->lowindis    = rrg->lowdis;
         p_OutletCoreDis->setValue(0,float(rrg->lowdis));
      }
      else if(!strcmp(M_OUTLET_CORE,pn) && rrg)
      {
         changed = CheckDiscretization(p_OutletCoreDis, &rrg->lowindis,
            &rrg->lowinbias, &rrg->lowin_type);
         if(rrg->lowindis != rrg->lowdis)
         {
            sendError(" Value for '%s(0)' will be ignored. Must be identical to '%s(0)'",
               M_OUTLET_CORE,M_MERID_OUTLET);
            rrg->lowindis    = rrg->lowdis;
            p_OutletCoreDis->setValue(0,float(rrg->lowdis));
         }

      }
#ifndef NO_INLET_EXT
      else if(!strcmp(M_MERID_INEXT,pn) && rrg)
         changed = CheckDiscretization(p_MeridInExtDis, &rrg->extdis,
               &rrg->extbias, &rrg->extbias_type);
#endif
      else if(!strcmp(M_SKEW_INLET,pn) && rrg)
         rrg->skew_runin = (int)p_SkewInlet->getValue();
      else if(!strcmp(M_PHI_SCALE,pn) && rrg)
      {
         changed = CheckUserFloatVectorValue(p_PhiScale, rrg->phi_scale,1.0,
            -5.0,5.0,rrg->phi_scale,2);
      }

      else if(!strcmp(M_PHI_SKEW,pn) && rrg)
      {
         changed = CheckUserFloatVectorValue(p_PhiSkew, rrg->phi_skew,1.0,
            -10.0,10.0,rrg->phi_skew,2);
      }
      else if(!strcmp(M_PHI_SKEWOUT,pn) && rrg) {
	  changed = CheckUserFloatVectorValue(p_PhiSkewOut, rrg->phi_skewout,
					   1.0,-10.0,10.0,rrg->phi_skewout,2);
      }
      else if(!strcmp(M_BOUND_LAY_RATIO,pn) && rrg)
      {
         rrg->bl_scale[0]  = 1.0f-p_BoundLayRatio->getValue(0);
         rrg->bl_scale[1]  = 1.0f-p_BoundLayRatio->getValue(1);
      }
      else if(!strcmp(M_V14_ANGLE,pn) && rrg)
      {
         for(int i = 0; i < 2; i++)
         {
            if((changed = CheckUserFloatVectorValue2(p_V14Angle, rrg->v14_angle[i],
               0.0,90.0,&rrg->v14_angle[i],i)))
               rrg->v14_angle[i] = RAD(rrg->v14_angle[i]);
            else p_V14Angle->setValue(i,GRAD(rrg->v14_angle[i]));
         }
      }
      else if(!strcmp(M_BLV14_PART,pn) && rrg)
      {
         changed = CheckUserFloatVectorValue(p_BlV14Part, rrg->bl_v14_part,1.0,
            0.0f,1.0f,rrg->bl_v14_part,2);
      }
      else if(!strcmp(M_SS_PART,pn) && rrg)
      {
         changed = CheckUserFloatVectorValue(p_SSPart, rrg->ss_part,
            1.0f, 0.01f,0.99f,rrg->ss_part,2);
      }
      else if(!strcmp(M_PS_PART,pn) && rrg)
      {
         changed = CheckUserFloatVectorValue(p_PSPart, rrg->ps_part,
            1.0f,0.01f,0.99f,rrg->ps_part,2);
      }
      else if(!strcmp(M_SSLE_PART,pn) && rrg)
      {
         changed = CheckUserFloatVectorValue(p_SSLePart,
            rrg->ssle_part,1.0f,
            0.00001f,0.1f,
            rrg->ssle_part,2);
      }
      else if(!strcmp(M_PSLE_PART,pn) && rrg)
      {
         changed = CheckUserFloatVectorValue(p_PSLePart,
            rrg->psle_part,1.0,
            0.00001f,0.1f,
            rrg->psle_part,2);
      }
      else if(!strcmp(M_OUT_PART,pn) && rrg)
      {
         if((changed = CheckUserFloatVectorValue2(p_OutPart, rrg->out_part[0],
            0.01f,0.49f,&rrg->out_part[0],0)))
         {
            rrg->out_part[1] = 1.0f - rrg->out_part[0];
            p_OutPart->setValue(1,rrg->out_part[1]);
         }
         else
         {
            if((changed = CheckUserFloatVectorValue2(p_OutPart, rrg->out_part[1],
               0.51f,0.99f,&rrg->out_part[1],1)))
            {
               rrg->out_part[0] = 1.0f - rrg->out_part[1];
               p_OutPart->setValue(0,rrg->out_part[0]);
            }
         }
      }
      else if(!strcmp(M_GRIDTYPE,pn) && rrg)
      {
         if(rrg)
         {
            changed = CheckUserChoiceValue(p_GridTypeChoice, 2);
            fprintf(stderr," p_GridTypeChoice = %d\n", p_GridTypeChoice->getValue());
            rrg->type = p_GridTypeChoice->getValue() + 1;
         }
         else
            sendWarning(" Grid does not exist yet! Selection has no effect!");
      }
      else if(!strcmp(M_PHI_EXT,pn) && rrg)
         changed = CheckUserFloatValue(p_PhiExtScale, rrg->phi0_ext,
               -5.0f,5.0f,&rrg->phi0_ext);
      else if(!strcmp(M_RRIN_ANGLE,pn) && rrg)
      {
         for(int i = 0; i < 2; i++)
         {
            if((changed = CheckUserFloatVectorValue2(p_RRInAngle, rrg->angle_ext[i],
               0.0f,90.0f,&rrg->angle_ext[i],i)))
               rrg->angle_ext[i] = RAD(rrg->angle_ext[i]);
            else p_RRInAngle->setValue(i,GRAD(rrg->angle_ext[i]));
         }
      }
      else
      {
         dprintf(3, "Sorry, no function for %s implemented\n", pn);
         changed = 0;
      }
      // FL:|| ... is necessary since 'hide' is not
      // executed in CreateMenu-Method (COVISE-bug?)
      if(!p_ForceCamb->getValue())
         p_TeSplineParameters->hide();
      if(p_ForceCamb->getValue() &&
         (p_TypeSelected->getValue() == 5))
         p_TeSplineParameters->show();

#ifdef VATECH
      if (!strcmp(M_OPERATING_POINT, pn) ||
         p_TypeSelected->getValue() == 1)
      {
         dprintf(6,"data:\n nED=%f\n QED=%f\n",
            geo->ar->nED,geo->ar->QED);
         dprintf(6,"data:\n H=%f\n Q=%f\n n=%f\n D1=%f\n",
            geo->ar->H,geo->ar->Q,geo->ar->n,geo->ar->D1);
         if(p_OpPoint->getValue() == 0)
         {
            p_nED->show();
            p_QED->show();
            p_alpha->show();
            p_Head->hide();
            p_Discharge->hide();
            p_ProtoSpeed->hide();
            p_ProtoDiam->hide();
         }
         else
         {
            p_nED->hide();
            p_QED->hide();
            p_alpha->hide();
            p_Head->show();
            p_Discharge->show();
            p_ProtoSpeed->show();
            p_ProtoDiam->show();
         }
      }
#endif                                      // VATECH
      dprintf(2, "Endof part with NO indices ...\n", pn);
      if(p_TypeSelected->getValue() == 1)
      {
         if(p_DDefineVRatio->getValue())
            p_DVRatio->show();
         else p_DVRatio->hide();
      }
      else p_DVRatio->hide();
   }
   dprintf(1, "Leaving AxialRunner::CheckUserInput: changed=%d\n", changed);
   return changed;
}

