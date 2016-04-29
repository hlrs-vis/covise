#ifndef _AR_SET_H
#define _AR_SET_H

#include "../lib/RunnerGridGen/include/rr_grid.h"
#include "../lib/RunnerGridGen/include/mesh.h"

#include <api/coModule.h>
using namespace covise;

#include <../lib/General/include/geo.h>
#include <../lib/AxialRunner/include/axial.h>
#include <../lib/AxialRunner/include/ar2cov.h>

#ifdef   VATECH
// -I../../VA_TECH
#include <ReadEuler/EuGri.h>
#endif                                            // VATECH

#define MAX_POINTS      20
#define INIT_POLY_PART  0.4

class AxialRunner : public coModule
{

   private:

      enum { MAX_ELEMENTS=20 };

      // AxialRunner.cpp
      virtual int   compute(const char *port);
      virtual void  param(const char *, bool inMapLoading);
      virtual void  quit();
      virtual void  CreateUserMenu();
      virtual void  CreatePortMenu();
      virtual void  CreatePlotPortMenu(int);
      virtual void  CreateMenuConformalView(int);
      virtual void  CreateMenuCamber(int);
      virtual void  CreateMenuNormCamber(int);
#ifdef VATECH
      virtual void  CreateMenuVATCFDSwitches();
#endif
      virtual void  CreateMenuDesignData();
      virtual void  CreateMenuRunnerData();
      virtual void  CreateMenuMachineDimensions();

	  // MiscLib.cpp
      virtual void  CreatePlot();
      virtual void  CreateGrid();
      virtual void  CreateGeo();
#ifdef CREATE_PROFILE_MENU
      virtual void  CreateMenuProfileData();
#endif                                         // CREATE_PROFILE_MENU
      virtual void  CreateMenuBladeData();
      virtual void  CreateMenuBladeEdges();
      virtual void  CreateMenuGridData();
      virtual void  ReducedModifyMenu();
      virtual void  CreateHubCornerMenu();
      virtual void  CreateHubCapMenu();
      virtual void  AddOneParam(const char *p);
      virtual void  Struct2CtrlPanel(void);
      virtual void  BladeElements2CtrlPanel(void);
      virtual void  BladeElements2Reduced(void);
      virtual char* IndexedParameterName(const char *, int);
      virtual int   SplitPortname(const char *, char *, int *);
      virtual int   CheckUserInput(const char *, struct geometry *, struct rr_grid *);
      virtual int   SetFloatDoubleVector(coFloatVectorParam *v, float val0, float val1);
      virtual int   CheckUserFloatValue(coFloatParam *f, float old, float min, float max, float *dest);
      virtual int   CheckUserFloatAbsValue(coFloatParam *f, float old, float scale,
         float min, float max, float *dest);
      virtual int   CheckUserFloatSliderAbsValue(coFloatSliderParam *f, float old, float scale,
         float min, float max, float *dest);
      virtual int   CheckUserFloatVectorValue(coFloatVectorParam *v, float *old, float scale,
         float min, float max, float *dest,int c);
      virtual int CheckUserFloatVectorValue2(coFloatVectorParam *v, float old,
         float min, float max, float *dest,int c);
      virtual int   CheckUserFloatSliderValue(coFloatSliderParam *f, float old, float min, float max,
         float *dest);
      virtual int   CheckUserChoiceValue( coChoiceParam *c, int max);
      virtual int   CheckUserBooleanValue(coBooleanParam *b, int old,
         int min, int max, int *dest);
      virtual int   CheckUserIntValue(coIntScalarParam *f, int old, int min, int max, int *dest);
      virtual void CreateMenuBasicGrid();
      virtual void CreateMenuGridTopo();
      virtual void CreateMenuBCandCFD();
      virtual void Grid2CtrlPanel(void);
      virtual int CheckDiscretization(coFloatVectorParam *v, int *dis, float *bias, int *type);
      virtual void ShowHideModifiedOptions(int type);



      coOutputPort     *blade;
      coOutputPort     *hub;
      coOutputPort     *shroud;
      coOutputPort     *grid;
      coOutputPort     *bcin;
      coOutputPort     *bcout;
      coOutputPort     *boco;
      coOutputPort     *bcwall;
      coOutputPort     *bcblade;
      coOutputPort     *bcperiodic;
      
      ///contains a set of coDoIntArr that define the boundary element faces
      coOutputPort *boundaryElementFaces;

      coFileBrowserParam *startFile;

      int isInitialized;
      int gridInitialized;

      struct geometry   *geo;
      struct rr_grid *rrg;

      // menu sections
#define M_EULER_EQN                     "use_eulers_equation"
#define M_MODEL_INLET                   "model_inlet_extension"
#define M_MODEL_BEND                   "model_bend_region"
#define M_MODEL_OUTLET                   "model_outlet_extension"
#define M_MODEL_ARB                   "model_inlet_arbitrary"
#define M_RUNNER_DATA            "global_runner_configuration"
#define M_MACHINE_DIMS           "machine_dimensions_normalized_values"
#define M_BLADE_EDGES            "blade_edge_configuration"
#define M_BLADE_PROFILE_DATA     "blade_profile_configuration"
#define M_BLADE_DATA          "blade_elements"
#define M_BLADE_ELEMENT_DATA     "blade_element_configuration"
#define M_REDUCED_MODIFY         "reduced_modification"
#define M_VAT_CFDSWITCHES                       "CFD_switches"
#define  M_DESIGN_DATA           "design_data"
#define  M_GRID_DATA           "grid_parameters"
#define  M_GRID_PARA_SELECT        "select_parameter_set"
#define M_BASIC_GRID                    "basic_grid_parameters"
#define M_GRID_TOPOLOGY                        "grid_topology_parameters"
#define M_BC_AND_CFD                        "bc_and_cfd"

      // VAT Euler Switches
#define M_RUN_VATEULER                          "run_Euler"
#define M_SHOW_VATRES                           "show_results"
#define M_ROTATE_GRID                           "rotate_grid"

      // design data
#define M_DESIGN_Q                      "discharge_opt"
#define M_DESIGN_H                      "head_opt"
#define M_DESIGN_N                      "revolutions"
#define M_DEFINE_VRATIO                 "define_velocity_ratio"
#define M_INLET_VRATIO                  "inlet_velocity_ratio"

      // global runner data
#define M_WRITE_BLADEDATA                    "write_blade_data"
#define M_FORCE_CAMB                   "force_camber"
#define M_ROTATE_CLOCKWISE                   "rotation_clockwise"
#define  M_NO_BLADES             "number_of_blades"
#define  M_OUTER_DIAMETER        "shroud_diameter_reference"
#define M_BLADE_ENLACEMENT       "blade_enlacement"
#define M_PIVOT_LOCATION         "pivot_location"
#define  M_BLADE_ANGLE           "blade_angle"
#define M_OPERATING_POINT        "specify_operating_point_by"
#define M_DIMLESS_N              "dimensionless_speed"
#define M_DIMLESS_Q           "dimensionless_discharge"
#define M_SPEED                  "speed_prototype"
#define M_HEAD                "head"
#define M_DIAMETER               "diameter_prototype"
#define M_FLOW                "discharge"
#define M_ALPHA                  "alpha"
#define M_OMEGA                  "Omega"

      // machine dimension data
#define M_HUB_DIAMETER           "hub_diameter"
#define M_INLET_HEIGHT           "inlet_extension_height"
#define M_INLET_DIAMETER         "inlet_extension_diameter"
#define M_INLET_PITCH         "inlet_pitch_angle"
#define M_ARB_PART         "arbitrary_inlet_spline_part"
#define M_HBEND_CORNER_A         "semi_vertical_axis_hub_bend"
#define M_HBEND_CORNER_B         "semi_horizontal_axis_hub_bend"
#define M_HBEND_NOS           "number_of_sections_hub_bend"
#define M_SBEND_RAD               "shroud_bend_radius_start_end"
#define M_SBEND_RAD1          "shroud_bend_start_radius"
#define M_SBEND_ANGLE            "shroud_bend_start_arc_angle"
#define M_SBEND_RAD2          "shroud_bend_end_radius"
#define M_RUNNER_HEIGHT          "runner_height_below_inlet"
#define M_SSPHERE_DIAMETER       "shroud_sphere_diameter"
#define M_SSPHERE_HEMI           "shroud_hemisphere"
#define M_SCOUNTER_ARC           "shroud_sphere_counter_arc"
#define M_SCOUNTER_NOS               "number_of_sections_counter_arc"
#define M_HSPHERE_DIAMETER       "hub_sphere_diameter"
#define M_DRAFT_HEIGHT              "draft_tube_inlet_height_below_runner"
#define M_DRAFT_DIAMETER         "draft_tube_inlet_diameter"
#define M_DRAFT_ANGLE            "draft_tube_opening_angle"
#define M_GEO_MANIPULATION       "geometry_manipulation"
#define M_HUB_CORNER          "hub_corner"
#define M_SHROUD_CORNER          "shroud_corner"
#define M_HBEND_MODIFY           "modify_hub_bend_points"
#define M_POINT_DATA          "select_point_to_modify"
#define M_HPOINT_VAL          "hub_point_diameter_and_height"
#define M_CPOINT_DATA            "select_cap_point_to_modify"
#define M_HCPOINT_VAL                  "hub_cap_point_diameter_and_height"
#define M_INLET                  "inlet_section"
#define M_RUNNER              "runner_section"
#define M_BEND                "bend_section"
#define M_HUB                 "hub"
#define M_SHROUD              "shroud"
#define M_OUTLET           "oulet_section_draft_tube"
#define M_HUB_SHROUD          "hub_and_shroud"
#define M_HUB_SHROUD_BEND        "hub_and_shroud_bend"
#define M_HUB_SHROUD_RUNNER         "hub_and_shroud_runner"

#ifdef CREATE_PROFILE_MENU
      // blade profile data
#define M_NO_PROFILE_SEC         "number_of_profile_sections"
#endif                                         // CREATE_PROFILE_MENU

      // blade element specifications
#define M_NO_BLADE_ELEMENTS         "number_of_blade_elements"
#define M_BLADE_BIAS_FACTOR         "blade_element_bias_factor"
#define M_BLADE_BIAS_TYPE        "blade_element_bias_type"
#define M_LESPLINE_PARAMETERS       "le_spline_parameters"
#define M_TESPLINE_PARAMETERS       "te_spline_parameters"
#define M_LOCK_PTHICK                           "lock_profile_thickness"

      // blade element data
#define M_INLET_ANGLE            "inlet_angle"
#define M_OUTLET_ANGLE           "outlet_angle"
#define  M_INLET_ANGLE_MODIFICATION "inlet_angle_modification"
#define  M_OUTLET_ANGLE_MODIFICATION   "outlet_angle_modification"
#define M_PROFILE_THICKNESS         "profile_thickness"
#define M_TE_THICKNESS           "trailing_edge_thickness"
#define M_MAXIMUM_CAMBER         "maximum_camber"
#define M_CAMBER_POSITION        "camber_position"
#define M_PROFILE_SHIFT          "blade_profile_shift"

      // blade edge data
#define M_TE_SHROUD_CON        "shroud_constriction_trailing_edge"
#define M_TE_HUB_CON        "hub_constriction_trailing_edge"
#define M_TE_NO_CON         "zero_constriction_trailing_edge"
#define M_LE_SHROUD_CON        "shroud_constriction_leading_edge"
#define M_LE_HUB_CON        "hub_constriction_leading_edge"
#define M_LE_NO_CON                      "zero_constriction_leading_edge"

      // basic grid parameters
#define M_GRIDTYPE                      "grid_type"
#define M_SHOW_COMPLETE_GRID            "show_complete_grid"
#define M_LAYERS2SHOW                   "layers_to_show_min_max"
#define M_B2B_CLASSIC                   "b2b_classic"
#define M_B2B_MODIFIED                  "b2b_modified"
#define M_GRID_MERIDS                   "radial_discretization"
#define M_CIRCUMF_DIS                   "circumferential_discretization"
#define M_CIRCUMF_DIS_LE                "circumf_discr_at_leading_edge"
#define M_MERID_INLET                   "meridional_discr_runner_inlet"
#define M_PS_DIS                        "pressure_side_discretization"
#define M_SS_DIS                        "suction_side_discretization"
#define M_BL_DIS                        "boundary_layer_discretization"
#define M_MERID_OUTLET                  "meridional_outlet"
#define M_OUTLET_CORE                   "meridional_outlet_core_region"
#define M_MERID_INEXT                   "meridional_inlet_extension"

      // grid topology params.
#define M_SKEW_INLET                    "radial_skew_runner_in_outlet"
#define M_PHI_SCALE                     "angle_coordinate_scaling_factor"
#define M_PHI_SKEW                      "angle_factor_for_skewed_inlet"
#define M_PHI_SKEWOUT                   "angle_factor_for_skewed_outlet"
#define M_BOUND_LAY_RATIO               "boundary_layer_thickness_ratio"
#define M_V14_ANGLE                     "le_b2b_spline_vector_angles_ss_ps"
#define M_BLV14_PART                    "partition_bound_layer_le_spline"
#define M_SS_PART                       "ss_partition_ratio_hub_shroud"
#define M_PS_PART                       "ps_partition_ratio_hub_shroud"
#define M_SSLE_PART                     "ratio_ssle_blade_hub_shroud"
#define M_PSLE_PART                     "ratio_psle_blade_hub_shroud"
#define M_OUT_PART                      "circumf_partition_outlet"
#define M_PHI_EXT                       "angle_coord_scaling_inlet_ext"
#define M_RRIN_ANGLE                    "runner_inlet_b2b_angles_ss_ps"

      // bc and cfd.
#define M_CREATE_BC                     "create_inlet_bc"
#define M_MESH_EXT                      "mesh_inlet_extension"
#define M_ROT_EXT                       "all_walls_rotating"
#define M_RUN_FEN                       "runFENFLOSS"
#define M_BCQ                           "discharge4CFD"
#define M_USE_Q                         "use_Q4BCvalues"
#define M_BCH                           "head4CFD"
#define M_BCALPHA                       "flow_angle"
#define M_USE_ALPHA                     "use_this_angle"
#define M_CONST_ALPHA                   "constant_inlet_angle"
#define M_TURB_PROF                     "turbulent_inlet_profile"

      // plot data select menue
#define NUM_PLOT_PORTS   2

#define M_2DPORT                        "_2DPort"
#define M_2DPLOT                        "_2DPlot"
#define M_MERIDIAN_CONTOUR_PLOT         "meridian_contour"
#define M_CONFORMAL_VIEW                "conformal_view"
#define M_SHOW_CONFORMAL                "show_view"
#define M_CAMBER                        "camber"
#define M_NORMCAMBER                    "normalized_camber"
#define M_SHOW_CAMBER                   "show_camber"
#define M_THICKNESS                     "thickness_distribution"
#define M_MAXTHICK                      "max_thickness_position"
#define M_OVERLAP                       "overlap"
#define M_BLADE_ANGLES                  "real_blade_angles"
#define M_SHOW_NORMCAMBER               "show_normcamb"
#define M_EULER_ANGLES                  "euler_blade_angles"
#define M_CHORD_ANGLE                   "chord_angle"
#define M_MERIDIAN_VELOCITY             "meridian_velocities"
#define M_PARAM_SLIDER                  "slider parameter"

#ifdef   VATECH
      virtual void AxialRunner::sockData(int soc);
      virtual void postInst();
      virtual void AxialRunner::SetFiFoCommandStatus(char *cmd);
      virtual void AxialRunner::ProcessSockCommand(char *cmd);
      virtual void AxialRunner::ShowEuler(struct EuGri **);
      virtual void AxialRunner::SetFileValue(coFileBrowserParam *f, const char *co, const char *en, const char *pat);
      virtual void AxialRunner::FreeMultiEu();
      // VAT Euler Switches
      coBooleanParam          *p_RunVATEuler;
      coBooleanParam          *p_RotateGrid;
      FILE *fifoin;
      char fifofilein[256];
      char fifofileout[256];
      struct EuGri *eu;
      struct EuGri **multieu;
      coOutputPort *gridOutPort;
      coOutputPort *velocityOutPort;
      coOutputPort *relVelocityOutPort;
      coOutputPort *pressureOutPort;
      coFloatParam *omega;
#define  M_EU_GRID_FILE "GridFile"
      coFileBrowserParam *filenameGrid;
      char *lfilenameGrid;
#define  M_EU_EULER_FILE   "EulerFile"
      coFileBrowserParam *filenameEuler;
      char *lfilenameEuler;
#endif                                         // VATECH

      coChoiceParam           *p_TypeSelected;
      coBooleanParam          *p_makeGrid;
      coBooleanParam          *p_EulerEqn;
      coBooleanParam          *p_ModelInlet;
      coBooleanParam          *p_ModelBend;
      coBooleanParam          *p_ModelOutlet;
      coBooleanParam          *p_ModelArb;

      // design data
      coFloatParam      *p_DDischarge;
      coFloatParam      *p_DHead;
      coFloatParam      *p_DRevolut;
      coBooleanParam          *p_DDefineVRatio;
      coFloatParam      *p_DVRatio;

      // global runner data
      coBooleanParam    *p_WriteBladeData;
      coBooleanParam    *p_RotateClockwise;
      coBooleanParam    *p_ForceCamb;
      coIntScalarParam  *p_NumberOfBlades;
      coFloatParam   *p_OuterDiameter;
      coFloatParam   *p_BladeEnlacement;
      coFloatParam   *p_PivotLocation;
      coFloatSliderParam   *p_BladeAngle;
#ifdef   VATECH
      coChoiceParam     *p_OpPoint;
      coFloatParam      *p_Head;
      coFloatParam      *p_Discharge;
      coFloatParam      *p_ProtoDiam;
      coFloatParam      *p_ProtoSpeed;
      coFloatParam      *p_nED;
      coFloatParam      *p_QED;
      coFloatParam      *p_alpha;
#endif                                         // VATECH

      // machine dimension data
      coBooleanParam    *p_HubBendModifyPoints;
      coBooleanParam    *p_HubCapModifyPoints;
      coChoiceParam     *p_HubPointSelected;
      coChoiceParam     *p_HubCPointSelected;
      coChoiceParam           *p_BendSelection;
      coIntScalarParam  *p_HubBendNOS;
      coIntScalarParam  *p_ShroudHemisphere;
      coIntScalarParam  *p_ShroudCounter;
      coIntScalarParam  *p_ShroudCounterNOS;
      coFloatParam   *p_HubDiameter;
      coFloatParam   *p_InletExtHeight;
      coFloatParam   *p_InletExtDiameter;
      coFloatParam   *p_InletPitch;
      coFloatParam   *p_ShroudRadius1;
      coFloatParam   *p_ShroudAngle;
      coFloatParam   *p_ShroudRadius2;
      coFloatParam   *p_HubCornerB;
      coFloatParam   *p_HubCornerA;
      coFloatParam   *p_RunnerHeight;
      coFloatParam   *p_ShroudSphereDiameter;
      coFloatParam   *p_HubSphereDiameter;
      coFloatParam   *p_DraftHeight;
      coFloatParam   *p_DraftDiameter;
      coFloatParam   *p_DraftAngle;
      coFloatVectorParam   *p_ShroudRadius;
      coFloatVectorParam   *p_HubPointValues;
      coFloatVectorParam   *p_HubCPointValues;
      coFloatVectorParam   *p_ArbPart;

#ifdef CREATE_PROFILE_MENU
      coFloatParam   *p_NumberOfProfileSections;
#endif                                         // CREATE_PROFILE_MENU

      // blade elements, bias data
      coIntScalarParam  *p_NumberOfBladeElements;
      coFloatParam   *p_BladeElementBiasFactor;
      coIntScalarParam  *p_BladeElementBiasType;
      coFloatVectorParam   *p_LeSplineParameters;
      coFloatVectorParam   *p_TeSplineParameters;
      coBooleanParam    *p_Lock[NUM_PARLOCK];

      // blade element data arrays
      coFloatParam   *p_InletAngle[MAX_ELEMENTS];
      coFloatParam   *p_OutletAngle[MAX_ELEMENTS];
      coFloatParam   *p_InletAngleModification[MAX_ELEMENTS];
      coFloatParam   *p_OutletAngleModification[MAX_ELEMENTS];
      coFloatParam   *p_ProfileThickness[MAX_ELEMENTS];
      coFloatParam   *p_TrailingEdgeThickness[MAX_ELEMENTS];
      coFloatParam   *p_MaximumCamber[MAX_ELEMENTS];
      coFloatParam   *p_CamberPosition[MAX_ELEMENTS];
      coFloatParam   *p_ProfileShift[MAX_ELEMENTS];
      coChoiceParam           *m_sliderChoice;

      // blade edge data
      coFloatParam   *p_LEShroudConstriction;
      coFloatParam   *p_LEHubConstriction;
      coFloatParam   *p_LENoConstriction;
      coFloatParam   *p_TEShroudConstriction;
      coFloatParam   *p_TEHubConstriction;
      coFloatParam   *p_TENoConstriction;

      // plot variables
      coChoiceParam           *m_2DplotChoice[NUM_PLOT_PORTS];
      coBooleanParam          *p_ShowConformal[MAX_ELEMENTS][NUM_PLOT_PORTS];
      coBooleanParam          *p_ShowCamber[MAX_ELEMENTS][NUM_PLOT_PORTS];
      coBooleanParam          *p_ShowNormCamber[MAX_ELEMENTS][NUM_PLOT_PORTS];
      coOutputPort               *plot2d[NUM_PLOT_PORTS];

      // grid parameters
      coBooleanParam          *p_ShowComplete;
      coIntVectorParam        *p_GridLayers;
      coChoiceParam           *p_GridTypeChoice;
      coBooleanParam          *p_SkewInlet;
      coFloatVectorParam      *p_GridMerids;
      coFloatVectorParam      *p_CircumfDis;
      coFloatVectorParam      *p_CircumfDisLe;
      coFloatVectorParam      *p_MeridInletDis;
      coFloatVectorParam      *p_PSDis;
      coFloatVectorParam      *p_SSDis;
      coFloatVectorParam      *p_BLDis;
      coFloatVectorParam      *p_MeridOutletDis;
      coFloatVectorParam      *p_OutletCoreDis;
      coFloatVectorParam      *p_MeridInExtDis;
      coChoiceParam           *m_paraset;

      // grid topology params
      coFloatVectorParam      *p_PhiScale;
      coFloatVectorParam      *p_PhiSkew;
      coFloatVectorParam      *p_PhiSkewOut;
      coFloatVectorParam      *p_BoundLayRatio;
      coFloatVectorParam      *p_V14Angle;
      coFloatVectorParam      *p_BlV14Part;
      coFloatVectorParam      *p_SSPart;
      coFloatVectorParam      *p_PSPart;
      coFloatVectorParam      *p_SSLePart;
      coFloatVectorParam      *p_PSLePart;
      coFloatVectorParam      *p_OutPart;
      coFloatParam      *p_PhiExtScale;
      coFloatVectorParam      *p_RRInAngle;

      // grid bc values
      coBooleanParam          *p_writeGrid;
      coBooleanParam          *p_RunFENFLOSS;
      coBooleanParam          *p_meshExt;
      coBooleanParam          *p_rotExt;
      coFloatParam      *p_bcQ;
      coFloatParam      *p_bcH;
      coFloatParam      *p_bcAlpha;
      coBooleanParam          *p_useAlpha;
      coBooleanParam          *p_constAlpha;
      coBooleanParam          *p_turbProf;
      coBooleanParam          *p_useQ;
      coBooleanParam          *p_createBC;

      // reduced modify menu
#define MAX_MODIFY   30
      char **ReducedModifyMenuPoints;
      int numReducedMenuPoints;

#define M_LEFT_POINT "hub_point"
#define M_MIDDLE_POINT  "inner_point"
#define M_RIGHT_POINT   "shroud_point"
      coFloatSliderParam   *p_HubPoint[MAX_MODIFY];
      coFloatSliderParam   *p_InnerPoint[MAX_MODIFY];
      coFloatSliderParam   *p_ShroudPoint[MAX_MODIFY];

   public:

      AxialRunner(int argc, char *argv[]);
};
#endif
