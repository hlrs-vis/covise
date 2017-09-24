#ifndef _RR_SET_H
#define _RR_SET_H

#include <api/coModule.h>
using namespace covise;

#include <General/include/geo.h>
#include <General/include/log.h>
#include <RadialRunner/include/radial.h>
#include <RadialRunner/include/rr2cov.h>
#include <RunnerGridGen/include/rr_grid.h>
#include <RunnerGridGen/include/mesh.h>

#include <util/coviseCompat.h>

#define  DIM(x)   (sizeof(x)/sizeof(*x))
#define RAD(x) (float((x) * M_PI/180.0))
#define GRAD(x)   (float((x) * 180.0/M_PI))

#ifndef YAC
class RadialRunner : public coModule
#else
class RadialRunner : public coSimpleModule
#endif
{
   COMODULE

   private:

      enum { MAX_ELEMENTS=30  };

      // RadialRunner.cpp
      virtual int   compute(const char *port);
      virtual void  param(const char *, bool inMapLoading);
      virtual void  postInst(void);
#ifndef YAC
      virtual void  quit();
#else
      virtual int  quit();
#endif
      virtual void  CreateUserMenu();
      virtual void  CreatePlotPortMenu(int);
      virtual void  CreatePortMenu();
      virtual void  CreateMenuConformalView(int);
      virtual void  CreateMenuCamber(int);
      virtual void  CreateMenuNormCamber(int);
      virtual void  CreateMenuRunnerData();
      virtual void  CreateMenuSpecials();
      virtual void  CreateMenuDesignData();
      virtual void  CreatePlot();
      virtual void  CreateGrid();
#ifdef CREATE_PROFILE_MENU
      virtual void  CreateMenuProfileData();
#endif                                         // CREATE_PROFILE_MENU
      virtual void  CreateMenuLeadingEdge();
      virtual void  CreateMenuTrailingEdge();
      virtual void  CreateMenuBladeData();
      virtual void  CreateMenuGridData();
      virtual void  ReducedModifyMenu();
      virtual void  AddOneParam(const char *p);
      virtual char* IndexedParameterName(const char *, int);
      virtual void  Struct2CtrlPanel(void);
      virtual void BladeElements2CtrlPanel(void);
      virtual void BladeElements2Reduced(void);
      virtual int   SplitPortname(const char *, char *, int *);
      virtual int  CheckUserInput(const char *, struct geometry *, struct rr_grid *);
      virtual int  CheckUserFloatValue(coFloatParam *f, float old,
         float min, float max, float *dest);
      virtual int CheckUserFloatSliderValue(coFloatSliderParam *f, float old,
         float min, float max, float *dest);
      virtual int  CheckUserIntValue(coIntScalarParam *f, int old, int min,
         int max, int *dest);
      virtual int CheckUserChoiceValue( coChoiceParam *c, int max);
      virtual int  CheckUserChoiceValue2(coChoiceParam *f, int old, int min,
         int max, int *dest);
      virtual int CheckUserFloatVectorValue(coFloatVectorParam *v, float *old,
         float min, float max, float *dest,int c);
      virtual int CheckUserFloatVectorValue2(coFloatVectorParam *v, float old,
         float min, float max, float *dest,int c);
      virtual  void ShowHideExtended(int flag);
      virtual void ShowHideModifiedOptions(int type);
      virtual void CreateMenuBasicGrid();
      virtual void CreateMenuGridTopo();
      virtual void CreateMenuBCandCFD();
      virtual void Grid2CtrlPanel(void);
      virtual int CheckDiscretization(coFloatVectorParam *v, int *dis, float *bias, int *type);

      coOutputPort *blade;
      coOutputPort *hub;
      coOutputPort *shroud;
      coOutputPort *grid;
      coOutputPort *bcin;
      coOutputPort *bcout;
      coOutputPort *boco;
      coOutputPort *bcwall;
      coOutputPort *bcblade;
      coOutputPort *bcperiodic;
      
      //@atismer: contains a set of coDoIntArr that define the boundary element faces
      //          necessary for writing a cgns file
      coOutputPort *boundaryElementFaces;      

      coFileBrowserParam      *startFile;

      int isInitialized;
      int gridInitialized;

      struct geometry *geo;
      struct rr_grid *rrg;

      // menue sections
#define M_EULER_EQN                     "use_eulers_equation"
#define M_PUMP                          "make_a_pump"
#define M_CAMBER_POS                     "force_maximum_camber"
#define M_CAMB2SURF                     "select_camber_function"
#define M_CAMB_RESULT                   "from_contour_and_angles"
#define M_CAMB_TEFIX                    "using_function_startat_te"
#define M_CAMB_LEFIX                    "using_function_startat_le"
#define M_CAMB_CLLEN                    "function_and_cl_length"
#define NUM_CAMBFUNCS  4
#define M_SHOW_EXTENSION                       "show_meridian_extensions"
#define M_EXTENDED_MENU                "extended_menu"
#define  M_DESIGN_DATA           "design_data"
#define  M_SPECIALS           "special_buttons"
#define  M_RUNNER_DATA           "runner_data"
#define  M_BLADE_PROFILE_DATA    "blade_profile"
#define  M_LEADING_EDGE_DATA        "leading_edge"
#define  M_TRAILING_EDGE_DATA    "trailing_edge"
#define  M_BLADE_DATA            "blade_elements"
#define  M_GRID_DATA           "grid_parameters"
#define  M_GRID_PARA_SELECT        "select_parameter_set"
#define  M_BLADE_ELEMENT_DATA    "blade_element_data"
#define M_REDUCED_MODIFY         "reduced_modification"
#define M_BASIC_GRID                    "basic_grid_parameters"
#define M_GRID_TOPOLOGY                        "grid_topology_parameters"
#define M_BC_AND_CFD                        "bc_and_cfd"

      // global runner data
#define M_STRAIGHT_HUB                    "hub_InOut_straight"
#define M_STRAIGHT_SHRD                    "shroud_InOut_straight"
#define M_ROTATE_CLOCKWISE                   "rotation_clockwise"
#define M_WRITE_BLADEDATA                    "write_blade_data"
#define  M_NUMBER_OF_BLADES         "number_of_blades"
#define  M_OUTLET_DIAMETER_ABS      "outlet_diameter_absolute"
#define  M_INLET_DIAMETER_REL    "inlet_diameter_relative"
#define  M_SHROUD_HEIGHT_DIFF    "shroud_height_difference"
#ifdef GAP
#define  M_GAP_WIDTH       "gap_width"
#endif
#define  M_CONDUIT_WIDTH         "conduit_width_in_out"
#define  M_CONTOUR_ANGLES     "contour_angles_in_out"
#define  M_INLET_OPEN_ANGLES     "inlet_opening_angles_hub_shroud"
#define  M_OUTLET_OPEN_ANGLES    "outlet_opening_angles_hub_shroud"
#define M_HUB_CURVE_PARAMETERS          "hub_curve_parameters"
#define M_SHROUD_CURVE_PARAMETERS       "shroud_curve_parameters"
#define M_HUB_STRAIGHT_PARAMETERS          "hub_straight_parameters"
#define M_SHROUD_STRAIGHT_PARAMETERS       "shroud_straight_parameters"

      // design data
#define M_DESIGN_Q                      "discharge"
#define M_DESIGN_H                      "head"
#define M_DESIGN_N                      "revolutions"
#define M_DEFINE_VRATIO                 "define_velocity_ratio"
#define M_INLET_VRATIO                  "le_velocity_ratio"

      // extension
#define M_INLET_ANGLE_EXT           "extension_inlet_angle"
#define M_HEIGHT_EXT                  "extension_shroud_height_in_out"
#define M_DIAM_EXT                 "extension_shroud_diameter_in_out"
#define M_WIDTH_EXT                   "extension_conduit_width_in_out"
#define M_HUB_CURVE_PEXT                "extension_hub_curve_params"
#define M_SHROUD_CURVE_PEXT             "extension_shroud_curve_params"

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
#define M_BCQ                           "discharge4CFD"
#define M_USE_Q                         "use_Q4BCvalues"
#define M_BCH                           "head4CFD"
#define M_BCALPHA                       "flow_angle"
#define M_USE_ALPHA                     "use_this_angle"

#ifdef CREATE_PROFILE_MENU
      // runner profile data
#define  M_NUMBER_OF_PROFILE_SEC    "number_of_profile_sections"
#define  M_REL_CHORD       "relative_chord"
#define  M_REL_THICKNESS         "relative_thickness"
#endif                                         // CREATE_PROFILE_MENU

      // blade edge data
#define  M_LE_HUB_PARM        "leading_edge_hub_parameter"
#define M_LE_HUB_ANGLE        "leading_edge_hub_off_contour_angle"
#define  M_LE_SHROUD_PARM     "leading_edge_shroud_parameter"
#define M_LE_SHROUD_ANGLE     "leading_edge_shroud_off_contour_angle"
#define M_LE_CURVE_PARAMETER            "leading_edge_curve_parameters"
#define  M_TE_HUB_PARM        "trailing_edge_hub_parameter"
#define M_TE_HUB_ANGLE        "trailing_edge_hub_off_contour_angle"
#define  M_TE_SHROUD_PARM     "trailing_edge_shroud_parameter"
#define M_TE_SHROUD_ANGLE         "trailing_edge_shroud_off_contour_angle"
#define M_TE_CURVE_PARAMETER            "trailing_edge_curve_parameters"

      // blade element specifications
#define  M_NUMBER_OF_BLADE_ELEMENTS "number_of_blade_elements"
#define M_BLADE_BIAS_FACTOR         "blade_element_bias_factor"
#define M_BLADE_BIAS_TYPE        "blade_element_bias_type"

      // blade element data
#define  M_MERIDIAN_PARAMETER    "meridian_parameter"
#define  M_INLET_ANGLE           "inlet_angle"
#define  M_OUTLET_ANGLE          "outlet_angle"
#define  M_PROFILE_THICKNESS        "profile_thickness"
#define  M_TE_THICKNESS          "trailing_edge_thickness"
#define  M_CENTRE_LINE_CAMBER    "centre_line_camber"
#define M_TE_WRAP_ANGLE          "trailing_edge_wrap_angle"
#define M_BL_WRAP_ANGLE          "blade_wrap_angle"
#define  M_BLADE_LENGTH_FACTOR    "blade_length_factor"
#define M_PROFILE_SHIFT          "blade_profile_shift"
#define  M_INLET_ANGLE_MODIFICATION "inlet_angle_modification"
#define  M_OUTLET_ANGLE_MODIFICATION   "outlet_angle_modification"
#define  M_CENTRE_LINE_CAMBER_POSN  "centre_line_camber_position"
#define  M_CAMBPARA  "camber_line_parametre"
#define M_REMAINING_SWIRL        "remaining_swirl"
#define M_BLADE_LESPLINE_PARAS         "le_spline_parameter"
#define M_BLADE_TESPLINE_PARAS         "te_spline_parameter"

      // plot data select menue
#define NUM_PLOT_PORTS   2

#define M_2DPORT                        "_2DPort"
#define M_2DPLOT                        "_2DPlot"
#define M_MERIDIAN_CONTOUR_PLOT         "meridian_contour"
#define M_MERIDIAN_CONTOUR_EXT          "meridian_contour_extended"
#define M_CONFORMAL_VIEW                "conformal_view"
#define M_SHOW_CONFORMAL                "show_view"
#define M_CAMBER                        "camber"
#define M_SHOW_CAMBER                   "show_camber"
#define M_NORMCAMBER                        "normalized_camber"
#define M_SHOW_NORMCAMBER                   "show_normalized_camber"
#define M_THICKNESS                     "thickness_distribution"
#define M_OVERLAP                       "overlap"
#define M_BLADE_ANGLES                  "real_blade_angles"
#define M_EULER_ANGLES                  "euler_blade_angles"
#define M_MERIDIAN_VELOCITY             "meridian_velocities"
#define M_CIRCUMF_VELOCITY             "circumferential_velocities"

      coBooleanParam          *p_makeGrid;
      coBooleanParam    *p_WriteBladeData;
      coBooleanParam    *p_RotateClockwise;
      coBooleanParam    *p_StraightHub;
      coBooleanParam    *p_StraightShrd;
      coBooleanParam          *p_EulerEqn;
      coBooleanParam          *p_Pump;
      coChoiceParam          *p_Camb2Surf;
      coBooleanParam          *p_CamberPos;
      coBooleanParam          *p_ShowExtensions;
      coBooleanParam          *p_ExtendedMenu;
      coIntScalarParam  *p_NumberOfBlades;
      coFloatParam   *p_OutletDiameterAbs;
      coFloatParam   *p_InletDiameterRel;
      coFloatParam   *p_ShroudHeightDiff;
#ifdef GAP
      coFloatParam   *p_GapWidth;
#endif
      coFloatVectorParam   *p_ConduitWidth;
      coFloatVectorParam   *p_ContourAngles;
      coFloatVectorParam   *p_InletOpenAngles;
      coFloatVectorParam   *p_OutletOpenAngles;
      coFloatVectorParam      *p_HubCurveParameters;
      coFloatVectorParam      *p_ShroudCurveParameters;
      coFloatVectorParam      *p_HubStraightParameters;
      coFloatVectorParam      *p_ShroudStraightParameters;

      // extended geometry
      coFloatParam         *p_InletAngleExt;
      coFloatVectorParam   *p_HeightExt;
      coFloatVectorParam   *p_DiamExt;
      coFloatVectorParam   *p_WidthExt;
      coFloatVectorParam   *p_HubCurveParaExt;
      coFloatVectorParam   *p_ShroudCurveParaExt;

      // design data
      coFloatParam      *p_DDischarge;
      coFloatParam      *p_DHead;
      coFloatParam      *p_DRevolut;
      coBooleanParam    *p_DDefineVRatio;
      coFloatParam      *p_DVRatio;

      // grid parameters
      coBooleanParam          *p_ShowComplete;
      coBooleanParam          *p_SkewInlet;
      coIntVectorParam        *p_GridLayers;
      coFloatVectorParam      *p_GridMerids;
      coFloatVectorParam      *p_CircumfDis;
      coFloatVectorParam      *p_CircumfDisLe;
      coFloatVectorParam      *p_MeridInletDis;
      coFloatVectorParam      *p_PSDis;
      coFloatVectorParam      *p_SSDis;
      coFloatVectorParam      *p_BLDis;
      coFloatVectorParam      *p_MeridOutletDis;
      coFloatVectorParam      *p_MeridInExtDis;

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
      coBooleanParam          *p_useQ;
      coBooleanParam          *p_createBC;

#ifdef CREATE_PROFILE_MENUE
      coIntScalarParam  *p_NumberOfProfileSecs;
      coFloatParam   *p_RelativeChord;
      coFloatParam   *p_RelativeThickness;
#endif                                         // CREATE_PROFILE_MENUE

      // leading edge
      coFloatParam   *p_LeHubParm;
      coFloatParam   *p_LeHubAngle;
      coFloatParam   *p_LeShroudParm;
      coFloatParam   *p_LeShroudAngle;
      coFloatVectorParam      *p_LeCurveParam;
      // trailing edge
      coFloatParam   *p_TeHubParm;
      coFloatParam   *p_TeHubAngle;
      coFloatParam   *p_TeShroudParm;
      coFloatParam   *p_TeShroudAngle;
      coFloatVectorParam      *p_TeCurveParam;

      coIntScalarParam  *p_NumberOfBladeElements;
      coFloatParam   *p_BladeElementBiasFactor;
      coIntScalarParam  *p_BladeElementBiasType;

      coFloatParam   *p_MeridianParm[MAX_ELEMENTS];
      coFloatParam   *p_InletAngle[MAX_ELEMENTS];
      coFloatParam   *p_OutletAngle[MAX_ELEMENTS];
      coFloatParam   *p_ProfileThickness[MAX_ELEMENTS];
      coFloatParam   *p_TrailingEdgeThickness[MAX_ELEMENTS];
      coFloatParam   *p_CentreLineCamber[MAX_ELEMENTS];
      coFloatParam   *p_TrailingEdgeWrap[MAX_ELEMENTS];
      coFloatParam   *p_BladeWrap[MAX_ELEMENTS];
      coFloatParam   *p_ProfileShift[MAX_ELEMENTS];
      coFloatParam   *p_InletAngleModification[MAX_ELEMENTS];
      coFloatParam   *p_OutletAngleModification[MAX_ELEMENTS];
      coFloatParam   *p_CentreLineCamberPosn[MAX_ELEMENTS];
      coFloatParam   *p_RemainingSwirl[MAX_ELEMENTS];
      coFloatParam   *p_BladeLePara[MAX_ELEMENTS];
      coFloatParam   *p_BladeTePara[MAX_ELEMENTS];
      coFloatParam   *p_CambPara[MAX_ELEMENTS];
      coFloatParam   *p_BladeLengthFactor[MAX_ELEMENTS];

      coChoiceParam           *m_2DplotChoice[NUM_PLOT_PORTS];
      coChoiceParam           *p_GridTypeChoice;
      coChoiceParam           *m_types;
      coChoiceParam           *m_paraset;
      coOutputPort            *plot2d[NUM_PLOT_PORTS];
      coBooleanParam          *p_ShowConformal[MAX_ELEMENTS][NUM_PLOT_PORTS];
      coBooleanParam          *p_ShowCamber[MAX_ELEMENTS][NUM_PLOT_PORTS];
      coBooleanParam          *p_ShowNormCamber[MAX_ELEMENTS][NUM_PLOT_PORTS];

      // reduced modify menu
#define  MAX_MODIFY  50
      char **ReducedModifyMenuPoints;
      int numReducedMenuPoints;

      // plot menue set parameter
      int PlotSelection[NUM_PLOT_PORTS];

#define  M_LEFT_POINT   "hub_point"
#define  M_MIDDLE_POINT "inner_point"
#define  M_RIGHT_POINT  "shroud_point"
      coFloatSliderParam  *p_HubPoint[MAX_MODIFY];
      coFloatSliderParam  *p_InnerPoint[MAX_MODIFY];
      coFloatSliderParam  *p_ShroudPoint[MAX_MODIFY];

   public:

      RadialRunner(int argc, char *argv[]);
};
#endif
