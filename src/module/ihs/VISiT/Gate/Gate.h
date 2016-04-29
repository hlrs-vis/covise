#ifndef _GG_SET_H
#define _GG_SET_H

#include <api/coSimpleModule.h>
#include "util/coviseCompat.h"
#include <General/include/geo.h>
#include <Gate/include/gate.h>
#include <Gate/include/ga2cov.h>

using namespace covise;

class Gate : public coSimpleModule
{
   COMODULE

   private:

      enum { MAX_ELEMENTS=20 };

      // Gate.cpp

#ifdef YAC
      virtual void paramChanged(coParam *param);
#endif

      virtual int   compute(const char *port);
      virtual void  param(const char *, bool inMapLoading);
      virtual void  postInst();
      virtual void  CreateUserMenu(void);
      virtual void  CreateMenuGateData(void);
      virtual void  CreateMenuMeridianData(void);
      virtual void  CreateMenuProfileData(void);
      virtual void  CreateMenuGatePNumbers(void);
      virtual void  CreateMenuGateCurveLengths(void);
      virtual void  CreateMenuGateCompressions(void);
      virtual void  CreateMenuGateShifts(void);
      virtual void  Struct2CtrlPanel(void);
      virtual void  CtrlPanel2Struct(void);
      int CheckUserInput(const char *portname, struct geometry *g);
#ifndef YAC
      coDistributedObject *GenerateNormals(int part, coDoPolygons *poly, const char *out_name);
#else
      coDistributedObject *GenerateNormals(int part, coDoPolygons *poly, coObjInfo out_name);
#endif

      // in ModLib.cpp
      virtual int   CheckUserFloatValue(coFloatParam *f, float old, float min, float max, float *dest);
      virtual int   CheckUserFloatVectorValue(coFloatVectorParam *f, int index, float old, float min, float max, float *dest);
      virtual int   CheckUserFloatSliderValue(coFloatSliderParam *f, float old, float min, float max, float *dest);
      virtual int   CheckUserIntValue(coIntScalarParam *f, int old, int min, int max, int *dest);
      virtual int   SplitPortname(const char *portname, char *name, int *index);
#ifndef YAC
      virtual void  quit();
#else
      virtual int  quit();
#endif

      coOutputPort   *grid;

      coOutputPort   *blade;
      coOutputPort   *hub;
      coOutputPort   *shroud;

      coOutputPort   *bladenormals;
      coOutputPort   *hubnormals;
      coOutputPort   *shroudnormals;

      coOutputPort   *bcin;
      coOutputPort   *bcout;
      coOutputPort   *bcwall;
      coOutputPort   *bcperiodic;

      coOutputPort   *boco;
      coOutputPort   *plot2d;

      coFileBrowserParam *startFile;

      int isInitialized;

      struct geometry   *geo;

      // menue sections
#define M_GATE_DATA              "gate_data"
#define M_MERIDIAN_DATA          "meridian_contour data"
#define M_BLADE_PROFILE_DATA     "blade_profile_data"
#define M_GRID_DATA              "grid_data"
#define M_GRID_DATA_POINTS       "grid_point numbers"
#define M_GRID_DATA_LENGTH       "grid_length paras"
#define M_GRID_DATA_COMPRESSION     "grid_compression paras"
#define M_GRID_DATA_SHIFT        "grid_shift_paras"

      // general parameters
#define  M_GENERATE_GRID            "make_grid"
#define  M_LOCK_GRID              "lock_make_grid_button"
#define  M_GEO_FROM_FILE            "read_geometry_from_file"
#define M_SAVE_GRID              "save_grid_geo_rb"
#define M_RADIAL_GATE            "radial_gate"

      // gate data
#define M_Q_OPT                  "Q_opt_m3_s"
#define M_N_OPT                  "n_opt_1_s"
#define M_H                   "H_m"
#define  M_Q                     "Q_m3_s"
#define M_N                   "n_1_s"
#define  M_NUMBER_OF_BLADES         "number_of_blades"
#define M_PIVOT_RADIUS           "blade_axis_radius"
#define  M_BLADE_ANGLE           "blade_angle"

      // meridian contour data
#define  M_INLET_HEIGHT          "inlet_height"
#define  M_INLET_RADIUS          "inlet_radius"
#define  M_INLET_Z               "inlet_z_shroud"
#define  M_OUTLET_INNER_RADIUS      "outlet_inner_radius"
#define  M_OUTLET_OUTER_RADIUS      "outlet_outer_radius"
#define M_OUTLET_Z               "outlet_z"
#define  M_SHROUD_RADIUS            "shroud_radius_a_b"
#define  M_HUB_RADIUS            "hub_radius_a_b"
#define  M_HUB_ARC_POINTS        "n_points_hub_arc"

      // blade element data
#define M_BLADE_BIAS_FACTOR         "blade_element_bias_factor"
#define M_BLADE_BIAS_TYPE        "blade_element_bias_type"
#define M_RADIAL_PARAMETER       "radial_parameter"
#define M_CHORD_LENGTH           "chord_length"
#define M_CHORD_PIVOT            "pivot_location"
#define M_CHORD_ANGLE            "chord_angle"
#define M_PROFILE_THICKNESS         "profile_thickness"
#define M_MAXIMUM_CAMBER         "maximum_camber"
#define M_PROFILE_SHIFT          "blade_profile_shift"

      //grid data
#define M_EDGE_PS             "blade_ps_area_border_percent"
#define M_EDGE_SS             "blade_ss_area_border_percent"
#define M_BOUND_LAYER            "boundary_layer_thickness"
#define M_N_RAD                  "n_radial"
#define M_N_BOUND             "n_boundary_layer"
#define M_N_OUT                  "n_outlet"
#define M_N_IN                "n_inlet"
#define M_N_PS_BACK              "n_ps_back"
#define M_N_PS_FRONT          "n_ps_front"
#define M_N_SS_BACK              "n_ss_back"
#define M_N_SS_FRONT          "n_ss_front"
#define M_LEN_OUT_HUB            "outlet_hub_area_start_percent"
#define M_LEN_OUT_SHROUD         "outlet_shroud_area_start_percent"
#define M_LEN_EXPAND_IN          "len_inlet_expansion"
#define M_LEN_EXPAND_OUT         "len_outlet_expansion"
#define M_COMP_PS_BACK           "comp_ps_back"
#define M_COMP_PS_FRONT          "comp_ps_front"
#define M_COMP_SS_BACK           "comp_ss_back"
#define M_COMP_SS_FRONT          "comp_ss_front"
#define M_COMP_TRAIL          "comp_trail"
#define M_COMP_OUT               "comp_outlet"
#define M_COMP_IN             "comp_inlet"
#define M_COMP_BOUND          "comp_boundary_layer"
#define M_COMP_MIDDLE            "comp_middle"
#define M_COMP_RAD               "comp_radial"
      //#define M_SHIFT_IN					"shift inlet"
#define M_SHIFT_OUT              "shift_outlet"

      // gate
      coFloatSliderParam   *p_Q_opt;
      coFloatSliderParam   *p_Q;
      coFloatSliderParam   *p_n_opt;
      coFloatSliderParam   *p_n;
      coFloatSliderParam   *p_H;
      coIntScalarParam  *p_NumberOfBlades;
      coFloatParam   *p_PivotRadius;        //Blade rotation axis x, y (open & close)
      coFloatSliderParam   *p_BladeAngle;

      // axial gate (standard) or radial gate
      coBooleanParam    *p_radialGate;

      // geometry from file
      coBooleanParam    *p_GeoFromFile;

      // hub & shroud parameters
      coFloatParam   *p_InletHeight;
      coFloatParam   *p_InletRadius;
      coFloatParam   *p_InletZ;
      coFloatParam   *p_OutletInnerRadius;
      coFloatParam   *p_OutletOuterRadius;
      coFloatParam   *p_OutletZ;
      coFloatVectorParam   *p_ShroudAB;
      coFloatVectorParam   *p_HubAB;
      coIntScalarParam  *p_HubArcPoints;

      // blade parameters
      coFloatParam   *p_ChordLength;
      coFloatParam   *p_PivotLocation;
      coFloatParam   *p_ChordAngle;
      coFloatParam   *p_ProfileThickness;
      coFloatParam   *p_MaximumCamber;
      coFloatParam   *p_ProfileShift;

      // Button for triggering Grid Generation
      coBooleanParam      *p_makeGrid;
      coBooleanParam      *p_lockmakeGrid;

      // Button for saving Grid (*.rb & *.geo - files)
      coBooleanParam      *p_saveGrid;

      // grid parameters
      // border position, boundary layer thickness
      coIntScalarParam  *p_grid_edge_ps;
      coIntScalarParam  *p_grid_edge_ss;
      coFloatParam   *p_grid_bound_layer;
      // number of points
      coIntScalarParam  *p_grid_n_rad;
      coIntScalarParam  *p_grid_n_bound;
      coIntScalarParam  *p_grid_n_out;
      coIntScalarParam  *p_grid_n_in;
      coIntScalarParam     *p_grid_n_blade_ps_back;
      coIntScalarParam     *p_grid_n_blade_ps_front;
      coIntScalarParam  *p_grid_n_blade_ss_back;
      coIntScalarParam  *p_grid_n_blade_ss_front;
      // lengths
      coIntSliderParam  *p_grid_len_start_out_hub;
      coIntSliderParam  *p_grid_len_start_out_shroud;
      coFloatParam   *p_grid_len_expand_in;
      coFloatParam   *p_grid_len_expand_out;
      // compressions
      coFloatParam   *p_grid_comp_ps_back;
      coFloatParam   *p_grid_comp_ps_front;
      coFloatParam   *p_grid_comp_ss_back;
      coFloatParam   *p_grid_comp_ss_front;
      coFloatParam   *p_grid_comp_trail;
      coFloatParam   *p_grid_comp_out;
      coFloatParam   *p_grid_comp_in;
      coFloatParam   *p_grid_comp_bound;
      coFloatParam   *p_grid_comp_middle;
      coFloatParam   *p_grid_comp_rad;
      // shifts
      coFloatParam   *p_grid_shift_out;

   public:

      Gate(int argc, char *argv[]);

};
#endif
