#ifndef _GG_SET_H
#define _GG_SET_H

#include "api/coModule.h"
#include <General/include/geo.h>
#include <RadialRunner/include/radial.h>
#include <RadialRunner/include/rr2cov.h>

#define  DIM(x)   (sizeof(x)/sizeof(*x))
#define RAD(x) ((x) * M_PI/180.0)
#define GRAD(x)   ((x) * 180.0/M_PI)

class RadialRunner : public coModule
{

   private:

      enum { MAX_ELEMENTS=30  };

      // RadialRunner.cpp
      virtual int   compute(const char *port);
      virtual void  param(const char *);
      virtual void  postInst();
      virtual void  quit();
      virtual void  CreateUserMenu();
      virtual void  CreateMenuRunnerData();
#ifdef CREATE_PROFILE_MENU
      virtual void  CreateMenuProfileData();
#endif                                         // CREATE_PROFILE_MENU
      virtual void  CreateMenuLeadingEdge();
      virtual void  CreateMenuTrailingEdge();
      virtual void  CreateMenuBladeData();
      virtual void  ReducedModifyMenu();
      virtual void  AddOneParam(const char *p);
      virtual char* IndexedParameterName(const char *, int);
      virtual void  Struct2CtrlPanel(void);
      virtual int   SplitPortname(const char *, char *, int *);
      virtual int  CheckUserInput(const char *, struct geometry *);
      virtual int  CheckUserFloatValue(coFloatParam *f, float old,
         float min, float max, float *dest);
      virtual int CheckUserFloatSliderValue(coFloatSliderParam *f, float old,
         float min, float max, float *dest);
      virtual int  CheckUserIntValue(coIntScalarParam *f, int old, int min,
         int max, int *dest);

      coOutputPort *blade;
      coOutputPort *hub;
      coOutputPort *shroud;

      coFileBrowserParam      *startFile;

      int isInitialized;

      struct geometry *geo;

      // menue sections
#define  M_RUNNER_DATA           "runner data"
#define  M_BLADE_PROFILE_DATA    "blade profile"
#define  M_LEADING_EDGE_DATA        "leading edge"
#define  M_TRAILING_EDGE_DATA    "trailing edge"
#define  M_BLADE_DATA            "blade elements"
#define  M_BLADE_ELEMENT_DATA    "blade element data"
#define M_REDUCED_MODIFY         "reduced modification"

      // global runner data
#define  M_NUMBER_OF_BLADES         "number of blades"
#define  M_OUTLET_DIAMETER_ABS      "outlet diameter, absolute"
#define  M_INLET_DIAMETER_REL    "inlet diameter, relative"
#define  M_SHROUD_HEIGHT_DIFF    "shroud height difference"
#define  M_INLET_CONDUIT_WIDTH      "inlet conduit width"
#define  M_OUTLET_CONDUIT_WIDTH     "outlet conduit width"
#ifdef GAP
#define  M_GAP_WIDTH       "gap width"
#endif
#define  M_INLET_CONTOUR_ANGLE      "inlet contour angle"
#define  M_OUTLET_CONTOUR_ANGLE     "outlet contour angle"
#define  M_INLET_OPEN_ANGLE_HUB     "inlet opening angle (hub)"
#define M_INLET_OPEN_ANGLE_SHROUD   "inlet opening angle (shroud)"
#define  M_OUTLET_OPEN_ANGLE_HUB    "outlet opening angle (hub)"
#define M_OUTLET_OPEN_ANGLE_SHROUD  "outlet opening angle (shroud)"

#ifdef CREATE_PROFILE_MENU
      // runner profile data
#define  M_NUMBER_OF_PROFILE_SEC    "number of profile sections"
#define  M_REL_CHORD             "relative chord"
#define  M_REL_THICKNESS            "relative thickness"
#endif                                         // CREATE_PROFILE_MENU

      // blade edge data
#define  M_LE_HUB_PARM           "leading edge hub parameter"
#define M_LE_HUB_ANGLE           "leading edge hub off-contour angle"
#define  M_LE_SHROUD_PARM        "leading edge shroud parameter"
#define M_LE_SHROUD_ANGLE        "leading edge shroud off-contour angle"
#define  M_TE_HUB_PARM           "trailing edge hub parameter"
#define M_TE_HUB_ANGLE           "trailing edge hub off-contour angle"
#define  M_TE_SHROUD_PARM        "trailing edge shroud parameter"
#define M_TE_SHROUD_ANGLE        "trailing edge shroud off-contour angle"

      // blade element specifications
#define  M_NUMBER_OF_BLADE_ELEMENTS "number of blade elements"
#define M_BLADE_BIAS_FACTOR         "blade element bias factor"
#define M_BLADE_BIAS_TYPE        "blade element bias type"

      // blade element data
#define  M_MERIDIAN_PARAMETER    "meridian parameter"
#define  M_INLET_ANGLE           "inlet angle"
#define  M_OUTLET_ANGLE          "outlet angle"
#define  M_PROFILE_THICKNESS        "profile thickness"
#define  M_TE_THICKNESS          "trailing edge thickness"
#define  M_CENTRE_LINE_CAMBER    "centre line camber"
#define M_TE_WRAP_ANGLE          "trailing edge wrap angle"
#define M_BL_WRAP_ANGLE          "blade wrap angle"
#define M_PROFILE_SHIFT          "blade profile shift"
#define  M_INLET_ANGLE_MODIFICATION "inlet angle modification"
#define  M_OUTLET_ANGLE_MODIFICATION   "outlet angle modification"
#define  M_CENTRE_LINE_CAMBER_POSN  "centre line camber position"
#define M_REMAINING_SWIRL        "remaining swirl"

      coIntScalarParam  *p_NumberOfBlades;
      coFloatParam   *p_OutletDiameterAbs;
      coFloatParam   *p_InletDiameterRel;
      coFloatParam   *p_ShroudHeightDiff;
      coFloatParam   *p_InletConduitWidth;
      coFloatParam   *p_OutletConduitWidth;
#ifdef GAP
      coFloatParam   *p_GapWidth;
#endif
      coFloatParam   *p_InletContourAngle;
      coFloatParam   *p_OutletContourAngle;
      coFloatParam   *p_InletOpenAngleHub;
      coFloatParam   *p_InletOpenAngleShroud;
      coFloatParam   *p_OutletOpenAngleHub;
      coFloatParam   *p_OutletOpenAngleShroud;

#ifdef CREATE_PROFILE_MENUE
      coIntScalarParam  *p_NumberOfProfileSecs;
      coFloatParam   *p_RelativeChord;
      coFloatParam   *p_RelativeThickness;
#endif                                         // CREATE_PROFILE_MENUE

      coFloatParam   *p_LeHubParm;
      coFloatParam   *p_LeHubAngle;
      coFloatParam   *p_LeShroudParm;
      coFloatParam   *p_LeShroudAngle;
      coFloatParam   *p_TeHubParm;
      coFloatParam   *p_TeHubAngle;
      coFloatParam   *p_TeShroudParm;
      coFloatParam   *p_TeShroudAngle;

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

      // reduced modify menu
#define  MAX_MODIFY  50
      char **ReducedModifyMenuPoints;
      int numReducedMenuPoints;

#define  M_LEFT_POINT   "hub point"
#define  M_MIDDLE_POINT "inner point"
#define  M_RIGHT_POINT  "shroud point"
      coFloatSliderParam  *p_HubPoint[MAX_MODIFY];
      coFloatSliderParam  *p_InnerPoint[MAX_MODIFY];
      coFloatSliderParam  *p_ShroudPoint[MAX_MODIFY];

   public:

      RadialRunner(int argc, char *argv[]);
};
#endif
