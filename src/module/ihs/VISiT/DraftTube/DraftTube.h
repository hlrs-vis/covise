#ifndef _GG_SET_H
#define _GG_SET_H

#include <api/coModule.h>
using namespace covise;
#include <General/include/geo.h>
#include <DraftTube/include/tube.h>

#define  P_GEOMETRY     "Geometry"
#define  P_GEOMETRYCS   "Cross_sections"
#define  P_GEOMETRYCSA  "Cross_sections_areas"
#define  P_GRID         "Grid"

static const char *direction[] = { "NE", "NW", "SW", "SE"};

class DraftTube : public coModule
{
   COMODULE

   private:

      enum { MAX_CROSS=50, NUM_INTERPOL=1 };

      // DraftTube.cpp
      virtual int   compute(const char *port);
      virtual void  param(const char *, bool inMapLoading);
      virtual void  postInst();
      virtual void  quit();
      virtual void  CreateMenu_GeometryCrossSection();
      virtual void  CreateMenu_AreaCrossSection();
      virtual void  CreateMenu_GridCrossSection();
      virtual void  CalcValuesForCSArea();
      virtual float CalcOneArea(int);

      // InterpolateAreas.cpp
      virtual void  InterpolateAreas();
      virtual float LinearInterpolation(float, float, float, float);

      // CheckUserInput.cpp
      virtual void  CheckUserInput(const char *, struct geometry *);
      virtual int   SplitPortname(const char *, char *, int *);
      virtual char *IndexedParameterName(const char *, int );

      coOutputPort *surf;
      coOutputPort *cross;
      coOutputPort *grid;
      coOutputPort *boco;
      coOutputPort *bc_in;

#define  GEO_SEC  "GeometrySection"
#define  PARAM_FILE  "ParameterFile"
      coFileBrowserParam      *paramFile;
      // geometry
      coBooleanParam      *p_absync;
      coFloatVectorParam  *p_m[MAX_CROSS];
      coFloatVectorParam  *p_hw[MAX_CROSS];
      coFloatVectorParam  *p_ab[MAX_CROSS][4];
      coBooleanParam      *p_angletype[MAX_CROSS];
      coFloatParam  *p_angle[MAX_CROSS];
      coFloatSliderParam  *p_cs_area[MAX_CROSS];
#define  P_ABSYNC    "ab_sync_mode"
#define  P_M            "MiddlePoint"
#define  P_HW        "Height_Width"
#define  P_AB        "a_b"
#define  P_ANGLETYPE    "AngleType"
#define  P_ANGLE        "Angle"
#define  P_CS_AREA      "Area"

      // Interpolation posibilities
      coChoiceParam     *p_ip_type;
      coIntScalarParam  *p_ip_S;
      coIntScalarParam  *p_ip_E;
      coBooleanParam      *p_ip_start;
      coBooleanParam      *p_ip_ab;
      coBooleanParam      *p_ip_height;
      coBooleanParam      *p_ip_width;
#define  P_IP_TYPE      "IP_Type"
#define  P_IP_S         "from_CS"
#define  P_IP_E         "until_CS"
#define  P_IP_START     "Start_of_IP"
#define  P_IP_AB        "IP_of_ab"
#define  P_IP_HEIGHT    "IP_of_height"
#define  P_IP_WIDTH     "IP_of_width"

      // grid
      coIntScalarParam    *p_numi[4];
      coIntScalarParam    *p_numo;
      coIntScalarParam    *p_elem[MAX_CROSS];
      coFloatParam  *p_part[MAX_CROSS][8];

      // Button for triggering Grid Generation
      coBooleanParam      *p_makeGrid;

      // some Menues ...
      coChoiceParam       *m_GeometryCrossSection;
      char               **cs_labels;

      struct geometry *geo;

   public:

      DraftTube(int argc, char *argv[]);

};
#endif
