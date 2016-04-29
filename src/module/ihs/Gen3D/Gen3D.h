#if !defined(__Gen3D_H)
#define __Gen3D_H

// includes
#include <api/coModule.h>
using namespace covise;

class Gen3D : public coModule
{

   private:
      virtual int compute(const char *port);
      int Diagnose();

      int isset;
      int numTimesteps;

      coInputPort *p_inPoly;
      coInputPort *p_inVelo;
      coInputPort *p_inPress;

      coOutputPort *p_outGrid;
      coOutputPort *p_outVelo;
      coOutputPort *p_outPress;

      coFloatParam *p_thick;

      coDoUnstructuredGrid *gridInObj;
      coDoUnstructuredGrid *gridOutObj;
      coDoVec3 *veloInObj;
      coDoVec3 *veloOutObj;
      coDoFloat *pressInObj;
      coDoFloat *pressOutObj;

      bool polygons;
      coDoPolygons *polyIn;
      coDoPolygons *polyOut;

      // set
      coDoSet *setPolyIn;
      coDoSet *setVeloIn;
      coDoSet *setPressIn;
      coDoSet *setGeoOut;
      coDoSet *setVeloOut;
      coDoSet *setPressOut;

      coDoSet *setOut;

      int expandGrid(int no_points, float *xCoords, float *yCoords, float *zCoords, float *xCoordsIn, float *yCoordsIn, float *zCoordsIn);
      int Gen3D::duplicateValuesUV3D(float *uIn, float *vIn, float *wIn, float *uOut, float *vOut, float *wOut);
      int Gen3D::duplicateValuesUS3D(float *pIn, float *pOut);

   public:

      Gen3D();
};
#endif                                            // __Gen3D_H
