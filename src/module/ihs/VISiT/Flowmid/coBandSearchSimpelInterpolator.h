#include "api/coModule.h"
#ifndef _HHCOINTERPOLATOR_
#define _HHCOINTERPOLATOR_
#include "coInterpolator.h"
#endif
#include <map>
#include <set>

class coBandSearchSimpelInterpolator: public coInterpolator
{

   private:
      set<int> nodeset;
      string type;
      coDoPolygons *polygons;
      coDoPolygons *targetPolygons;
      coDoIntArr *targetNodes;
      coDoFloat *scalardata;
      coDoVec3 *fielddata;
      int iband;
      double source_rmin, source_rmax;
      double target_rmin, target_rmax;
      double ratio;
      float *xcoords, *ycoords, *zcoords;
      vector<double> bandIntervalle;

      struct git_grp
      {
         double r, rmin, rmax;
         double scalar;
         double field[3];
         int nkn;
         vector<int> nodelist;
      };

      map<int,git_grp *> baender;

      void zerlegeBandIntervalle();
      void gruppenMitteln();
      void gruppenCheck();
      double linearInterpolation(double, double, double, double, double);

   public:
      coBandSearchSimpelInterpolator(coDistributedObject *, coDistributedObject *);
      int getScalarValue(double, double, double, double *);
      int getFieldValue(double, double, double, double *);
      void setTargetArea(coDoPolygons *, coDoIntArr *);
      void writeInfo(char *);
      string getType();
};
