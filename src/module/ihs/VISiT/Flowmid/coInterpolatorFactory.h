#include "api/coModule.h"
#include "coBandSearchSimpelInterpolator.h"
#include <string>
#include <map>

class coInterpolatorFactory
{
   private:
      coInterpolatorFactory();
      enum interpolationMethodEnumeration {Default,BandSearchSimpel,DeBoorSpline,NearestNeighbour,TrilinearIsoparametric}; // Make this member const ...
      static map<string,int> interpolatorMap; // Make this member const as well ...
   public:
      static coInterpolator *getInterpolator(coDistributedObject *, coDistributedObject *,string); // Make the arguments const ...
      static map<string,int>& getInterpolatorMap(); // Make this memeber const ...
};
