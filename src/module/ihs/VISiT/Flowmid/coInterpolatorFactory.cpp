#include "coInterpolatorFactory.h"
#include <map>

coInterpolatorFactory::coInterpolatorFactory()
{
   cout << "coInterpolatorFactory Constructor should never be called!" << endl;
}


map<string,int> coInterpolatorFactory::interpolatorMap;

coInterpolator *coInterpolatorFactory::getInterpolator(coDistributedObject *cellobj, coDistributedObject *data,string type)
{

   coInterpolator *interpolator = NULL;

   switch(interpolatorMap[type])                  //TODO: Warning einbauen, falls getInterpolatorMap noch nicht aufgerufen
   {

      case Default:
      case BandSearchSimpel:
         cout << "Creating BandSearchSimpelInterpolator Object." << endl;
         interpolator = new coBandSearchSimpelInterpolator(cellobj,data);
         break;

      case DeBoorSpline:
      case NearestNeighbour:
      case TrilinearIsoparametric:
         cout << "Not implemented yet, " << endl;

      default:
         cout << "Invalid Interpolation Method!" << endl;
         //interpolator = NULL;
         //throw some Exception
   }
   return interpolator;
}


map<string,int>& coInterpolatorFactory::getInterpolatorMap()
{

   interpolatorMap["Default"] = Default;          //TODO: muss ueberdacht werden, vielleicht doch Constructor?
   interpolatorMap["BandSearchSimpel"] = BandSearchSimpel;
   interpolatorMap["DeBoorSpline"] = DeBoorSpline;
   interpolatorMap["NearestNeighbour"] = NearestNeighbour;
   interpolatorMap["TrilinearIsoparametric"] = TrilinearIsoparametric;

   return interpolatorMap;
}
