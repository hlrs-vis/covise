#include "api/coModule.h"
namespace FlowmidClasses                          //TODO: Erweitern auf alle Klassen aus *.cpp
{

   class FlowmidUtil
   {
      private:
         FlowmidUtil();
      public:
         static double mapIntervall(double,double,double,double,double);
         static coDoIntArr *dupIntArr(coDoIntArr *);
         static coDoFloat *dupFloatArr(coDoFloat *);
   };

}
