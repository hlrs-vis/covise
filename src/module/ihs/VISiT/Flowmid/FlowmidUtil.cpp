#include "FlowmidUtil.h"

double FlowmidClasses::FlowmidUtil::mapIntervall(double source_min, double source_max, double target_min, double target_max, double r_target)
{

   double delta_source(0);
   double delta_target(0);
   double r_target_rel(0);
   double r_source(0);

   delta_source = source_max - source_min;
   delta_target = target_max - target_min;
   r_target_rel = r_target - target_min;

   r_source = source_min + (r_target_rel/delta_target)*delta_source;

   return r_source;
}


coDoIntArr *FlowmidClasses::FlowmidUtil::dupIntArr(coDoIntArr *intarr)
{

   int numdim;
   char *name;
   int *adr;

   numdim = intarr->getNumDimensions();
   name = intarr->get_name();
   adr = intarr->getAddress();

   int *dimarray = new int[numdim];

   for(int i(0); i<numdim; i++)
      dimarray[i] = intarr->get_dim(i);

   coDoIntArr *ret = new coDoIntArr(name,numdim,dimarray,adr);

   return ret;
}


coDoFloat *FlowmidClasses::FlowmidUtil::dupFloatArr(coDoFloat *floatarr)
{

   int numvalues;
   char *name;
   float *adr;

   numvalues = floatarr->getNumPoints();
   name = floatarr->get_name();
   floatarr->getAddress(&adr);

   coDoFloat *ret = new coDoFloat(name,numvalues,adr);

   return ret;
}
