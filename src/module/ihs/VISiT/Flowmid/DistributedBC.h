#include "api/coModule.h"
#ifndef _HHCOINTERPOLATOR_
#define _HHCOINTERPOLATOR_
#include "coInterpolator.h"
#endif
#include "FlowmidUtil.h"
#ifndef RNUM
#define RNUM 16
#endif

class DistributedBC
{
   private:
      coDistributedObject *const *bocoArr;
      coDistributedObject **pboco;
      int igeb;
   public:
      DistributedBC(coDoSet *);
      int setBoundaryCondition(coInterpolator *, coDoPolygons *);
      coDistributedObject **getPortObj();
};
