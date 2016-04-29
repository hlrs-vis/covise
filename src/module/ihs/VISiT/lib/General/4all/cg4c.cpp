#include <stdio.h>
#include "../lib/General/include/geo.h"

#ifdef   DRAFT_TUBE
#include "../lib/General/include/cov.h"
#include "../lib/DraftTube/include/tube.h"
#endif

#ifdef   AXIAL_RUNNER
#include "../lib/AxialRunner/include/axial.h"
#include "../lib/AxialRunner/include/ar2cov.h"
#endif

#ifdef   RADIAL_RUNNER
#include "../lib/RadialRunner/include/radial.h"
#include "../lib/RadialRunner/include/rr2cov.h"
#endif

#ifdef   GATE
#include "../lib/Gate/include/gate.h"
#include "../lib/Gate/include/ga2cov.h"
#endif

struct covise_info *CreateGeometry4Covise(struct geometry *g)
{
   struct covise_info *ci = NULL;

   switch (g->type)
   {
      case GT_TUBE:
#ifdef   DRAFT_TUBE
         ci = Tube2Covise(g->tu);
#endif
         break;
      case GT_RRUNNER:
#ifdef   RADIAL_RUNNER
         ci = Radial2Covise(g->rr);
#endif
         break;
      case GT_ARUNNER:
#ifdef   AXIAL_RUNNER
         ci = Axial2Covise(g->ar);
#endif
         break;
      case GT_GATE:
#ifdef   GATE
         ci = Gate2Covise(g->ga);
#endif
         break;
      default:
         break;
   }
   return ci;
}
