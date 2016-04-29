#ifndef _DRAFTTUBE_H
#define _DRAFTTUBE_H

// drafttube class
// D. Rainer
// 09-nov-2001
// (C) 2001 RUS

#include <cover/coVRModuleSupport.h>

#include "../lib/DraftTube/include/tube.h"
#include "CrossSection.h"
class DraftTube
{
   private:

      struct tube *dt;                            // ihs drafttube struct
      int numCS;                                  // number of cross sections
      pfDCS *groupDCS;                            // the dcs above the group node
      coDistributedObject *setObj;                  // the SETELE
      CrossSection **csList;
      coInteractor *feedback;
      int createGridState;                        // state of the toggle button
      static void   recomputeCallback(void* , buttonSpecCell* );
      static void   gridCallback(void* , buttonSpecCell* );

   public:
      static DraftTube *vrdt;

      DraftTube(coInteractor *inter);
      ~DraftTube();

      void preFrame();

      // set the peformer dcs of the group node
      void setNode(pfDCS *dcs){groupDCS=dcs;};

      // set the COVISE object (SETELE)
      void update(coInteractor *inter);

      coDistributedObject *getSetObject(){return setObj;};

      void setCrossSection(coInteractor *inter, int index);

      void deleteCrossSection(int index);

      int getNumCrossSection(){return numCS;};

      void setCrossSectionGeode(int index, pfGeode *geode);
      void lockCrossSection(int index);
      void unlockAllCrossSection();

      void extractInfo(coInteractor *inter);      // gets the numCS and the middle point coordinates
      float MP[100][3];                           // grumpf, auch denken wir nach dem Review noch mal gut nach ...
};
#endif
