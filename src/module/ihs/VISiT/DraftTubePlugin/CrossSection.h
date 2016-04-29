#ifndef _CROSSSECTION_H
#define _CROSSSECTION_H

// cossection class
// D. Rainer
// 09-nov-2001
// (C) 2001 RUS

#include <cover/coVRModuleSupport.h>
#include "coVR1DTransInteractor.h"
#include <Performer/pr/pfHighlight.h>

class CrossSection:public coAction
{
   private:

      int index;                                  // number of cross sections (identifier)
      coDistributedObject *polyObj;                 // the SETELE
      coInteractor *feedback;
      struct tube *dt;
      pfVec3 middlePoint;
      pfVec3 widthPoint;
      float sliderZero;                           // 0.5*width at starttime
      float sliderCurrent;                        // 0.5 * current width of COVISE module (not interactor)
      pfVec3 dir;                                 // direction for moving width interactor
      float radius;
      void updatePoints(coInteractor *inter, float n[3]);
      coVR1DTransInteractor *widthInteractor;
      pfGeode *outlineGeode;
      pfGeoSet *outlineGeoset;
      pfHighlight *isectedHl, *selectedHl;
      float interactorSize;

      // height !!!!
      pfVec3 heightPoint;
      pfVec3 heightDir;                           // direction for moving width interactor
      char * heightIName;                         //
      float  heightS0;                            // 0.5*width at starttime
      float  heightSC;                            // 0.5 * current width of COVISE module (not interactor)
      coVR1DTransInteractor *heightInteractor;

#ifdef   IA_AREA
      // area
      pfVec3 areaPoint;
      pfVec3 areaDir;                             // direction for moving width interactor
      char * areaIName;                           //
      float  areaS0;                              // 0.5*width at starttime
      float  areaSC;                              // 0.5 * current width of COVISE module (not interactor)
      coVR1DTransInteractor *areaInteractor;
#endif

#ifdef   IA_AB
      // a
      pfVec3 aPoint;
      pfVec3 aDirPoint;                           // direction for moving width interactor
      pfVec3 aDir;                                // direction for moving width interactor
      char * aIName;                              //
      float  aS0;                                 // 0.5*width at starttime
      float  aSC;                                 // 0.5 * current width of COVISE module (not interactor)
      coVR1DTransInteractor *aInteractor;

      // b
      pfVec3 bPoint;
      pfVec3 bDirPoint;                           // direction for moving width interactor
      pfVec3 bDir;                                // direction for moving width interactor
      char * bIName;                              //
      float  bS0;                                 // 0.5*width at starttime
      float  bSC;                                 // 0.5 * current width of COVISE module (not interactor)
      coVR1DTransInteractor *bInteractor;
#endif

      void createOutline();
      void updateOutline();
      void deleteOutline();
      void showOutline();
      void hideOutline();
      void highlightOutline();

      pfVec3 *lc;
      int *len;
      pfGeode *csGeode;
      bool isected;
      bool grabbed;
      bool interactionOngoing;
      bool enabled;
   public:

      CrossSection(coInteractor *inter, tube *dt, int index, float n[3]);
      ~CrossSection();

      void buildInteractors(struct tube *dt, int index);

      void preFrame();

      void setGeode(pfGeode *geode);
      // set the COVISE object (SETELE)
      void update(coInteractor *inter);

      coDistributedObject *getPolyObject(){return polyObj;};

      virtual int hit(pfVec3 &hitPoint,pfHit *hit);
      virtual void miss();

      void enable();
      void disable();

};
#endif
