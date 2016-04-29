#ifndef _CO_VR_1D_TRANS_INTERACTOR_H
#define _CO_VR_1D_TRANS_INTERACTOR_H

#include <Performer/pf/pfGroup.h>
#include <Performer/pf/pfNode.h>
#include <Performer/pf/pfDCS.h>
#include <Performer/pf/pfText.h>
#include <Performer/pf/pfGeode.h>
#include <Performer/pf/pfChannel.h>
#include <Performer/pf/pfTraverser.h>
#include <Performer/pr/pfGeoSet.h>
#include <Performer/pr/pfString.h>
#include <Performer/pr/pfMaterial.h>
#include <Performer/pr/pfLinMath.h>
#include <Performer/pr/pfHighlight.h>

#include <cover/coVRModuleSupport.h>
#include <cover/coIntersection.h>

//       objectsRoot
//            |
//        worldDCS
//            |
//          interactorRoot
//          |   |
// lineGeode   sphereTransDCS
//                  |
//             sphereScaleDCS
//                  |
//             sphereGeode

class coVR1DTransInteractor: public coAction
{

   private:
      pfDCS *sphereTransDCS, *sphereScaleDCS, *worldDCS;
      pfGeode *lineGeode, *sphereGeode;
      pfGroup *objectsRoot, *interactorRoot;
      pfGeoSet *sphereGeoset, *lineGeoset;
      pfGeoState *geostate, *unlightedGeostate;

      bool isI;
      bool isS;
      bool isE;
      bool interactionOngoing;
      bool justHit;
      int showConstraints;
      char *interactorName;

      pfHighlight *isectedHl, *selectedHl;
      pfVec3 origin, dir;
      pfVec3 initHandPos;                         // hand pos at button press
      float minVal, maxVal, currVal;
      float interSize;                            //[absolute size in mm]
      int debugLevel;

      int checkRange();
      pfGeoState* loadDefaultGeostate();
      pfGeoState* loadUnlightedGeostate();
      pfMatrix getAllTransformations(pfNode *node);
      pfGeode* createLine();
      pfGeode* createSphere();

      void showLine();
      void hideLine();

   public:
      enum {SHOW_CONSTRAINTS_ALWAYS=0, SHOW_CONSTRAINTS_ONTOUCH, SHOW_CONSTRAINTS_ONSELECT};

      // build scene graph
      coVR1DTransInteractor(pfVec3 o, pfVec3 d, float min, float max, float val,
         float size, int showConstraints, char *name);

      // delete scene graph
      ~coVR1DTransInteractor();

      // called every time when the geometry is intersected
      virtual int hit(pfVec3 &hitPoint,pfHit *hit);

      // called once when the geometry is not intersected any more
      virtual void miss();

      // make the interactor intersection sensitive
      void enable();

      // make the interactor intersection insensitive
      void disable();

      // make the interactor visible
      void show();

      // make the interactor invisible
      void hide();

      // return the dcs NEEDED?
      //pfDCS *getDCS(){return transDCS;};

      // return the geode NEEDED?
      //pfGeode *getGeode(){return geode;};

      // return the intersected state
      int isIntersected(){return isI;};

      // retunr the selected state,
      // the interactor can be selected, but not intersected any more
      int isSelected(){return isS;};

      // set a new position, keep old restrictions
      void setValue(float v);

      // set position and restriction
      //void setValue(float min, float mx, float val);

      // retunr the current position
      float getValue(){return currVal;};

      // scale sphere to keep the size when the world scaling changes
      void keepSize();

      // move the interactor relatively to it's old position
      // according to the hand movements
      void move();

      // start the interaction (grab pointer, set selected hl, store dcsmat)
      void startMove();

      // stop the interaction
      void stopMove();

      // called in preframe, does the interaction
      void preFrame();
};
#endif
