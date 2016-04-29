#include "coVR1DTransInteractor.h"
#include <Performer/pfdu.h>

#define ACTION_BUTTON 1
#define DRIVE_BUTTON 56
#define XFORM_BUTTON 112

coVR1DTransInteractor::coVR1DTransInteractor(pfVec3 o, pfVec3 d, float min, float max, float val, float size, int showConstr, char *name)
{
   pfMatrix wm, tm, sm;
   pfVec3 zaxis(0,0,1);

   origin = o;
   dir = d;
   dir.normalize();
   minVal = min;
   maxVal = max;
   currVal = val;
   interSize = size;
   interactorName= new char[strlen(name)+1];
   strcpy(interactorName, name);
   if ((showConstr == SHOW_CONSTRAINTS_ALWAYS)
      || (showConstr == SHOW_CONSTRAINTS_ONTOUCH)
      || (showConstr == SHOW_CONSTRAINTS_ONSELECT))
      showConstraints = showConstr;

   else
   {
      fprintf(stderr,"\nERROR in coVR1DTransInteractor::coVR1DTransInteractor\n");
      fprintf(stderr,"\tillegal value [%d] for showConstraints\n", showConstr);

      showConstraints = SHOW_CONSTRAINTS_ALWAYS;
   }
   float interScale = size/cover->getScale();
   fprintf(stderr,"interactor size=%f cover scale = %f\n", interSize, cover->getScale());
   // initialize flags

   interactionOngoing = false;
   isI = false;
   isS = false;
   isE = false;
   justHit = true;

   // get debug level
   const char *line = CoviseConfig::getEntry("COVERConfig.DEBUG_LEVEL");
   if (line)
      sscanf(line,"%d", &debugLevel);
   else
      debugLevel = 0;

   // debugprint
   if (debugLevel>0)
   {
      fprintf(stderr,"\ncoVR1DTransInteractor:coVR1DTransInteractor [%s]\n", interactorName);
      fprintf(stderr,"\tshowConstraints=[%d]\n", showConstraints);
      fprintf(stderr,"\torigin=[%f %f %f]\n", origin[0], origin[1], origin[2]);
      fprintf(stderr,"\tdir=[%f %f %f]\n", dir[0], dir[1], dir[2]);
      fprintf(stderr,"\tminVal=[%f] maxVal=[%f] currVal=[%f]\n", minVal, maxVal, currVal);

   }

   // check and adjust the ranges
   checkRange();

   // load geostates only once
   geostate = loadDefaultGeostate();
   unlightedGeostate = loadUnlightedGeostate();

   //       objectsRoot
   //            |
   //        worldDCS
   //            |
   //       interactorRoot
   //          |   |
   // lineGeode   sphereTransDCS
   //                  |
   //             sphereScaleDCS
   //                  |
   //             sphereGeode

   if(debugLevel>1)
      fprintf(stderr,"\tcreating the scenegraph\n");

   objectsRoot = cover->getObjectsRoot();

   worldDCS = new pfDCS();
   wm.makeVecRotVec(zaxis, dir);
   wm[3][0] = origin[0];
   wm[3][1] = origin[1];
   wm[3][2] = origin[2];
   worldDCS->setMat(wm);

   interactorRoot = new pfGroup();

   sphereTransDCS = new pfDCS();
   tm.makeTrans(0, 0, currVal);
   sphereTransDCS->setMat(tm);

   sphereScaleDCS = new pfDCS();
   sm.makeScale(interScale, interScale, interScale);
   sphereScaleDCS->setMat(sm);

   sphereGeode = createSphere();
   lineGeode = createLine();

   objectsRoot->addChild(worldDCS);
   worldDCS->addChild(interactorRoot);
   interactorRoot->addChild(lineGeode);
   interactorRoot->addChild(sphereTransDCS);
   sphereTransDCS->addChild(sphereScaleDCS);
   sphereScaleDCS->addChild(sphereGeode);

   if (showConstraints == SHOW_CONSTRAINTS_ALWAYS)
      showLine();
   else
      hideLine();

   if(debugLevel>1)
      fprintf(stderr,"\tcreating the highlights\n");

   // highlights
   isectedHl = new pfHighlight();
   isectedHl->setMode(PFHL_FILL);
   isectedHl->setColor(PFHL_FGCOLOR, 0.6, 0.6, 0.0);
   isectedHl->setLineWidth(3.0);

   selectedHl = new pfHighlight();
   selectedHl->setMode(PFHL_FILL);
   selectedHl->setColor(PFHL_FGCOLOR, 0.0, 0.6, 0.0);
   selectedHl->setLineWidth(3.0);

}


coVR1DTransInteractor::~coVR1DTransInteractor()
{
   if (debugLevel>0)
      fprintf(stderr,"\ncoVR1DTransInteractor::loadDefaultGeostate\n");

   pfGroup *parent;

   // delete scene graph
   if (sphereGeode)
   {
      parent = sphereGeode->getParent(0);
      if (parent)
         parent->removeChild(sphereGeode);

      pfDelete(sphereGeode);
   }

   if (sphereScaleDCS)
   {
      parent = sphereScaleDCS->getParent(0);
      if (parent)
         parent->removeChild(sphereScaleDCS);

      pfDelete(sphereScaleDCS);
   }

   if (sphereTransDCS)
   {
      parent = sphereTransDCS->getParent(0);
      if (parent)
         parent->removeChild(sphereTransDCS);

      pfDelete(sphereTransDCS);
   }

   if (lineGeode)
   {
      parent = lineGeode->getParent(0);
      if (parent)
         parent->removeChild(lineGeode);

      pfDelete(lineGeode);
   }
   if (interactorRoot)
   {
      parent = interactorRoot->getParent(0);
      if (parent)
         parent->removeChild(interactorRoot);

      pfDelete(interactorRoot);
   }

   if (worldDCS)
   {
      parent = worldDCS->getParent(0);
      if (parent)
         parent->removeChild(worldDCS);

      pfDelete(worldDCS);
   }

   // delete the highights
   pfDelete(selectedHl);
   pfDelete(isectedHl);
}


int coVR1DTransInteractor::checkRange()
{
   if (debugLevel>1)
      fprintf(stderr,"\ncoVR1DTransInteractor:checkRange\n");

   int ok = true;

   if (currVal < minVal)
   {
      fprintf(stderr,"\nWARNING coVR1DTransInteractor::checkRange\n");
      fprintf(stderr,"\tminVal=[%f] > currVal=[%f]\n", minVal, currVal);
      fprintf(stderr,"\tadjusting minVal to currVal=[%f]", currVal);
      minVal = currVal;
      ok = false;
   }

   if (currVal > maxVal)
   {
      fprintf(stderr,"\nWARNING coVR1DTransInteractor::checkRange\n");
      fprintf(stderr,"\tmaxVal=[%f] > currVal=[%f]\n", maxVal, currVal);
      fprintf(stderr,"\tadjusting maxVal to currVal=[%f]", currVal);
      maxVal = currVal;
      ok = false;
   }

   return(ok);
}


pfGeoState*
coVR1DTransInteractor::loadDefaultGeostate()
{
   if (debugLevel>1)
      fprintf(stderr,"\ncoVR1DTransInteractor::loadDefaultGeostate\n");

   pfGeoState *geoState;
   pfMaterial *mtl;

   mtl = new pfMaterial;
   mtl->setSide(PFMTL_BOTH);
   mtl->setColorMode(PFMTL_BOTH, PFMTL_CMODE_AMBIENT_AND_DIFFUSE);
   mtl->setColor( PFMTL_AMBIENT, 0.2f, 0.2f, 0.2f);
   mtl->setColor( PFMTL_DIFFUSE, 0.9f, 0.9f, 0.9f);
   mtl->setColor( PFMTL_SPECULAR, 0.9f, 0.9f, 0.9f);
   mtl->setColor( PFMTL_EMISSION, 0.0f, 0.0f, 0.0f);
   mtl->setShininess(16.0f);

   geoState = new pfGeoState();
   geoState->makeBasic();
   geoState->setAttr(PFSTATE_FRONTMTL, mtl);
   geoState->setAttr(PFSTATE_BACKMTL, mtl);
   geoState->setMode(PFSTATE_ENLIGHTING, PF_ON);
   geoState->setMode(PFSTATE_TRANSPARENCY, PFTR_ON);

   return(geoState);

}


pfGeoState*
coVR1DTransInteractor::loadUnlightedGeostate()
{

   pfGeoState *geoState;
   pfMaterial *mtl;

   mtl = new pfMaterial;
   mtl->setSide(PFMTL_BOTH);
   mtl->setColorMode(PFMTL_BOTH, PFMTL_CMODE_AMBIENT_AND_DIFFUSE);
   mtl->setColor( PFMTL_AMBIENT, 0.2f, 0.2f, 0.2f);
   mtl->setColor( PFMTL_DIFFUSE, 0.9f, 0.9f, 0.9f);
   mtl->setColor( PFMTL_SPECULAR, 0.9f, 0.9f, 0.9f);
   mtl->setColor( PFMTL_EMISSION, 0.0f, 0.0f, 0.0f);
   mtl->setShininess(16.0f);

   geoState = new pfGeoState();
   geoState->makeBasic();
   geoState->setAttr(PFSTATE_FRONTMTL, mtl);
   geoState->setAttr(PFSTATE_BACKMTL, mtl);
   geoState->setMode(PFSTATE_ENLIGHTING, PF_OFF);
   geoState->setMode(PFSTATE_TRANSPARENCY, PFTR_ON);

   return(geoState);

}


pfGeode *
coVR1DTransInteractor::createLine()
{
   pfVec3 *lc;
   pfVec4 *color;

   lc = (pfVec3 *) pfCalloc(2, sizeof(pfVec3), pfGetSharedArena());
   lc[0].set(0, 0, minVal);
   lc[1].set(0,0, maxVal);

   if (debugLevel>1)
   {
      fprintf(stderr,"\tp0=%f %f %f\n", origin[0]+dir[0]*minVal, origin[1]+dir[1]*minVal, origin[2]+dir[2]*minVal);
      fprintf(stderr,"\tp1=%f %f %f\n", origin[0]+dir[0]*minVal, origin[1]+dir[1]*minVal, origin[2]+dir[2]*minVal);
   }

   color  = (pfVec4 *) pfCalloc(1, sizeof(pfVec4), pfGetSharedArena());
   color[0].set(1, 1, 1, 1);

   lineGeoset = new pfGeoSet();
   lineGeoset->setAttr(PFGS_COORD3, PFGS_PER_VERTEX, lc, NULL);
   lineGeoset->setAttr(PFGS_COLOR4, PFGS_OVERALL, color, NULL);
   lineGeoset->setPrimType(PFGS_LINES);
   lineGeoset->setNumPrims(1);
   lineGeoset->setGState(unlightedGeostate);
   lineGeode = new pfGeode();
   lineGeode->addGSet(lineGeoset);

   return(lineGeode);

}


pfGeode *
coVR1DTransInteractor::createSphere()
{
   sphereGeoset = pfdNewSphere(50, pfGetSharedArena());

   pfVec4 *redcolor = (pfVec4*)pfCalloc(1, sizeof(pfVec4), pfGetSharedArena());
   redcolor->set(1.0, 0.0, 0.0, 1.0);
   sphereGeoset->setAttr(PFGS_COLOR4, PFGS_OVERALL, redcolor, NULL);

   sphereGeoset->setGState(geostate);

   sphereGeode = new pfGeode();
   sphereGeode->addGSet(sphereGeoset);

   return(sphereGeode);

}


void
coVR1DTransInteractor::show()
{
   if (debugLevel>0)
      fprintf(stderr,"\ncoVR1DTransInteractor::show [%s]\n", interactorName);

   worldDCS->setTravMask(PFTRAV_DRAW, 1, PFTRAV_SELF | PFTRAV_DESCEND, PF_SET);

}


void
coVR1DTransInteractor::hide()
{
   if (debugLevel>0)
      fprintf(stderr,"coVR1DTransInteractor::hide [%s]\n", interactorName);

   worldDCS->setTravMask(PFTRAV_DRAW, 0, PFTRAV_SELF | PFTRAV_DESCEND, PF_SET);

}


void
coVR1DTransInteractor::showLine()
{
   if (debugLevel>0)
      fprintf(stderr,"\ncoVR1DTransInteractor::showLine [%s]\n", interactorName);

   lineGeode->setTravMask(PFTRAV_DRAW, 1, PFTRAV_SELF | PFTRAV_DESCEND, PF_SET);

}


void
coVR1DTransInteractor::hideLine()
{
   if (debugLevel>0)
      fprintf(stderr,"coVR1DTransInteractor::hideLine [%s]\n", interactorName);

   lineGeode->setTravMask(PFTRAV_DRAW, 0, PFTRAV_SELF | PFTRAV_DESCEND, PF_SET);

}


void
coVR1DTransInteractor::enable()
{
   if (debugLevel>0)
      fprintf(stderr,"\ncoVR1DTransInteractor::enable [%s]\n", interactorName);
   isE = true;
   intersector.add(sphereTransDCS, this);
}


void
coVR1DTransInteractor::disable()
{
   if (debugLevel>0)
      fprintf(stderr,"\ncoVR1DTransInteractor::disable [%s]\n", interactorName);
   isE = false;
   intersector.remove(sphereTransDCS);
}


int
coVR1DTransInteractor::hit(pfVec3 &hitPoint,pfHit *hit)
{
   if (debugLevel>0)
   {
      if (justHit)
         fprintf(stderr,"\ncoVR1DTransInteractor::justhit [%s]\n", interactorName);
   }
   else if (debugLevel>1)
      fprintf(stderr,"\ncoVR1DTransInteractor::hit [%s]\n", interactorName);

   if (!cover->isPointerLocked() && !cover->isNavigating())
   {
      isI = true;
      if (showConstraints == SHOW_CONSTRAINTS_ONTOUCH)
         showLine();

      // get pointer state
      coPointerButton *pb = cover->getButton();

      if (pb->wasPressed() && (pb->getButtonStatus() /*==  ACTION_BUTTON*/) )
      {
         if (debugLevel>0)
            fprintf(stderr,"\tchanging state to selected\n");
         isS = true;
         sphereGeoset->setHlight(selectedHl);

      }
      else
      {
         if (!isS)
            sphereGeoset->setHlight(isectedHl);
      }
   }
   justHit = false;

   return ACTION_CALL_ON_MISS;
}


void
coVR1DTransInteractor::miss()
{
   justHit = true;
   if (debugLevel>0)
      fprintf(stderr,"\ncoVR1DTransInteractor::miss\n");

   if (!isS)
   {
      if (showConstraints == SHOW_CONSTRAINTS_ONTOUCH)
         hideLine();

      sphereGeoset->setHlight(PFHL_OFF);
   }
   isI = false;

}


void
coVR1DTransInteractor::startMove()
{
   pfMatrix initHandMat;

   if (debugLevel>0)
      fprintf(stderr,"\ncoVR1DTransInteractor::startMove\n");

   // grab the pointer this means, disable menu intersection and other interactions
   // remove pointer icon
   cover->lockPointer();

   // store hand mat
   cover->getPointer()->getMat(initHandMat);
   initHandPos[0] = initHandMat[3][0];
   initHandPos[1] = initHandMat[3][1];
   initHandPos[2] = initHandMat[3][2];
}


void
coVR1DTransInteractor::stopMove()
{
   if (debugLevel>0)
      fprintf(stderr,"\ncoVR1DTransInteractor::stopMove\n");

   isS = false;
   if (showConstraints == SHOW_CONSTRAINTS_ONTOUCH)
      hideLine();
   sphereGeoset->setHlight(PFHL_OFF);
   cover->unlockPointer();

}


void
coVR1DTransInteractor::move()
{
   pfVec3 relPos_o;
   if (debugLevel>1)
      fprintf(stderr,"\ncoVR1DTransInteractor::move\n");

   pfMatrix currHandMat, oldMat, newMat, relMat, o_to_w, w_to_o;
   pfVec3 currHandPos, currHandPos_o, initHandPos_o;

   cover->getPointer()->getMat(currHandMat);
   currHandPos[0] = currHandMat[3][0];
   currHandPos[1] = currHandMat[3][1];
   currHandPos[2] = currHandMat[3][2];

   //fprintf(stderr,"--- diff world= %f\n", currHandPos[0] - initHandPos[0]);

   // get hand mat in interactor coords
   o_to_w = getAllTransformations(interactorRoot);
   w_to_o.invertFull(o_to_w);
   currHandPos_o.fullXformPt(currHandPos, w_to_o);
   initHandPos_o.fullXformPt(initHandPos, w_to_o);

   relPos_o = currHandPos_o - initHandPos_o;

   relMat.makeTrans(relPos_o[0], relPos_o[1], relPos_o[2]);
   fprintf(stderr,"--- relPos_o[2]= %f\n", relPos_o[2]);

   // we ignore the x and y movement of the hand because
   // interactor movement is restricted to line

   sphereTransDCS->getMat(oldMat);

   // relPos_o[2] old oldPos[2] is currentVal
   float cv = oldMat[3][2]+relPos_o[2];

   // but we have to check min max
   if (cv > maxVal)
      cv = maxVal;
   if (cv < minVal)
      cv = minVal;

   // and now we apply it
   setValue(cv);

   initHandPos = currHandPos;

}


void
coVR1DTransInteractor::keepSize()
{
   if (debugLevel>1)
      fprintf(stderr,"\ncoVR1DTransInteractor::keepSize\n");

   pfMatrix mat;
   float interScale = interSize/cover->getScale();

   mat.makeScale(interScale, interScale, interScale);
   sphereScaleDCS->setMat(mat);
}


void
coVR1DTransInteractor::setValue(float v)
{
   pfMatrix tm;

   currVal = v;
   tm.makeTrans(0, 0, currVal);
   sphereTransDCS->setMat(tm);

}


pfMatrix
coVR1DTransInteractor::getAllTransformations(pfNode *node)
{
   if (debugLevel>1)
      fprintf(stderr,"\ncoVR1DTransInteractor::getAllTransformations\n");
   pfNode *parent;
   pfMatrix tr;
   pfMatrix dcsMat;
   tr.makeIdent();
   parent = node->getParent(0);
   while(parent!=NULL)
   {
      if (pfIsOfType(parent,pfDCS::getClassType()))
      {
         ((pfDCS *)parent)->getMat(dcsMat);
         //dcsMat.print(0, 1, "getAllTransformations: ", stderr);
         tr.postMult(dcsMat);
      }
      if(parent->getNumParents())
         parent = parent->getParent(0);
      else
         parent = NULL;
   }
   return tr;
}


void
coVR1DTransInteractor::preFrame()
{
   keepSize();
   if (!cover->isPointerLocked() && !cover->isNavigating() && !interactionOngoing && isSelected() && (cover->getButton()->getButtonStatus() /*==  ACTION_BUTTON*/) )
   {

      startMove();
      interactionOngoing = true;
   }
   if (interactionOngoing)
   {
      if (cover->getButton()->wasReleased() )
      {
         stopMove();
         interactionOngoing = false;
      }
      else
         move();
   }

}
