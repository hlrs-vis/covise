#include "CrossSection.h"
#include "DraftTube.h"
#include "../lib/DraftTube/include/tube.h"
#include "../lib/DraftTube/include/tube_vr.h"
#include "../lib/DraftTube/include/t2c.h"
#include "../lib/General/include/points.h"

#define ACTION_BUTTON 1
CrossSection::CrossSection(coInteractor *inter, tube *t, int ind, float n[3])
{

   fprintf(stderr,"CrossSection::CrossSection\n");
   outlineGeode = NULL;
   csGeode=NULL;
   isected = false;
   grabbed = false;
   interactionOngoing = false;
   enabled=false;

   // we store the feedback once
   feedback=inter;
   feedback->incRefCount();
   polyObj=(coDistributedObject*) inter->getObject();
   dt=t;
   index=ind;
   if (n)
      fprintf(stderr, "\tn[0]=%f, n[1]=%f, n[2]=%f\n", n[0], n[1], n[2]);
   else
      fprintf(stderr, "\tn[0]=NULL, n[1]=NULL, n[2]=NULL\n");
   updatePoints(inter, n);

   buildInteractors(t, ind);

   // make wireframe cross section
   // get
   createOutline();
   //hideOutline();

   // highlights
   isectedHl = new pfHighlight();
   isectedHl->setMode(PFHL_FILL);
   isectedHl->setColor(PFHL_FGCOLOR, 1.0, 1.0, 0.0);
   isectedHl->setLineWidth(3.0);

   selectedHl = new pfHighlight();
   selectedHl->setMode(PFHL_FILL);
   selectedHl->setColor(PFHL_FGCOLOR, 0.0, 1.0, 0.0);
   selectedHl->setLineWidth(3.0);
}


CrossSection::~CrossSection()
{
   fprintf(stderr,"CrossSection::~CrossSection\n");
   delete widthInteractor;
   delete heightInteractor;
#ifdef   IA_AREA
   delete areaInteractor;
#endif
#ifdef   IA_AB
   delete aInteractor;
   delete bInteractor;
#endif
   deleteOutline();

   // delete the highights
   pfDelete(selectedHl);
   pfDelete(isectedHl);
}


void
CrossSection::buildInteractors(struct tube *dt, int index)
{
   char buf[255];

   const char *line = CoviseConfig::getEntry("DraftTubePlugin.INTERACTOR_SIZE");

   if (line)
      interactorSize=atof(line);
   else
   {
      fprintf(stderr,"WARNING: DraftTubePlugin.INTERACTOR_SIZE missing in covise.config\n");
      fprintf(stderr,"         setting size to 15 [mm]\n");
      interactorSize=15.0;
   }
   dir        = widthPoint-middlePoint;
   sliderZero = 0.5*dt->cs[index]->c_width;
   widthInteractor = new coVR1DTransInteractor(widthPoint, dir, -0.5, 0.5, sliderCurrent-sliderZero, interactorSize, coVR1DTransInteractor::SHOW_CONSTRAINTS_ALWAYS, polyObj->getName());
   widthInteractor->disable();
   widthInteractor->hide();

   sprintf(buf, "%s_height", polyObj->getName());
   heightIName      = strdup(buf);
   heightDir        = heightPoint-middlePoint;
   heightS0         = 0.5*dt->cs[index]->c_height;
   heightInteractor = new coVR1DTransInteractor(heightPoint, heightDir, -0.5, 0.5, heightSC-heightS0, interactorSize, coVR1DTransInteractor::SHOW_CONSTRAINTS_ALWAYS, heightIName);
   heightInteractor->disable();
   heightInteractor->hide();

#ifdef   IA_AREA
   sprintf(buf, "%s_area", polyObj->getName());
   areaIName      = strdup(buf);
   areaDir        = areaPoint-middlePoint;
   areaS0         = 0.5*CalcOneCSArea(dt->cs[index]);
   areaInteractor = new coVR1DTransInteractor(areaPoint, areaDir, -0.5, 0.5, areaSC-areaS0, interactorSize, coVR1DTransInteractor::SHOW_CONSTRAINTS_ALWAYS, areaIName);
   areaInteractor->disable();
   areaInteractor->hide();
#endif

#ifdef   IA_AB
   sprintf(buf, "%s_a", polyObj->getName());
   aIName      = strdup(buf);
   aDir        = aPoint-aDirPoint;
   aS0         = aSC;
   aInteractor = new coVR1DTransInteractor(aPoint, aDir, -aSC, 0.5, aSC-aS0, interactorSize, coVR1DTransInteractor::SHOW_CONSTRAINTS_ALWAYS, aIName);
   aInteractor->disable();
   aInteractor->hide();

   sprintf(buf, "%s_b", polyObj->getName());
   bIName      = strdup(buf);
   bDir        = bPoint-bDirPoint;
   bS0         = bSC;
   bInteractor = new coVR1DTransInteractor(bPoint, bDir, dt->cs[index]->c_height/2, 0.5, bSC-bS0, interactorSize, coVR1DTransInteractor::SHOW_CONSTRAINTS_ALWAYS, bIName);
   bInteractor->disable();
   bInteractor->hide();
#endif
}


void
CrossSection::createOutline()
{

   pfVec4 *colors;
   pfGeoState *geoState;
   pfMaterial *mtl;
   int n;
   int i;
   struct Point *p;

   p = CS_BorderPoints(dt->cs[index]);
   n=p->nump+1;                                   // aktuelle Anz

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

   lc = (pfVec3 *) pfCalloc(n, sizeof(pfVec3), pfGetSharedArena());
   for (i=0; i<n-1; i++)
   {
      lc[i].set(p->x[i], p->y[i], p->z[i]);
   }
   lc[n-1]=lc[0];
   FreePointStruct(p);

   colors  = (pfVec4 *) pfCalloc(1, sizeof(pfVec4), pfGetSharedArena());
   colors[0].set(1, 0, 0, 1);

   len = (int *) pfCalloc(1, sizeof(pfVec4), pfGetSharedArena());
   len[0]=n;

   outlineGeoset = new pfGeoSet();
   outlineGeoset->setAttr(PFGS_COORD3, PFGS_PER_VERTEX, lc, NULL);
   outlineGeoset->setAttr(PFGS_COLOR4, PFGS_OVERALL, colors, NULL);
   outlineGeoset->setPrimType(PFGS_LINESTRIPS);
   outlineGeoset->setPrimLengths(len);
   outlineGeoset->setLineWidth(3.0);
   outlineGeoset->setNumPrims(1);
   outlineGeoset->setGState(geoState);

   outlineGeode = new pfGeode();
   outlineGeode->addGSet(outlineGeoset);
   cover->getObjectsRoot()->addChild(outlineGeode);

#ifdef   DEBUG_ALL_XXX
   fprintf(stderr,"numPoints=[%d]\n", n);
   for (i=0; i<n; i++)
   {
      fprintf(stderr,"lc[%d]=[%f %f %f]\n", i, lc[i][0], lc[i][1], lc[i][2]);
   }
#endif
}


void
CrossSection::showOutline()
{
   fprintf(stderr,"\nSHOW OUTLINE [%d]\n", index);
   outlineGeode->setTravMask(PFTRAV_DRAW, 1, PFTRAV_SELF | PFTRAV_DESCEND, PF_SET);
}


void
CrossSection::hideOutline()
{
   fprintf(stderr,"\nHIDE OUTLINE [%d]\n", index);
   outlineGeode->setTravMask(PFTRAV_DRAW, 0, PFTRAV_SELF | PFTRAV_DESCEND, PF_SET);
}


void
CrossSection::updateOutline()
{

   int n;
   int i;
   struct Point *p;

   p = CS_BorderPoints(dt->cs[index]);
   n=p->nump+1;                                   // aktuelle Anz

   if (lc)
   {
      pfDelete(lc);
      lc=NULL;
   }
   if (len)
   {
      pfDelete(len);
      len=NULL;
   }
   lc = (pfVec3 *) pfCalloc(n, sizeof(pfVec3), pfGetSharedArena());
   for (i=0; i<n-1; i++)
   {
      lc[i].set(p->x[i], p->y[i], p->z[i]);
   }
   lc[n-1]=lc[0];
   FreePointStruct(p);

   len = (int *) pfCalloc(1, sizeof(pfVec4), pfGetSharedArena());
   len[0]=n;

   outlineGeoset->setAttr(PFGS_COORD3, PFGS_PER_VERTEX, lc, NULL);
   outlineGeoset->setPrimLengths(len);

#ifdef   DEBUG_ALL_XXX
   fprintf(stderr,"numPoints=[%d]\n", n);
   for (i=0; i<n; i++)
   {
      fprintf(stderr,"lc[%d]=[%f %f %f]\n", i, lc[i][0], lc[i][1], lc[i][2]);
   }
#endif
}


void
CrossSection::deleteOutline()
{
   if (outlineGeode)
      cover->getObjectsRoot()->removeChild(outlineGeode);
}


void
CrossSection::preFrame()
{
   widthInteractor->preFrame();
   heightInteractor->preFrame();
#ifdef   IA_AREA
   areaInteractor->preFrame();
#endif
#ifdef   IA_AB
   aInteractor->preFrame();
   bInteractor->preFrame();
#endif

   if (!interactionOngoing)
   {
      if (grabbed)
      {
         interactionOngoing = true;
         //fprintf(stderr,"\nSTART [%d]\n", index);
      }
   }
   else                                           // if interaction ongoing
   {

      if ( cover->getButton()->wasPressed()  && (cover->getPointerButton() == ACTION_BUTTON) &&  !widthInteractor->isSelected()
         && !heightInteractor->isSelected()
   #ifdef   IA_AREA
         && !areaInteractor->isSelected()
   #endif
   #ifdef   IA_AB
         && !aInteractor->isSelected()
         && !bInteractor->isSelected()
   #endif
         )
      {
         //fprintf(stderr,"\nRELEASED [%d]\n", index);
         //fprintf(stderr,"\nSTOP [%d]\n", index);
         // stop move
         interactionOngoing = false;

         grabbed = false;
         isected = false;
         outlineGeoset->setHlight(PFHL_OFF);
         widthInteractor->hide();
         widthInteractor->disable();
         heightInteractor->hide();
         heightInteractor->disable();
#ifdef   IA_AREA
         areaInteractor->hide();
         areaInteractor->disable();
#endif
#ifdef   IA_AB
         aInteractor->hide();
         aInteractor->disable();
         bInteractor->hide();
         bInteractor->disable();
#endif
         //feedback->executeModule();
         DraftTube::vrdt->unlockAllCrossSection();
      }
      else
      {
         //fprintf(stderr,"\nONGOING [%d]\n", index);
         if (widthInteractor->isSelected())
         {
            float v=widthInteractor->getValue();
            float params[2];
            params[0]=dt->cs[index]->c_height;
            dt->cs[index]->c_width=2.0*sliderZero+2.0*v;
            params[1]=dt->cs[index]->c_width;
            CalcCSGeometry(dt, index, NULL);      // update geometry
            updateOutline();
            feedback->setVectorParam(feedback->getParaName(0), 2, params);
            fprintf(stderr,"updating cross section [%d] width to [%f]\n", index, params[1]);
         }
         if (heightInteractor->isSelected())
         {
            float v = heightInteractor->getValue();
            float params[2];
            dt->cs[index]->c_height=2.0*heightS0+2.0*v;
            params[0]=dt->cs[index]->c_height;
            params[1]=dt->cs[index]->c_width;
            CalcCSGeometry(dt, index, NULL);      // update geometry
            updateOutline();
            feedback->setVectorParam(feedback->getParaName(0), 2, params);
            fprintf(stderr,"updating cross section [%d] height to [%f]\n", index, params[0]);
         }
#ifdef   IA_AREA
         if (areaInteractor->isSelected())
         {
            fprintf(stderr,"updating cross section [%d] area to [%f]\n", index, -999.99);
         }
#endif
#ifdef   IA_AB
         if (aInteractor->isSelected())
         {
            float v = aInteractor->getValue();
            float params[2];
            int j;

            dt->cs[index]->c_a[0] = dt->cs[index]->c_a[1] = dt->cs[index]->c_a[2]
               = dt->cs[index]->c_a[3] = aS0+v;
            params[0]=dt->cs[index]->c_a[0];
            params[1]=dt->cs[index]->c_b[0];
            CalcCSGeometry(dt, index, NULL);      // update geometry
            updateOutline();
            for (j = 0; j < 4; j++)
            {
               feedback->setVectorParam(feedback->getParaName(4+j), 2, params);
               fprintf(stderr,"updating cross section [%d] a to [%f]\n", index, params[0]);
            }
         }
         if (bInteractor->isSelected())
         {
            float v = bInteractor->getValue();
            float params[2];
            int j;

            dt->cs[index]->c_b[0] = dt->cs[index]->c_b[1] = dt->cs[index]->c_b[2]
               = dt->cs[index]->c_b[3] = bS0+v;
            params[0]=dt->cs[index]->c_a[0];
            params[1]=dt->cs[index]->c_b[0];
            CalcCSGeometry(dt, index, NULL);      // update geometry
            updateOutline();
            for (j = 0; j < 4; j++)
            {
               feedback->setVectorParam(feedback->getParaName(4+j), 2, params);
               fprintf(stderr,"updating cross section [%d] b to [%f]\n", index, params[1]);
            }
         }
#endif
      }
   }
}


void
CrossSection::update(coInteractor *inter)
{
   fprintf(stderr,"CrossSection::update\n");
   // here we don;t store the interactor, just use it once
   //hideOutline();
   polyObj = (coDistributedObject*) inter->getObject();
   updatePoints(inter, NULL);
   widthInteractor->setValue(sliderCurrent-sliderZero);
   heightInteractor->setValue(heightSC-heightS0);
#ifdef   IA_AREA
   areaInteractor->setValue(areaSC-areaS0);
#endif
#ifdef   IA_AB
   aInteractor->setValue(aSC);
   bInteractor->setValue(bSC);
#endif
   updateOutline();
}


void
CrossSection::updatePoints(coInteractor *inter, float n[3])
{
   fprintf(stderr,"CrossSection::updatePoints (index=%d)\n", index);

   int num;
   int j;
   float *ftmp;

   // height, width
   inter->getFloatVectorParam(0, num, ftmp);

   if (num != 2)
   {
      fprintf(stderr, "num != 2 (%s, %d)\n", __FILE__, __LINE__);
      exit(1);
   }
   dt->cs[index]->c_height = ftmp[0];
   dt->cs[index]->c_width  = ftmp[1];
   sliderCurrent = 0.5*dt->cs[index]->c_width;
   fprintf(stderr,"width=[%f]\n", dt->cs[index]->c_width);

   // middle point
   inter->getFloatVectorParam(1, num, ftmp);
   dt->cs[index]->c_m_x = ftmp[0];
   dt->cs[index]->c_m_y = ftmp[1];
   dt->cs[index]->c_m_z = ftmp[2];
   fprintf(stderr,"middlePoint=[%f %f %f]\n", dt->cs[index]->c_m_x, dt->cs[index]->c_m_y, dt->cs[index]->c_m_z);

   // angle
   inter->getFloatScalarParam(0, dt->cs[index]->c_angle);
   inter->getBooleanParam(0, dt->cs[index]->c_angletype);

   // a, b (corners)
   for (j = 0; j < 4; j++)
   {
      inter->getFloatVectorParam(4+j, num, ftmp);
      dt->cs[index]->c_a[j] = ftmp[0];
      dt->cs[index]->c_b[j] = ftmp[1];
   }
   heightSC = 0.5*dt->cs[index]->c_height;
#ifdef   IA_AREA
   areaSC   = 0.5*CalcOneCSArea(dt->cs[index]);
#endif
#ifdef   IA_AB
   aSC      = dt->cs[index]->c_width  - dt->cs[index]->c_a[0];
   bSC      = dt->cs[index]->c_height - dt->cs[index]->c_b[0];
#endif

   struct tubeCS_VR *tvr;
   if (n)
      fprintf(stderr, "\tn[0]=%f, n[1]=%f, n[2]=%f\n", n[0], n[1], n[2]);
   else
      fprintf(stderr, "\tn[0]=NULL, n[1]=NULL, n[2]=NULL\n");

   CalcCSGeometry(dt, index, n);
   tvr = TubeCsVrPoints(dt->cs[index], interactorSize);

   middlePoint.set(tvr->p->x[VR_MIDDLE], tvr->p->y[VR_MIDDLE], tvr->p->z[VR_MIDDLE]);
   widthPoint.set(tvr->p->x[VR_WIDTH], tvr->p->y[VR_WIDTH], tvr->p->z[VR_WIDTH]);
   heightPoint.set(tvr->p->x[VR_HEIGHT], tvr->p->y[VR_HEIGHT], tvr->p->z[VR_HEIGHT]);
#ifdef   IA_AREA
   areaPoint.set(tvr->p->x[VR_AREA], tvr->p->y[VR_AREA], tvr->p->z[VR_AREA]);
#endif
#ifdef   IA_AB
   aPoint.set(tvr->p->x[VR_A], tvr->p->y[VR_A], tvr->p->z[VR_A]);
   aDirPoint.set(tvr->dir->x[VR_A], tvr->dir->y[VR_A], tvr->dir->z[VR_A]);
   bPoint.set(tvr->p->x[VR_B], tvr->p->y[VR_B], tvr->p->z[VR_B]);
   bDirPoint.set(tvr->dir->x[VR_B], tvr->dir->y[VR_A], tvr->dir->z[VR_B]);
#endif
   radius=tvr->r;
   FreeTubeCsVrStruct(tvr);

   fprintf(stderr,"middlePoint[%d]=[%f %f %f]\n", index, middlePoint[0], middlePoint[1], middlePoint[2]);
   fprintf(stderr,"widthPoint[%d]=[%f %f %f]\n", index, widthPoint[0], widthPoint[1], widthPoint[2]);
   fprintf(stderr,"heightPoint[%d]=[%f %f %f]\n", index, heightPoint[0], heightPoint[1], heightPoint[2]);
}


int
CrossSection::hit(pfVec3 &hitPoint,pfHit *hit)
{
   if (!cover->isPointerLocked() && !cover->isNavigating() && !grabbed)
   {
      if (!isected)
      {
         isected = true;
         //showOutline();

         outlineGeoset->setHlight(isectedHl);
      }
      // get pointer state
      coPointerButton *pb = cover->getButton();

      if (pb->wasPressed() && (pb->getButtonStatus() ==  ACTION_BUTTON)  && !widthInteractor->isSelected()
         && !heightInteractor->isSelected()
   #ifdef   IA_AREA
         && !areaInteractor->isSelected()
   #endif
   #ifdef   IA_AB
         && !aInteractor->isSelected()
         && !bInteractor->isSelected()
   #endif
         )
      {
         fprintf(stderr,"\nGRABBED [%d]\n", index);
         widthInteractor->show();
         widthInteractor->enable();
         heightInteractor->show();
         heightInteractor->enable();
#ifdef   IA_AREA
         areaInteractor->show();
         areaInteractor->enable();
#endif
#ifdef   IA_AB
         aInteractor->show();
         aInteractor->enable();
         bInteractor->show();
         bInteractor->enable();
#endif
         grabbed = true;
         DraftTube::vrdt->lockCrossSection(index);
      }
   }

   return ACTION_CALL_ON_MISS;
}


void
CrossSection::miss()
{
   isected = false;

   fprintf(stderr,"\ncoVR1DTransInteractor::miss\n");

   if (!grabbed)
   {
      //hideOutline();

      outlineGeoset->setHlight(PFHL_OFF);
   }

}


void
CrossSection::setGeode(pfGeode *geode)
{
   fprintf(stderr,"CrossSection::setGeode");
   if (csGeode)
   {
      intersector.remove(csGeode);
      fprintf(stderr,"CrossSection::disable [cs_%d]\n", index);
   }
   csGeode=geode;
   if (csGeode)
   {
      intersector.add(csGeode, this);
      fprintf(stderr,"CrossSection::enable [cs_%d]\n", index);

   }
   enabled=true;
}


void
CrossSection::enable()
{
   fprintf(stderr,"CrossSection::enable [cs_%d]\n", index);
   if (csGeode && !enabled)
   {
      intersector.add(csGeode, this);
   }
   enabled=true;
}


void
CrossSection::disable()
{
   fprintf(stderr,"CrossSection::disable [cs_%d]\n", index);

   if (csGeode && enabled)
   {
      intersector.remove(csGeode);

   }
   enabled=false;
}
