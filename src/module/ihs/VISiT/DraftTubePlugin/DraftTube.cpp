#include "DraftTube.h"
DraftTube *DraftTube::vrdt=NULL;

DraftTube::DraftTube(coInteractor *inter)
{
   int i;
   vrdt=this;

   fprintf(stderr,"DraftTube::DraftTube (Konstruktor)\n");
   inter->incRefCount();
   feedback=inter;

   // get the COVISE set object
   setObj=(coDistributedObject*)inter->getObject();

   // get the number of cross sections
   fprintf(stderr,"\tDraftTube::DraftTube (before call of extractInfo())\n");
   extractInfo(inter);
   fprintf(stderr,"\tDraftTube::DraftTube (after call of extractInfo())\n");

   cover->addToggleButton("RecomputeCSGeom", NULL, false, (void*)recomputeCallback, this);

   // get the state of the create grid button
   inter->getBooleanParam(0, createGridState);
   cover->addToggleButton("CreateDraftubeGrid", NULL, false, (void*)gridCallback, this);

   dt = AllocTube();

   csList = new CrossSection*[numCS];
   for (i=0; i< numCS; i++)
      csList[i] = NULL;
}


DraftTube::~DraftTube()
{
   fprintf(stderr,"DraftTube::~DraftTube\n");
   cover->removeButton("CreateDraftubeGrid", NULL);
   cover->removeButton("RecomputeCSGeom", NULL);
   delete []csList;
   FreeTube(dt);
   //feedback->decRefCount();
}


void
DraftTube::update(coInteractor *inter)
{
   fprintf(stderr,"DraftTube::update numCS\n");
   int createGridState;

   extractInfo(inter);
   setObj=(coDistributedObject*)inter->getObject();

   inter->getBooleanParam(0, createGridState);
   cover->setButtonState("CreateDraftubeGrid", false);
   cover->setButtonState("RecomputeCSGeom", false);

}


void
DraftTube::extractInfo(coInteractor *inter)
{
   char *tmp, *p;
   const char *str;

   fprintf(stderr,"DraftTube::extractInfo\n");
   str = inter->getString(0);

   if (str)
   {
      int i;
      sscanf(str, "NumCS=%d", &numCS);

      tmp = strdup(str);
      p = strtok(tmp, ";");
      for (i = 0; i < numCS; i++)
      {
         p = strtok(NULL, ";");
         sscanf(p, "%f:%f:%f", &MP[i][0], &MP[i][1], &MP[i][2]);
         fprintf(stderr, "MP[%d] = (%f/%f/%f)\n", i, MP[i][0], MP[i][1], MP[i][2]);
      }
   }
}


void
DraftTube::gridCallback(void *d, buttonSpecCell* spec)
{
   if (spec->state)
   {
      ((DraftTube*)d)->createGridState=true;
      ((DraftTube*)d)->feedback->setBooleanParam(((DraftTube*)d)->feedback->getParaName(0), true);
   }
   else
   {
      fprintf(stderr,"Seems that DraftTube module is still creating the grid\n");
   }
}


void
DraftTube::recomputeCallback(void *d, buttonSpecCell* spec)
{
   if (spec->state)
   {
      ((DraftTube*)d)->feedback->executeModule();
   }

}


void
DraftTube::preFrame()
{
   int i;
   for (i=0; i< numCS; i++)
      csList[i]->preFrame();
}


void
DraftTube::setCrossSection(coInteractor *inter, int index)
{
   fprintf(stderr,"DraftTube::setCrossSection\n");

   if (csList[index])
   {
      fprintf(stderr, "\told CS: index=%d\n", index);
      csList[index]->update(inter);
   }
   else
   {
      fprintf(stderr, "\tNEW CS: MP[%d]=(%f/%f/%f)\n", index, MP[index][0], MP[index][1], MP[index][2]);
      AllocT_CS(dt);
      csList[index] = new CrossSection(inter, dt, index, (index < numCS-1 ? MP[index+1] : NULL));
   }
}


void
DraftTube::deleteCrossSection(int index)
{
   fprintf(stderr,"DraftTube::deleteCrossSection\n");

   if (csList[index])
      delete csList[index];

}


void
DraftTube::setCrossSectionGeode(int index, pfGeode *geode)
{
   if (geode)
      fprintf(stderr,"DraftTube::setCrossSectionGeode [%s]\n", geode->getName());
   else
      fprintf(stderr,"DraftTube::setCrossSectionGeode NULL\n");
   if (csList[index])
      csList[index]->setGeode(geode);
}


void
DraftTube::lockCrossSection(int index)
{
   int i;
   for (i=0; i< numCS; i++)
   {
      if (index != i)
         csList[i]->disable();
   }
}


void
DraftTube::unlockAllCrossSection()
{
   int i;
   for (i=0; i< numCS; i++)
   {

      csList[i]->enable();
   }
}
