/****************************************************************************\
 **                                                            (C)1999 RUS   **
 **                                                                          **
 ** Description: Plugin for StarCD simulation coupled to COVISE              **
 **                                                                          **
 **                                                                          **
 ** Author: D. Rainer                                                        **
 **                                                                          **
 ** History:                                                                 **
 ** October-99                                                               **
 **                                                                          **
 **                                                                          **
\****************************************************************************/
#ifdef WIN32
#include <winsock2.h>
#endif

#include <util/common.h>
#include <config/CoviseConfig.h>
#include <do/coDoData.h>

#include <osg/Group>
#include <osg/Node>
#include <osg/MatrixTransform>
#include <osg/Geode>
#include <osg/StateSet>
#include <osg/Material>

#include <cover/coVRPluginSupport.h>
#include <cover/coInteractor.h>

#include "StarCDPlugin.h"

#include <cover/RenderObject.h>
#include <vrml97/vrml/VrmlNodeCOVER.h>

using std::cerr;
using std::endl;

//StarCDPlugin * plugin = 0;


void StarCDPlugin::newInteractor(const RenderObject *, coInteractor *i)
{
   cerr << "StarCD::coVRNewInteractor info: called" << endl;
   const char *moduleName = i->getModuleName();
   if (strcmp(moduleName, "StarCD") == 0)
   {
      pluginObjectList.push_back(new PluginObject(i->getObject()));
      getFeedbackInfo(i);
   }
}


void StarCDPlugin::addObject(const RenderObject *, osg::Group *, const RenderObject *geomobj, const RenderObject *, const RenderObject *, const RenderObject *)
{
   if (!geomobj)
       return;

   cerr << "StarCD::addObject info: adding " << geomobj->getName()<< endl;

   if (strstr(geomobj->getName(), "StarCD"))
   {
      fprintf(stderr,"GOT STAR-CD OBJECT [%s]\n", geomobj->getName());

      if (geomobj->getAttribute("LABEL"))
      {
         fprintf(stderr,"found plot data\n");
         pluginObjectList.push_back(new PluginObject(geomobj));
         delete[] residualObjectName;
         residualObjectName = new char[strlen(geomobj->getName())+1];
         strcpy(residualObjectName,geomobj->getName());
         addResidualMenu(geomobj);
      }

   }
}


void StarCDPlugin::removeObject(const char *objName, bool r)
{

   cerr << "StarCD::coVRRemoveObject info: removing " << objName << endl;
   for (PluginObjectList::iterator it = pluginObjectList.begin(), next;
           it != pluginObjectList.end();
           it = next)
   {
       next = it+1;
       PluginObject *p = *it;
       if (strcmp(p->objName, objName) == 0)
       {
           pluginObjectList.erase(it);
           if (residualObjectName && (strcmp(objName,residualObjectName) == 0 ))
               removeResidualMenu();
           else
           {
               getRemoveObjectInfo(objName, r);
               break;
           }
       }
   }
}


void StarCDPlugin::addNode(osg::Node * node, RenderObject * obj)
{
   cerr << "StarCD::coVRAddNode info: adding node " << obj->getName() << endl;
   for (PluginObjectList::iterator it = pluginObjectList.begin();
           it != pluginObjectList.end();
           ++it)
   {
       PluginObject *p = *it;
       if (p->dobj == obj)
       {
           p->node = node;
           getAddNodeInfo(node, obj);
           break;
       }
   }
}


void StarCDPlugin::removeNode(osg::Node * node, bool /*isGroup*/, osg::Node * /*realNode*/)
{
   for (PluginObjectList::iterator it = pluginObjectList.begin();
           it != pluginObjectList.end();
           ++it)
   {
       PluginObject *p = *it;
      if (p->node == node)
      {
         cerr << "StarCD::coVRRemoveNode info: removing node "
              << node->asGroup()->getChild(0)->getName().c_str() << endl;

         p->node = NULL;
         getRemoveNodeInfo(node);
         break;
      }
   }
}


//----------------------------------------------------------------------------//

StarCDPlugin::StarCDPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
}

bool StarCDPlugin::init()
{
   //fprintf(stderr,"StarCDPlugin::StarCDPlugin\n");
   debug = coCoviseConfig::getInt("COVER.DebugLevel",0);
   debug = coCoviseConfig::getInt("COVER.Plugin.StarCD.DebugLevel",debug);

   newConfig = true;
   removed = false;
   firsttime = true;
   showDials = false;
   objectsRoot = 0;

   //currentSetObj = 0;
   currentSetObjName = 0;

   simControlParamNameList = 0;

   regionList = 0;
   regionNameList = 0;

   numRegions = 0;
   showInteractors = false;

   currentSetInter = 0;
   currentObjInter = 0;

   residualObjectName = 0;
   interactionOngoing = false;

   // make a button in the main menu
   // the entry opens a submenu

   cover->addSubmenuButton("StarCD ...", 0, "SimControl", false, 0, -1, this);

   cover->addSubmenuButton("Residuals ...", "SimControl", "Residuals", false, 0, -1, this);

   cover->addToggleButton("ShowDials", "SimControl", false, dialButtonCallback, this);

   // get the root of the covise objects
   objectsRoot = cover->getObjectsRoot();

   return true;
}


void StarCDPlugin::regionButtonCallback(void * c, buttonSpecCell * spec)
{

   StarCDPlugin *p;
   int i, j, k;

   p = static_cast<StarCDPlugin*>(c);

   if (p->debug)
      fprintf(stderr,"StarCDPlugin::regionButtonCallback [%s]=[%f]\n", spec->name, spec->state);

   // if the performer group object exists
   if (p->regionList)
   {

      fprintf(stderr,"... searching list of [%d] regions\n", p->numRegions);
      // find the appropriate region
      for (i = 0; i < p->numRegions; i++)
      {
         fprintf(stderr,"... comparing region [%d][%s]\n", i, p->regionNameList[i]);
         char *ename = new char[strlen(p->regionNameList[i])+10];
         sprintf(ename,"%s...", p->regionNameList[i]);
         if ( strcmp(ename, spec->name) ==0 )
         {
            fprintf(stderr,"... found region [%d][%s]\n", i, p->regionNameList[i]);
            delete []ename;
            if (spec->state == 1.0)
            {

               // draw region outlined
               p->regionList[i]->enableHighLight();

               // this version always show the arrow if the patch is visible
               for (j = 1; j < p->regionList[i]->getNumParams(); j++)
               {
                  k=j-1;
                  if (p->regionList[i]->interactorList[k]->getType() == StarRegionInteractor::Vector)
                  {
                     p->regionList[i]->interactorList[k]->show();
                     p->regionList[i]->interactorSensorList[k]->enable();
                  }
                  if (p->regionList[i]->interactorList[k]->getType() == StarRegionInteractor::Scalar)
                  {
                     if (p->showDials)
                     {
                        p->regionList[i]->interactorList[k]->show();
                        p->regionList[i]->interactorSensorList[k]->enable();
                     }
                  }

               }
            }
            else
            {
               // draw normal
               p->regionList[i]->disableHighLight();

               for (j = 1; j < p->regionList[i]->getNumParams(); j++)
               {
                  k=j-1;
                  p->regionList[i]->interactorList[k]->hide();
                  p->regionList[i]->interactorSensorList[k]->disable();
               }
            }
         }

      }                                           // end for
   }
   else
   {
      if (p->debug)
         fprintf(stderr,"\t no geometry ... doing nothing\n");
   }

}


void
StarCDPlugin::addResidualMenu(RenderObject *obj)
{
   char **names, **contents;
   float val[9];

   int i=0;
   obj->getAllAttributes(names,contents);
   //                        U--------V--------W--------P--------K--------EPS------T--------VIS------DEN-----
   //                        .--------.--------.--------.--------.--------.--------.--------.--------.--------
   sprintf(residualLabels, "%s             %s             %s             %s             %s             %s         %s             %s           %s",
      contents[1],contents[2],contents[3],contents[4],contents[5],contents[6],contents[7],contents[8],contents[9]);

   fprintf(stderr,"residualL=[%s]\n", residualLabels);

   for (i=0; i< 9; i++)
      ((coDoFloat*)obj)->getPointValue(i, &(val[i]));
   sprintf(residualNumbers, "%.2e %.2e %.2e %.2e %.2e %.2e %.2e %.2e %.2e",
      val[0],val[1], val[2],val[3],val[4],val[5],val[6],val[7],val[8]);

   fprintf(stderr,"residualN=[%s]\n", residualNumbers);

   cover->addFunctionButton(residualLabels, "Residuals", 0, this);
   cover->addFunctionButton(residualNumbers, "Residuals", 0, this);
}


void StarCDPlugin::removeResidualMenu()
{
   residualObjectName = 0;
   cover->removeButton(residualNumbers, "Residuals");
   cover->removeButton(residualLabels, "Residuals");
   //cover->removeButton("Residuals ...", "SimControl");

}


void StarCDPlugin::switchButtonCallback(void* c, buttonSpecCell* spec)
{

   StarCDPlugin *p;

   p = static_cast<StarCDPlugin*>(c);
   if (p->debug)
      fprintf(stderr,"StarCDPlugin::switchButtonCallback for name=[%s] state=[%f]\n", spec->name, spec->state);

   if (p->currentSetInter)
   {
      if (spec->state == 1.0)
      {
         // set the imm module parameter runSim
         p->currentSetInter->setBooleanParam(spec->name, true);
         if (p->debug)
            fprintf(stderr,"\tsetting module parameter %s to ON\n", spec->name);
      }
      else
      {
         p->currentSetInter->setBooleanParam(spec->name, false);
         if (p->debug)
            fprintf(stderr,"\tsetting module parameter %s to OFF\n", spec->name);

      }
   }
   else
   {
      if (p->debug)
         fprintf(stderr,"\tno interactor, can't send parameter\n");
   }

}


void StarCDPlugin::dialButtonCallback(void* c, buttonSpecCell* spec)
{

   StarCDPlugin *p;

   p = static_cast<StarCDPlugin*>(c);

   if (spec->state == 1.0)
   {
      p->showDials = true;
   }
   else
   {
      p->showDials = false;

   }

}


void StarCDPlugin::sliderButtonCallback(void* c, buttonSpecCell* spec)
{

   StarCDPlugin * p;

   p = static_cast<StarCDPlugin*>(c);

   if (p->debug)
      fprintf(stderr,"StarCDPlugin::sliderButtonCallback for name=[%s] state=[%f]\n", spec->name, spec->state);

   if (p->currentSetInter)
   {

      p->currentSetInter->setScalarParam(spec->name, (int) spec->state);

   }
   else
   {
      if (p->debug)
         fprintf(stderr,"\tno interactor, can't send parameter\n");
   }

}


StarCDPlugin::~StarCDPlugin()
{

}


// SET: user string with status, region names, parameter freeRun, parameter runSim
//    new nameList, new names
//    new regionList (new regions in addNode callback because we need geometry)
//
// SET ELEMENT: parameters which are modifiable boundary conditions
//    increment counter
void StarCDPlugin::getFeedbackInfo(coInteractor *inter)
{

   int i, j, k;

   // the order is
   // addObject
   // preFrame
   // replaceObject
   //   removeObject
   //   addObject
   // preFrame
   // replaceObject
   //  removeObject
   //  addObject

   // check if the interaction info comes from the module StarCD
   if (strcmp(inter->getPluginName(), "StarCD") == 0)
   {

      // check if it is the set
      if (inter->getObject()->isSet())
      {
         if (debug)
            fprintf(stderr,"\nStarCDPlugin::getFeedbackInfo for the SET\n");

         // get a pointer to the COVISE set object
         delete [] currentSetObjName;
         currentSetObjName = new char[strlen(inter->getObject()->getName())+1];
         strcpy(currentSetObjName,inter->getObject()->getName());

         // check if we got a new configuration
         // strings: NEW VENT1 VENT2 ...
         int numUserStrings = inter->getNumUser();
         if (numUserStrings> 0)
         {
            if (!strcmp(inter->getString(0), "NEW"))
            {
               newConfig = true;
               if (debug)
                  fprintf(stderr,"NEW CONFIGURATION\n");
            }
            else
               newConfig = false;
         }
         if (firsttime)
         {
            firsttime=false;
            newConfig=true;
         }
         if (removed)
         {
            removed=false;
            newConfig=true;
         }
         // if we got a new configuration or objects were really removed
         if (newConfig)
         {

            // the interactors/simControl menu
            // are new or have to be replaced

            // delete the old regions and interactors
            if (regionList && regionNameList)
            {

               for (i = 0; i< numRegions; i++)
               {

                  delete regionList[i];
                  char *ename = new char[strlen(regionNameList[i])+10];
                  sprintf(ename,"%s...", regionNameList[i]);
                  cover->removeButton(ename, "SimControl");
                  delete []ename;
                  delete []regionNameList[i];
               }

               delete []regionList;
               delete []regionNameList;
               regionList = 0;
               regionNameList = 0;

               // now we can decr the ref count
               currentSetInter->decRefCount();
               currentSetInter = 0;
            }

            // delete the sim control menu
            deleteSimControlMenu();

            // increase reference count because we want to use it
            inter->incRefCount();

            // save the interactor of the set to send parameters back
            currentSetInter = inter;

            // extract the region names
            numRegions = numUserStrings-1;

            regionNameList = new char*[numRegions];
            for (i = 0; i < numRegions; i++)
            {
               const char *tmp = inter->getString(i+1);
               regionNameList[i] = new char[strlen(tmp)+1];
               strcpy(regionNameList[i], tmp);
               if (debug)
                  fprintf(stderr,"region[%d]=[%s]\n", i, regionNameList[i]);
            }

            // create a (new) list of region objects
            regionList = new StarRegion*[numRegions];
            for (i = 0; i < numRegions; i++)
            {
               regionList[i]=0;
            }
            // create the submenu entries
            createSimControlMenu(inter);

            // reset the counter
            counter = -1;

         }
         else                                     // we have an old configuration
         {
            updateSimControlMenu(inter);
         }

      }
      else                                        // type is "POLYGON"
      {
         if (debug)
            fprintf(stderr,"\nStarCDPlugin::getFeedbackInfo for the POLYGON\n");

         currentObj = inter->getObject();

         if (newConfig)
         {

            currentObjInter = inter;
            inter->incRefCount();

            if (counter < numRegions)
            {
               // counter is the element index and needed in addNode
               counter++;
            }
            else
               fprintf(stderr,"StarCDPlugin ERROR: region index=[%d] numRegions=[%d]\n", counter, numRegions);

            // create new region
            regionList[counter] = new StarRegion(regionNameList[counter], counter, currentObjInter, true);

            // we don't need the element inter any more
            //// neuerdings doch //currentObjInter->decRefCount();

            // disable
            ////regionList[counter]->disableHighLight();
            for (j = 1; j < regionList[counter]->getNumParams(); j++)
            {
               k=j-1;
               regionList[counter]->interactorList[k]->hide();
               regionList[counter]->interactorSensorList[k]->disable();
            }

         }
         else
         {
            // find region
            for (i=0; i<numRegions; i++)
            {
               if (strcmp(inter->getString(0), regionList[i]->getName())==0)
               {
                  fprintf(stderr,"updating region [%s]\n", regionList[i]->getName());
                  regionList[i]->update(inter);
               }
            }
         }
      }
   }

}


void StarCDPlugin::createSimControlMenu(coInteractor *inter)
{
   int i;

   numSimControlParams = inter->getNumParam();

   if (debug)
      fprintf(stderr,"StarCDPlugin::createSimControlMenu - %d parameters\n", numSimControlParams);

   // alloc mem for the boolean parameter name list
   // may be less than numSimControlParams
   simControlParamNameList = new char*[numSimControlParams];

   for (i = 0; i < numSimControlParams; i++)
   {
      simControlParamNameList[i] = new char[strlen(inter->getParaName(i))+1];
      strcpy(simControlParamNameList[i], inter->getParaName(i));

      // the parameters attached to the set are
      // runSim (bool)
      // freeRun (bool)
      // numSteps (int scalar)
      // region (choice)
      if (strcmp(inter->getParaType(i), "BOOL") == 0)
      {

         int flag;
         inter->getBooleanParam(i, flag);

         if (debug)
            fprintf(stderr,"\taddButton [%s]\n", inter->getParaName(i));
         cover->addToggleButton(inter->getParaName(i), "SimControl", flag, switchButtonCallback, this);

      }
      else if ( strcmp(inter->getParaType(i), "INTSCA") == 0 )
      {
         int v;
         inter->getIntScalarParam(i, v);
         cover->addSliderButton(inter->getParaName(i),"SimControl", 1, 20, v, sliderButtonCallback, this);
      }
      else
      {
         fprintf(stderr,"ERROR in StarCDPlugin::createSimControlMenu - unsupported parameter type for [%s]\n", inter->getParaName(i) );
      }
   }

   // create the region entries
   for (i=0; i < numRegions; i++)
   {
      char *ename = new char[strlen(regionNameList[i])+10];
      sprintf(ename,"%s...", regionNameList[i]);
      cover->addSubmenuButton(ename, "SimControl",  regionNameList[i],false, regionButtonCallback, -1, this);

      delete []ename;
   }

}


void StarCDPlugin::deleteSimControlMenu()
{
   int i;
   if (debug)
      fprintf(stderr,"StarCDPlugin::deleteSimControlMenu\n");

   // first time?
   if (simControlParamNameList)
   {
      // remove the menu buttons
      for (i = 0; i < numSimControlParams; i++)   // delete runsim, freerun, step
      {
         if (debug)
            fprintf(stderr,"remove button [%s] from menu SimControl\n", simControlParamNameList[i]);

         if (simControlParamNameList[i]!=0)
         {
            cover->removeButton(simControlParamNameList[i], "SimControl");
            delete []simControlParamNameList[i];
         }
      }
      delete []simControlParamNameList;
      simControlParamNameList = 0;
   }
   else
   {
      fprintf(stderr,"\tsimControlParamNameList=0 - no button remove\n");
   }

}


void StarCDPlugin::updateSimControlMenu(coInteractor *inter)
{
   int i;
   int numSimControlParams = inter->getNumParam();

   for (i = 0; i < numSimControlParams; i++)
   {
      // the control parameters are typically runSim, freeRun, numSteps
      // only boolean control parameters are supported here
      if (strcmp(inter->getParaType(i), "BOOL") == 0)
      {
         int flag;
         inter->getBooleanParam(i, flag);
         cover->setButtonState(inter->getParaName(i), flag);
         if (debug)
            fprintf(stderr,"StarCDPlugin::updateSimControlMenu - [%s]=[%d]", inter->getParaName(i), flag);
      }
      else if ( strcmp(inter->getParaType(i), "INTSCA") == 0 )
      {
         int v;
         inter->getIntScalarParam(i, v);
         cover->setSliderValue(inter->getParaName(i), v);
      }

   }

}


// SET:
//    save performer group node
// SET ELEMENT:
//    set the geode

void
StarCDPlugin::getAddNodeInfo(osg::Node *node, RenderObject *obj)
{

   if (currentSetObjName && (strcmp(obj->getName(),currentSetObjName)==0))
   {
      if (debug)
         fprintf(stderr,"\nStarCDPlugin::getAddNodeInfo for the Performer GROUP\n");

   }

   else if (obj == currentObj)
   {
      int i;
      osg::Geode * geode  = dynamic_cast<osg::Geode*>(node->asGroup()->getChild(0));
      // go through all regions and check, if it has this distributed object
      for (i=0; i< numRegions; i++)
      {
         if (regionList[i])
         {
            if (strcmp(regionList[i]->getCoviseObjectName(),obj->getName())==0)
            {
               cerr << "StarCDPlugin::getAddNodeInfo info: " <<geode->getName() << endl;
               regionList[i]->setPatchesGeode(geode);
            }
         }
      }
   }

}


// SET:
//    delete names, delete nameList
//    delete regions and regionList
// SET ELEMENT:
//    -
void StarCDPlugin::getRemoveObjectInfo(const char *objName, int replace)
{
   int i;
   // test if it is our set object
   if (currentSetObjName && (!strcmp(currentSetObjName, objName)))
   {
      if (debug)
         fprintf(stderr,"\nStarCDPlugin::getRemoveObjectInfo for the Covise SET\n");

      if (!replace)                               // a real remove
      {

         // delete the regions
         // deleteRegions(); ... not yet implemented
         // delete the old regions and interactors
         if (regionList && regionNameList)
         {

            for (i = 0; i< numRegions; i++)
            {

               delete regionList[i];

               char *ename = new char[strlen(regionNameList[i])+10];
               sprintf(ename,"%s...", regionNameList[i]);
               cover->removeButton(ename, "SimControl");
               delete []ename;

               delete []regionNameList[i];
            }

            delete []regionList;
            delete []regionNameList;
            regionList = 0;
            regionNameList = 0;

            deleteSimControlMenu();

            // now we can decr the ref count
            currentSetInter->decRefCount();
            currentSetInter = 0;
         }

         // set a flag to indicate that next time the configuration hasto be read as if it is NEW
         removed = true;
      }

   }

}


void StarCDPlugin::getRemoveNodeInfo(osg::Node *node)
{

   /*
       if (!interactionOngoing)
       {
           // check if it is the set node
           if (node == perfGroupNode)
           {
               fprintf(stderr,"\nStarCDPlugin::getRemoveNodeInfo for the Performer GROUP\n");

               perfGroupNode = 0;

           }
   }
   */
   int i;
   osg::ref_ptr<osg::Geode> geode = dynamic_cast<osg::Geode*>(node->asGroup()->getChild(0));
   // go through all regions and check, if it has this node
   for (i=0; i< numRegions; i++)
   {
      if (regionList[i]->getPatchesGeode() == geode)
      {
         regionList[i]->setPatchesGeode(0);
      }
   }

}


void StarCDPlugin::preFrame()
{
   int i, j, k;
   //osg::Vec3f currentPos;
   //osg::Vec3f currentPos_o;                           //current hand position in object coordinates

   //osg::Vec3f center_o, center_w;

   if (regionList)
   {
      for (i = 0; i < numRegions; i++)
      {
         for (j = 1; j < regionList[i]->getNumParams(); j++)
         {
            k=j-1;
            regionList[i]->interactorList[k]->updateScale();
            if(regionList[i]->interactorList[k]->getType() == StarRegionInteractor::Vector)
            {
               if((i<8)&&(theCOVER))
               {
                  //regionList[i]->interactorList[k]->setTransform(theCOVER->transformations[i]);
                  osg::Matrix mat;
                  for(int l=0; l<16; l++)
                  {
                     mat.ptr()[l] = theCOVER->transformations[i][l];
                  }
                  regionList[i]->interactorList[k]->setTransform(mat);

                  osg::Vec3f currentVel;
                  regionList[i]->interactorList[k]->getValue(&(currentVel[0]), &(currentVel[1]), &(currentVel[2]));
                  fprintf(stderr,"--- set matrix : param = %s value = %f %f %f\n", regionList[i]->interactorList[k]->getParamName().c_str(), currentVel[0], currentVel[1], currentVel[2]);

               }
               if(regionList[i]->interactorList[k]->becameVisible())
               {

                  osg::Vec3f currentVel;
                  regionList[i]->interactorList[k]->getValue(&(currentVel[0]), &(currentVel[1]), &(currentVel[2]));

                  fprintf(stderr,"--- interactor became visible: param = %s value = %f %f %f\n", regionList[i]->interactorList[k]->getParamName().c_str(), currentVel[0], currentVel[1], currentVel[2]);
                  //currentSetInter->setVectorParam(regionList[i]->interactorList[k]->getParamName().c_str(), currentVel[0], currentVel[1], currentVel[2]);
                  regionList[i]->feedback->setVectorParam(regionList[i]->interactorList[k]->getParamName().c_str(), currentVel[0], currentVel[1], currentVel[2]);
                  regionList[i]->feedback->print(stderr);
                  for (int ctr = 0 ; ctr < regionList[i]->feedback->getNumParam(); ++ctr)
                  {
                     cerr << regionList[i]->feedback->getParaName(ctr) << " ("
                          << regionList[i]->feedback->getParaType(ctr) << "): "
                          << regionList[i]->feedback->getParaValue(ctr) << endl;
                  }
                  ////setInter->executeModule();

                  // beep 1x
                  fflush(stdout);
                  fprintf(stdout,"\a");
                  fflush(stdout);
               }
            }
            if (regionList[i]->interactorList[k]->isIsected())
            {

               // interactor is intersected & button is pressed
               // -> start dragging
               if (regionList[i]->interactorSensorList[k]->wasStarted())
               {

                  fprintf(stderr,"\nStarCDPlugin::preFrame interactionOngoing=[%d]\n", interactionOngoing);
                  regionList[i]->interactorList[k]->setSelectedHighLight();
                  // warum? regionList[i]->interactorSensorList[k]->disable();

                  initHandMat = cover->getPointerMat();

                  switch (regionList[i]->interactorList[k]->getType())
                  {
                     case StarRegionInteractor::Vector:
                        ((StarRegionVectorInteractor *)regionList[i]->interactorList[k])->startInteraction(initHandMat);
                        break;
                     case  StarRegionInteractor::Scalar:
                        ((StarRegionScalarInteractor *)regionList[i]->interactorList[k])->startRotation(initHandMat);
                        break;
                  }

               }

               // interactor is intersected
               else if (regionList[i]->interactorSensorList[k]->isRunning())
               {
                  //fprintf(stderr,"\n *** DRAGGING ONGOING\n");

                  switch (regionList[i]->interactorList[k]->getType())
                  {
                     case StarRegionInteractor::Vector:
                     {
                        // get the current hand matrix
                        currentHandMat = cover->getPointerMat();

                        ((StarRegionVectorInteractor *)regionList[i]->interactorList[k])->doInteraction(currentHandMat);
                        break;
                     }
                     case  StarRegionInteractor::Scalar:
                        currentHandMat = cover->getPointerMat();
                        // set the curent value through the initial matrix and the current matrix
                        ((StarRegionScalarInteractor *)regionList[i]->interactorList[k])->setRotation(currentHandMat);
                        break;
                  }

               }
               // dragging stop
               else if (regionList[i]->interactorSensorList[k]->wasStopped())
               {

                  fprintf(stderr,"\nStarCDPlugin::preFrame interactionOngoing=[%d]\n", interactionOngoing);
                  regionList[i]->interactorList[k]->setNormalHighLight();
                  //warum? regionList[i]->interactorSensorList[k]->enable();

                  osg::Vec3f currentVel;
                  regionList[i]->interactorList[k]->getValue(&(currentVel[0]), &(currentVel[1]), &(currentVel[2]));
                  // do vector interaction
                  if (regionList[i]->interactorList[k]->getType() == StarRegionInteractor::Vector)
                  {

                     /* float vmag = sqrt(  (currentVel[0] * currentVel[0])
                                        + (currentVel[1] * currentVel[1])
                                        + (currentVel[2] * currentVel[2]) );

                      fprintf(stderr,"VMAG=[%f]\n", vmag);
                      currentVel.normalize();
                      //float vmag = ((StarRegionVectorInteractor *)regionList[i]->interactor)->getMagnitude();
                      currentVel.scale(vmag, currentVel);
                     */

                     // beep 1x
                     fflush(stdout);
                     fprintf(stdout,"\a");
                     fflush(stdout);

                     fprintf(stderr,"--- param = %s value = %f %f %f\n", regionList[i]->interactorList[k]->getParamName().c_str(), currentVel[0], currentVel[1], currentVel[2]);
                     currentSetInter->setVectorParam(regionList[i]->interactorList[k]->getParamName().c_str(), currentVel[0], currentVel[1], currentVel[2]);
                     ////setInter->executeModule();

                     // beep 1x
                     fflush(stdout);
                     fprintf(stdout,"\a");
                     fflush(stdout);
                  }

                  // do scalar interaction
                  if (regionList[i]->interactorList[k]->getType() == StarRegionInteractor::Scalar)
                  {
                     float min, max, val;
                     regionList[i]->interactorList[k]->getValue(&min, &max, &val);
                     // beep 1x
                     fflush(stdout);
                     fprintf(stdout,"\a");
                     fflush(stdout);

                     fprintf(stderr,"--- param = %s value = %f %f %f\n", regionList[i]->interactorList[k]->getParamName().c_str(), min, max, val);
                     currentSetInter->setSliderParam(regionList[i]->interactorList[k]->getParamName().c_str(), min, max, val);
                     ////setInter->executeModule();

                     // beep 1x
                     fflush(stdout);
                     fprintf(stdout,"\a");
                     fflush(stdout);

                  }

               }
            }                                     // end interaction with interactor
         }

      }                                           // end for all regions
   }                                              // endif region

}


PluginObject::PluginObject(RenderObject *d)
{
   dobj=d;
   node=0;
   objName = new char[strlen(d->getName())+1];
   strcpy(objName,d->getName());
};

PluginObject::~PluginObject()
{
   delete[] objName;
}

COVERPLUGIN(StarCDPlugin)
