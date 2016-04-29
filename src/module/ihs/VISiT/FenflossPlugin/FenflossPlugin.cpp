#include <coVRModuleSupport.h>
#include <coVRModuleList.h>

#include "FenflossPlugin.h"

FenflossPlugin *plugin = NULL;

int coVRInit(coVRModule *m)
{
   fprintf(stderr,"\n---- Initialize Plugin %s\n", m->getName());

   plugin = new FenflossPlugin(m);

   return (plugin ? 1 : 0);
}


void coVRDelete(coVRModule *m)
{
   fprintf(stderr,"\n---- Delete Plugin %s\n", m->getName());

   delete plugin;
}


void coVRNewInteractor(coDoGeometry *cont, coInteractor *inter)
{
   const char *moduleName = inter->getModuleName();
   if (strcmp(moduleName, "Fenfloss") == 0)
   {
      //fprintf(stderr,"\n---- coVRNewInteractor [%s]\n", inter->getObject()->getName());

      plugin->pluginObjectList.append(new PluginObject(inter->getObject()));
      plugin->newInteractor(cont, inter);
   }
}


void coVRRemoveObject(const char *objName, int replace)
{
   plugin->pluginObjectList.reset();
   while (plugin->pluginObjectList.current())
   {
      if (strcmp(plugin->pluginObjectList.current()->dobj->getName(), objName) == 0)
      {
         //fprintf(stderr,"\n---- coVRRemoveObject [%s]\n", objName);

         plugin->pluginObjectList.remove();
         plugin->removeObject(objName, replace);
         break;
      }
      plugin->pluginObjectList.next();
   }
}


//-----------------------------------------------------------------------------

FenflossPlugin::FenflossPlugin(coVRModule *m)
{
   feedback=NULL;
}


FenflossPlugin::~FenflossPlugin()
{
   fprintf(stderr, "FenflossPlugin::~FenflossPlugin() ...\n");

}


void FenflossPlugin::newInteractor(coDoGeometry *container, coInteractor *inter)
{
   /*
   #ifdef	DEBUG_ALL
      int i;

       fprintf(stderr, "##################################################################\n");
       fprintf(stderr, "------------------------------------------------------------------\n");
       fprintf(stderr, "FenflossPlugin::newInteractor(): %s\n", inter->getObject()->getType());
       fprintf(stderr, "------------------------------------------------------------------\n");
       fprintf(stderr, "##################################################################\n");
   #endif
   */
   int i;
   int gid;
   bool state;
   num_bcSelect = sel_bcSelect = 0;
   num_updateDelay = sel_updateDelay = 0;

   inter->getChoiceParam(0, num_bcSelect, v_bcSelect, sel_bcSelect);
   inter->getChoiceParam(1, num_updateDelay, v_updateDelay, sel_updateDelay);
   inter->getBooleanParam(2, v_pauseSim);
   inter->getBooleanParam(3, v_GetSimData);
   inter->getBooleanParam(4, v_stopSim);

   if (!feedback)
   {
      feedback=inter;
      inter->incRefCount();

      cover.addSubmenuButton("Fenfloss...", NULL, "Fenfloss", true, NULL, -1, this);
      // parameter 0 (load bc)
      cover.addSubmenuButton("Load...", "Fenfloss", "Load", false, NULL, -1, this);
      gid=cover.uniqueButtonGroup();
      for (i=0; i<num_bcSelect; i++)
      {
         if (i==(sel_bcSelect-1))
            state=true;
         else
            state=false;

         cover.addGroupButton(v_bcSelect[i], "Load", state, (void*)lastCallback, gid, this);
      }

      // parameter 1 (iterations)
      cover.addSubmenuButton("Iter...", "Fenfloss", "Iter", false, NULL, -1, this);
      gid=cover.uniqueButtonGroup();
      for (i=0; i<num_updateDelay; i++)
      {
         if (i==(sel_updateDelay-1))
            state=true;
         else
            state=false;

         cover.addGroupButton(v_updateDelay[i], "Iter", state, (void*)iterCallback, gid, this);
      }

      // parameter 2 (pause)
      cover.addToggleButton("pauseSim", "Fenfloss", v_pauseSim, (void*)pauseCallback, this);

      // parameter 3 (getData in between)
      cover.addToggleButton("getData", "Fenfloss", v_GetSimData, (void*)getdataCallback, this);

      // parameter 3 (stop)
      cover.addToggleButton("stopSim", "Fenfloss", v_stopSim, (void*)stopCallback, this);
   }
   else                                           // update
   {
      cover.setButtonState(v_bcSelect[sel_bcSelect-1], true);
      cover.setButtonState(v_updateDelay[sel_updateDelay-1], true);
      cover.setButtonState("pauseSim", v_pauseSim);
      cover.setButtonState("getData", v_GetSimData);
      cover.setButtonState("getData", v_stopSim);
   }
   /*
   #ifdef	DEBUG_ALL
      fprintf(stderr, "v_pauseSim=%d\n", v_pauseSim);
      fprintf(stderr, "v_GetSimData=%d\n", v_GetSimData);
      fprintf(stderr, "v_stopSim=%d\n", v_stopSim);

      fprintf(stderr, "num_updateDelay=%d, sel_updateDelay=%d\n",num_updateDelay ,sel_updateDelay);
      for (i = 0; i < num_updateDelay; i++)
         fprintf(stderr, "  %d: %s\n", i, v_updateDelay[i]);

      fprintf(stderr, "num_bcSelect=%d, sel_bcSelect=%d\n",num_bcSelect ,sel_bcSelect);
   for (i = 0; i < num_bcSelect; i++)
   fprintf(stderr, "  %d: %s\n", i, v_bcSelect[i]);
   #endif
   */
}


void
FenflossPlugin::removeObject(const char *name, int replace)
{
   int i;
   //fprintf(stderr, "FenflossPlugin::removeObject() ...\n");
   if (!replace)
   {

      for (i=0; i<num_bcSelect; i++)
      {
         cover.removeButton(v_bcSelect[i], "Load");
      }
      cover.removeButton("Load...", "Fenfloss");

      for (i=0; i<num_updateDelay; i++)
      {
         cover.removeButton(v_updateDelay[i], "Load");
      }
      cover.removeButton("Iter...", "Fenfloss");
      cover.removeButton("pauseSim", "Fenfloss");
      cover.removeButton("getData", "Fenfloss");
      cover.removeButton("stopSim", "Fenfloss");
      cover.removeButton("Fenfloss...", NULL);
   }
}


void
FenflossPlugin::lastCallback(void *f, buttonSpecCell *spec)
{
   int i;
   for (i=0; i<((FenflossPlugin*)f)->num_bcSelect; i++)
   {
      if (strcmp(spec->name, ((FenflossPlugin*)f)->v_bcSelect[i]) == 0)
      {
         if (spec->state == 1)
         {

            fprintf(stderr,"button [%s] pressed\n", ((FenflossPlugin*)f)->v_bcSelect[i]);
            ((FenflossPlugin*)f)->feedback->setChoiceParam(((FenflossPlugin*)f)->feedback->getParaName(0), ((FenflossPlugin*)f)->num_bcSelect, ((FenflossPlugin*)f)->v_bcSelect, i+1);
         }
         else
            fprintf(stderr,"button [%s] released\n", ((FenflossPlugin*)f)->v_bcSelect[i]);

      }
   }

}


void
FenflossPlugin::iterCallback(void *f, buttonSpecCell *spec)
{
   int i;
   for (i=0; i<((FenflossPlugin*)f)->num_updateDelay; i++)
   {
      if (strcmp(spec->name, ((FenflossPlugin*)f)->v_updateDelay[i]) == 0)
      {
         if (spec->state == 1)
         {

            fprintf(stderr,"button [%s] pressed\n", ((FenflossPlugin*)f)->v_updateDelay[i]);
            ((FenflossPlugin*)f)->feedback->setChoiceParam(((FenflossPlugin*)f)->feedback->getParaName(1), ((FenflossPlugin*)f)->num_updateDelay, ((FenflossPlugin*)f)->v_updateDelay, i+1);
         }
         else
            fprintf(stderr,"button [%s] released\n", ((FenflossPlugin*)f)->v_updateDelay[i]);

      }
   }

}


void
FenflossPlugin::pauseCallback(void *f, buttonSpecCell *spec)
{
   if (spec->state)
   {
      ((FenflossPlugin*)f)->feedback->setBooleanParam(((FenflossPlugin*)f)->feedback->getParaName(2), true);
   }
   else
   {
      ((FenflossPlugin*)f)->feedback->setBooleanParam(((FenflossPlugin*)f)->feedback->getParaName(2), false);
   }

}


void
FenflossPlugin::getdataCallback(void *f, buttonSpecCell *spec)
{
   if (spec->state)
   {
      ((FenflossPlugin*)f)->feedback->setBooleanParam(((FenflossPlugin*)f)->feedback->getParaName(3), true);
   }
   else
   {
      ((FenflossPlugin*)f)->feedback->setBooleanParam(((FenflossPlugin*)f)->feedback->getParaName(3), false);
   }

}


void
FenflossPlugin::stopCallback(void *f, buttonSpecCell *spec)
{
   if (spec->state)
   {
      ((FenflossPlugin*)f)->feedback->setBooleanParam(((FenflossPlugin*)f)->feedback->getParaName(4), true);
   }
   else
   {
      ((FenflossPlugin*)f)->feedback->setBooleanParam(((FenflossPlugin*)f)->feedback->getParaName(4), false);
   }

}
