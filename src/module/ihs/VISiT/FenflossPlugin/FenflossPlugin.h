#ifndef _FENFLOSSP_PLUGIN_H
#define _FENFLOSSP_PLUGIN_H

#include <coVRModuleSupport.h>

class PluginObject;

class FenflossPlugin
{
   private:
      int v_pauseSim;
      int v_GetSimData;
      char **v_updateDelay;
      int num_updateDelay;
      int sel_updateDelay;
      int v_stopSim;
      char **v_bcSelect;
      int num_bcSelect;
      int sel_bcSelect;
      coInteractor *feedback;
      static void   lastCallback(void* , buttonSpecCell* );
      static void   iterCallback(void* , buttonSpecCell* );
      static void   pauseCallback(void* , buttonSpecCell* );
      static void   getdataCallback(void* , buttonSpecCell* );
      static void   stopCallback(void* , buttonSpecCell* );

   public:

      DLinkList<PluginObject*> pluginObjectList;

      FenflossPlugin(coVRModule *m);
      ~FenflossPlugin();

      // this will be called in PreFrame
      void preFrame();

      // this will be called if an object with feedback arrives
      void newInteractor(coDoGeometry *container, coInteractor *i);

      // this will be called if a COVISE object has to be removed
      void removeObject(const char *name, int replace);

      //
      void addNode(pfNode *node, coDistributedObject *obj);

      void removeNode(pfNode *node);

};

class PluginObject
{
   public:
      coDistributedObject *dobj;
      pfNode *node;

      PluginObject(coDistributedObject *d){dobj=d;node=NULL;};
      ~PluginObject(){};
};
#endif
