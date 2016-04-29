#ifndef _DRAFTTUBE_PLUGIN_H
#define _DRAFTTUBE_PLUGIN_H

// plugin class for drafttube
// D. Rainer
// 08-nov-2001
// (C) 2001 RUS

#include <cover/coVRModuleSupport.h>
#include "DraftTube.h"

class PluginObject;
class DraftTubePlugin
{
   private:

      DraftTube *vrtube;
      int csCounter;                              // counter is incremented after each newinteractor(poly)
      int numCS;                                  // number of cross sections
      int createGrid;                             // boolean to start the computation
   public:

      DLinkList<PluginObject*> pluginObjectList;

      DraftTubePlugin(coVRModule *m);
      ~DraftTubePlugin();

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
