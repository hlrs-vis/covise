// plugin class for drafttube
// D. Rainer
// 08-nov 2001
// (C) 2001 University of Stuttgart

#include <cover/coVRModuleSupport.h>
#include <cover/coVRModuleList.h>

#include "DraftTubePlugin.h"

DraftTubePlugin *plugin = NULL;

int coVRInit(coVRModule *m)
{
   fprintf(stderr,"\n---- Initialize Plugin %s\n", m->getName());

   plugin = new DraftTubePlugin(m);

   return (plugin ? 1 : 0);
}


void coVRDelete(coVRModule *m)
{
   fprintf(stderr,"\n---- Delete Plugin %s\n", m->getName());

   delete plugin;
}


void coVRPreFrame()
{
   plugin->preFrame();
}


void coVRNewInteractor(coDoGeometry *cont, coInteractor *inter)
{
   const char *moduleName = inter->getModuleName();
   if (strcmp(moduleName, "DraftTube") == 0)
   {
      fprintf(stderr,"\n---- coVRNewInteractor [%s]\n", inter->getObject()->getName());

      plugin->pluginObjectList.append(new PluginObject(inter->getObject()));
      plugin->newInteractor(cont, inter);

   }
}


void coVRAddNode(pfNode *node, coDistributedObject *obj)
{
   plugin->pluginObjectList.reset();
   while (plugin->pluginObjectList.current())
   {
      if (plugin->pluginObjectList.current()->dobj == obj)
      {
         fprintf(stderr, "\n---- coVRAddNode dcs of [%s]\n", ((pfDCS*)node)->getChild(0)->getName());

         plugin->pluginObjectList.current()->node = node;
         plugin->addNode(node, obj);
         break;
      }
      plugin->pluginObjectList.next();
   }
}


void coVRRemoveNode(pfNode *node)
{

   plugin->pluginObjectList.reset();
   while (plugin->pluginObjectList.current())
   {
      if (plugin->pluginObjectList.current()->node == node)
      {
         fprintf(stderr, "\n---- coVRRemoveNode dcs of [%s]\n", ((pfDCS*)node)->getChild(0)->getName());

         plugin->pluginObjectList.current()->node=NULL;
         plugin->removeNode(node);
         break;
      }
      plugin->pluginObjectList.next();
   }
}


void coVRRemoveObject(const char *objName, int replace)
{
   plugin->pluginObjectList.reset();
   while (plugin->pluginObjectList.current())
   {
      if (strcmp(plugin->pluginObjectList.current()->dobj->getName(), objName) == 0)
      {

         fprintf(stderr,"\n---- coVRRemoveObject [%s]\n", objName);

         plugin->pluginObjectList.remove();
         plugin->removeObject(objName, replace);
         break;
      }
      plugin->pluginObjectList.next();
   }
}


//-----------------------------------------------------------------------------

DraftTubePlugin::DraftTubePlugin(coVRModule *m)
{
   fprintf(stderr, "DraftTubePlugin::DraftTubePlugin() ...\n");
   vrtube=NULL;

}


DraftTubePlugin::~DraftTubePlugin()
{
   fprintf(stderr, "DraftTubePlugin::~DraftTubePlugin() ...\n");

}


void DraftTubePlugin::newInteractor(coDoGeometry *container, coInteractor *inter)
{
   fprintf(stderr, "DraftTubePlugin::newInteractor() ...\n");

   // im ersten Durchlauf kommt SET
   if (strcmp(inter->getObject()->getType(), "SETELE" ) == 0 )
   {
      if (vrtube)                                 // then we had a remove with replace=true before
         vrtube->update(inter);
      else                                        // then it is firststime or we had remove with replace=false
         vrtube = new DraftTube(inter);

      // to start the computation (with grid generation)
      inter->getBooleanParam(0, createGrid);

      numCS=vrtube->getNumCrossSection();
      csCounter=-1;
   }
   else                                           // hier kommt ein SET-Element (Cross-Section)
   {
      csCounter++;
      vrtube->setCrossSection(inter, csCounter);
   }
}


void
DraftTubePlugin::removeObject(const char *name, int replace)
{
   fprintf(stderr, "DraftTubePlugin::removeObject() ...\n");
   if (!replace)
   {

      // it is the set
      if (strcmp(vrtube->getSetObject()->getName(), name) == 0)
      {
         csCounter=0;
         // the set object is deleted
         delete vrtube;
         vrtube=NULL;
      }
      //  it is the set element
      else
      {
         csCounter++;
         vrtube->deleteCrossSection(csCounter);
      }

   }
}


void DraftTubePlugin::preFrame()
{

   vrtube->preFrame();

}


void
DraftTubePlugin::addNode(pfNode *node, coDistributedObject *obj)
{
   fprintf(stderr, "DraftTubePlugin::addNode() ...\n");

   if (csCounter!= -1)
   {
      pfGeode *geode = (pfGeode *)((pfDCS*)node)->getChild(0);
      vrtube->setCrossSectionGeode(csCounter, geode);
   }

   if (csCounter == numCS-1)
   {
      csCounter=-1;
   }

}


void DraftTubePlugin::removeNode(pfNode *node)
{
   fprintf(stderr, "DraftTubePlugin::removeNode() ...\n");
   if (csCounter!= -1)
   {
      vrtube->setCrossSectionGeode(csCounter, NULL);
   }
}
