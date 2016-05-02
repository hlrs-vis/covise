#ifndef _STARCD_PLUGIN_H
#define _STARCD_PLUGIN_H
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

#include <list>
#include <osg/MatrixTransform>

#include <cover/coVRPluginSupport.h>
using namespace covise;
using namespace opencover;
using namespace vrui;


#include "StarRegion.h"

// order of function calls is:
// newInteractor for set
// addNode for set
// newInteractor for set element
// addNode for set element
// removeObject for set element
// removeNode for set element
// removeObject for set
// remove node for set

class PluginObject;

class StarCDPlugin : public coVRPlugin
{
 private:
   coInteractor *currentSetInter;              //interactor which comes with the set
   // we use this interactor to send parameters back
   // the interactors from the set elements are not saved
   coInteractor *currentObjInter;              // interactor of current set element

   int numRegions;
   int showInteractors;
   int counter;                                // incremented after a set element is added
   
   int interactionOngoing;                     // true if dragging an interactor
   
   char **regionNameList;
   StarRegion **regionList;
   StarRegion *activeRegion;
   int activeRegionIndex;                      // -1=off 0 = "vent1" 1 = "vent2" ...
   
   char *currentSetObjName;                    // pointer to the current covise set
   RenderObject *currentObj;              // pointer to the current covise element
   
   osg::ref_ptr<osg::Group> objectsRoot;
   
   // button group id for the radio buttons in the submenu
   ////int subMenuSwitchGroupId;
   
   // buttons in submenu"simcontrol"
   static void regionButtonCallback(void* c, buttonSpecCell* spec);
   
   static void switchButtonCallback(void* c, buttonSpecCell* spec);
   static void sliderButtonCallback(void* c, buttonSpecCell* spec);
   static void dialButtonCallback(void* c, buttonSpecCell* spec);
   
   int numSimControlParams;                    // number of sim control params
   char **simControlParamNameList;             // names of the boolean parameters which appear in the menu

   osg::Matrixf currentHandMat, initHandMat;

   int debug;
   int newConfig;                              // true if object is from a new configuration (different region geometry)
   int removed;                                // true if removeObhject was a real remove not a replace
   // in this case the controlSim menu and the regions were removed
   // and have to be created in addFeedback as if configuratons is new
   bool firsttime;
   void createSimControlMenu(coInteractor *inter);
   void deleteSimControlMenu();
   void updateSimControlMenu(coInteractor *inter);

   char residualLabels[200];
   char residualNumbers[200];

 public:
   typedef std::list<PluginObject*> PluginObjectList;
   PluginObjectList pluginObjectList;

   StarCDPlugin();
   ~StarCDPlugin();
   bool init();

   void newInteractor(RenderObject *, coInteractor *i);
   void addObject(RenderObject *,
                   RenderObject *geomobj, RenderObject *,
                   RenderObject *, RenderObject *,
                   osg::Group *,
                   int , int , int ,
                   float *, float *, float *, int *,
                   int , int ,
                   float *, float *, float *,
                   float );
   void removeObject(const char *objName, bool r);
   void addNode(osg::Node * node, RenderObject * obj);
   void removeNode(osg::Node * node, bool isGroup, osg::Node *realNode);

   // this will be called in PreFrame
   void preFrame();

   // this will be called if an object with feedback arrives
   // keyword SET:
   //   replace the old ist of region names with the new names
   //   if the new names are different replace the submenu
   //   replace list of StarRegion objects
   //   replace the StarRegion objects
   // keyword REGION: set the parameters (not yet implemented)
   void getFeedbackInfo(coInteractor *i);

   // this will be called if a COVISE object has to be removed
   // if obj is the set:
   //    if replace:
   //        test if the names changed
   //        if namesChanged
   //            delete regionNameList
   //            delete the submenu
   //        else
   //            set
   //    delete the list of StarRegion objects
   // if obj is a geode: delete the StarRegion objects
   void getRemoveObjectInfo(const char *objName, int r);
   
   // called if a node is appended to the COVER scene graph
   // if obj is the set: create a list of StarRegionsSensor objects
   // if obj is a region geometry: create a StarRegionsSensor object
   
   void getAddNodeInfo(osg::Node *node, RenderObject *obj);
   
   // called if a node is removed from the COVER scene graph
   // if obj is the set: delete the list of StarRegionsSensor objects
   // if obj is a region geometry: delete the StarRegionsSensor object
   void getRemoveNodeInfo(osg::Node *node);
   
   char *residualObjectName;
   void addResidualMenu(RenderObject *obj);
   void removeResidualMenu();
   
   bool showDials;
};

class PluginObject
{
 public:
   RenderObject *dobj;
   PluginObject(RenderObject *d);

   ~PluginObject();
   osg::ref_ptr<osg::Node> node;
   char *objName;
};
#endif
