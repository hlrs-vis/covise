// Sa abend eingecheckt

// Fr abend eingecheckt

// Di abend eingecheckt
/****************************************************************************\ 
 **                                                            (C)1999 RUS   **
 **                                                                          **
 ** Description: class describing a STAR boundary condition region in COVER  **
 **                                                                          **
 **                                                                          **
 ** Author: D. Rainer                                                        **
 **                                                                          **
 ** History:                                                                 **
 ** October-99                                                               **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#ifndef _STAR_REGION_H
#define _STAR_REGION_H
#ifdef WIN32
#include <winsock2.h>
#endif

#include <osg/Geode>
#include <osg/StateSet>

#include <kernel/coVRPluginSupport.h>
using namespace covise;
using namespace opencover;


#include "StarRegionInteractor.h"
#include "InteractorSensor.h"

class StarRegion
{
 private:

   char *regionName;                           // "Duese links Mittelkonsole"
   int regionIndex;                            // c++ numbering from 0-numRegions

   osg::Vec3f center, normal;                      // region center and normal - works only for flat regions
   osg::Vec3f local;                               // euler angles of StarCD local coordinate system
   float radius;
   char *coviseObjName;
   
   osg::ref_ptr<osg::Geode> patchesGeode;
   osg::ref_ptr<osg::Drawable> geoset;
   osg::ref_ptr<osg::StateSet> gs;

   int numParams;                              // number of parameters
   int numInteractors;                         // number of interactive parameters
   //pfHighlight *highLight;

   bool enabled;
   int debug;

   void computeCenterAndNormal();
   void setCenter(const osg::Vec3f & p);
   void setLocal(const osg::Vec3f & l);
   static void sliderButtonCallback(void* c, buttonSpecCell* spec);

 public:

   coInteractor *feedback;
   StarRegionInteractor **interactorList;
   InteractorSensor **interactorSensorList;

   // constructor sets also th name e. g. "Duese links aussen"
   // called in addFeedback
   //the geode will be deleted in the destructor
   StarRegion(const char *name, int index, coInteractor *inter, int debug);

   //
   ~StarRegion();
   
   // new execute
   void update(coInteractor *inter);

   // return the name, e. g. "Duese links aussen"
   char *getName();

   int getNumParams(){return numParams;}
   
   // return the covise object Name
   char* getCoviseObjectName();
   
   // set the patches geode for highlighting
   void setPatchesGeode(osg::Geode *geode);
   
   // return the performer geode node
   osg::Geode* getPatchesGeode();
   
   // outlined appearance
   void enableHighLight();
   
   // normal appearance
   void disableHighLight();

   // ...
   void createInteractors();
   void deleteInteractors();
};
#endif
