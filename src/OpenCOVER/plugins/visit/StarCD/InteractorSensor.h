#ifndef _INTERACTOR_SENSOR_H
#define _INTERACTOR_SENSOR_H

#include <cover/coVRPluginSupport.h>
using namespace covise;
using namespace opencover;
using namespace vrui;


#include "StarRegionInteractor.h"

#include <OpenVRUI/coTrackerButtonInteraction.h>
#include <PluginUtil/coSensor.h>


namespace osg
{
   class Node;
}

class InteractorSensor : public coTrackerButtonInteraction, public coPickSensor
{
 private:
   StarRegionInteractor *interactor;
public:
   InteractorSensor(osg::Node * node, StarRegionInteractor *i);
   ~InteractorSensor();
   void activate();
   void disactivate();
};

#endif
