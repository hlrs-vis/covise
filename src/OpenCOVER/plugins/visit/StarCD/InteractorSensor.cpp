#include "InteractorSensor.h"

#include <OpenVRUI/coInteractionManager.h>

#include <osg/Node>

InteractorSensor::InteractorSensor(osg::Node * node, StarRegionInteractor *i)
   : coTrackerButtonInteraction(coInteraction::ButtonA, "StarCD") , coPickSensor(node)
{
   interactor = i;
   //fprintf(stderr,"InteractorSensor::InteractorSensor for %s\n", interactor->getParamName());

}


InteractorSensor::~InteractorSensor()
{
}


void InteractorSensor::activate()
{
   if (enabled)
   {
      coInteractionManager::the()->registerInteraction(this);
      //fprintf(stderr,"InteractorSensor::activate %s\n", interactor->getParamName());
      //interactor->setIntersectedHighLight();
   }
}


void InteractorSensor::disactivate()
{
   coInteractionManager::the()->unregisterInteraction(this);
   if (enabled)
   {
      //fprintf(stderr,"InteractorSensor::disactivate %s\n",  interactor->getParamName());
//       if (!interactor->isSelected())
//          interactor->setNormalHighLight();
   }
}
