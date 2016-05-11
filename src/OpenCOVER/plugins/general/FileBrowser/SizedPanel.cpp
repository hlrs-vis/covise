#include <OpenVRUI/coPanel.h>
#include <OpenVRUI/coPanelGeometry.h>
#include <OpenVRUI/coLabel.h>

#include <OpenVRUI/sginterface/vruiHit.h>
#include <OpenVRUI/sginterface/vruiIntersection.h>
#include <OpenVRUI/sginterface/vruiRendererInterface.h>
#include <OpenVRUI/sginterface/vruiTransformNode.h>

#include <OpenVRUI/util/vruiLog.h>

#include "SizedPanel.h"

using namespace std;

#define BORDERWIDTH 0.5f

SizedPanel::SizedPanel(coPanelGeometry * geometry) : coPanel(geometry)
{
    scale = 1.0;
    sWidth = 200;
    sHeight = 200;
}

SizedPanel::~SizedPanel()
{

}

void SizedPanel::setScale(float s)
{
   scale = s;
   myChildDCS->setScale(scale, scale, scale);
}

void SizedPanel::removeElement(coUIElement * el)
{
   coUIContainer::removeElement(el);
   if(el->getDCS())
   {
      myChildDCS->removeChild(el->getDCS());
   }
}

void SizedPanel::resize()
{
   float maxX = -100000;
   float maxY = -100000;
   float minX = 100000;
   float minY = 100000;
   //float minX = 0.0;
   //float minY = 0.0;
   float minZ = 100000;

   float xOff, yOff, zOff = 0;


   for (list<coUIElement *>::iterator i = elements.begin(); i != elements.end(); ++i)
   {
      if(maxX < (*i)->getXpos() + (*i)->getWidth())
         maxX = (*i)->getXpos() + (*i)->getWidth();

      if(maxY < (*i)->getYpos() + (*i)->getHeight() )
         maxY = (*i)->getYpos() + (*i)->getHeight();

      if(minX > (*i)->getXpos())
         minX = (*i)->getXpos();

      if(minY > (*i)->getYpos())
         minY = (*i)->getYpos();

      if(minZ > (*i)->getZpos())
         minZ = (*i)->getZpos();
   }



   //if(sHeight > 0)
   //{
   myHeight = sHeight + (float)(2*BORDERWIDTH);
   //}
   //else
   //{
      //myHeight = (maxY - minY) + (float)(2*BORDERWIDTH);
   //}
   xOff = minX - (float)BORDERWIDTH;
   yOff = minY - (float)BORDERWIDTH;

   if(myGeometry)
      zOff = minZ - myGeometry->getDepth();

   //if(sWidth > 0)
   //{
   myWidth = sWidth + (float)(2*BORDERWIDTH);
   //}
   //else
   //{
     // myWidth = (maxX - minX) + (float)(2*BORDERWIDTH);
   //}

   //myChildDCS->setTranslation(-xOff * scale, -yOff * scale, -zOff * scale);
   myChildDCS->setTranslation(-xOff * scale, -yOff * scale, 0.0);
   myChildDCS->setScale(scale, scale, scale);


   if(myGeometry)
   {

      myPosDCS->setScale(getWidth()/myGeometry->getWidth(),getHeight()/myGeometry->getHeight(),1.0);
   }

    if(getParent())
        getParent()->childResized();

}


void SizedPanel::setPos(float x, float y, float )
{
   resize();
   myX = x;
   myY = y;
   myDCS->setTranslation(myX, myY, myZ);
}

int SizedPanel::hit(vruiHit * hit)
{

   //VRUILOG("coPanel::hit info: called")

   Result preReturn = vruiRendererInterface::the()->hit(this, hit);
   if (preReturn != ACTION_UNDEF) return preReturn;

   return ACTION_CALL_ON_MISS;

}


/**miss is called once after a hit, if the panel is not intersected
 anymore*/
void SizedPanel::miss()
{
   vruiRendererInterface::the()->miss(this);
}


const char * SizedPanel::getClassName() const
{
   return "SizedPanel";
}


bool SizedPanel::isOfClassName(const char * classname) const
{
   // paranoia makes us mistrust the string library and check for NULL.
   if(classname && getClassName())
   {
      // check for identity
      if( !strcmp( classname, getClassName() ) )
      {                                           // we are the one
         return true;
      }
      else
      {                                           // we are not the wanted one. Branch up to parent class
         return coPanel::isOfClassName(classname);
      }
   }

   // nobody is NULL
   return false;
}

vruiTransformNode * SizedPanel::getDCS()
{
   return myDCS;
}

void SizedPanel::setSize(float w, float h)
{
   sWidth = w;
   sHeight = h;

   this->resize();
}

