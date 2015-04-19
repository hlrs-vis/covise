#include <OpenVRUI/osg/OSGVruiTransformNode.h>
#include <OpenVRUI/coPanel.h>

namespace vrui {
class coPanelGeometry;
class vruiTransformNode;
}

using namespace vrui;

class SizedPanel: public coPanel
{
    public:
       SizedPanel(vrui::coPanelGeometry * geometry);
       virtual ~SizedPanel();
       
       virtual int hit(vruiHit * hit);
       virtual void miss();
 
       void resize();

       virtual void removeElement(coUIElement * element);

       void setPos(float x, float y, float z = 0.0f);
       virtual float getWidth()  { return myWidth  * scale; }
       virtual float getHeight() { return myHeight * scale; }
       virtual float getXpos()   const { return myX; }
       virtual float getYpos()   const { return myY; }
       virtual float getZpos()   const { return myZ; }
       //virtual void setHeight(float h)   { sHeight = h; } 
       virtual void setSize(float w, float h);

       virtual void setScale(float s);
       virtual vrui::vruiTransformNode * getDCS();

       /// get the Element's classname
       virtual const char * getClassName() const;
       /// check if the Element or any ancestor is this classname
       virtual bool isOfClassName(const char *) const;

    protected:
       float sWidth, sHeight;
};
