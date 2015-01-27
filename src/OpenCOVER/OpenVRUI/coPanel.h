/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_PANEL_H
#define CO_PANEL_H

#include <OpenVRUI/coAction.h>
#include <OpenVRUI/coUIContainer.h>

namespace vrui
{

class coPanelGeometry;

class vruiTransformNode;

/** This class provides background panel for GUI elements.
 It can contain mutiple GUI elements, it gows to accommodate
 all children but does not automatically shrink to fit,
 if a child is removed.
 This class does not provide any geometry, the actual
 geometry is defined in a another class, @see coPanelGeometry
*/
class OPENVRUIEXPORT coPanel : public coAction, public coUIContainer
{
public:
    coPanel(coPanelGeometry *geom);
    virtual ~coPanel();

    // hit is called whenever the button
    // with this action is intersected
    // return ACTION_CALL_ON_MISS if you want miss to be called
    // otherwise return ACTION_DONE
    virtual int hit(vruiHit *hit);

    // miss is called once after a hit, if the button is not intersected
    // anymore
    virtual void miss();

    void resize();
    virtual void addElement(coUIElement *element);
    void hide(coUIElement *element);
    void show(coUIElement *element);
    virtual void showElement(coUIElement *element);

    void setPos(float x, float y, float z = 0.0f);
    virtual float getWidth() const
    {
        return myWidth * scale;
    }
    virtual float getHeight() const
    {
        return myHeight * scale;
    }
    virtual float getXpos() const
    {
        return myX;
    }
    virtual float getYpos() const
    {
        return myY;
    }
    virtual float getZpos() const
    {
        return myZ;
    }

    virtual void setScale(float s);
    virtual vruiTransformNode *getDCS();

    /// get the Element's classname
    virtual const char *getClassName() const;
    /// check if the Element or any ancestor is this classname
    virtual bool isOfClassName(const char *) const;

protected:
    vruiTransformNode *myDCS;
    vruiTransformNode *myPosDCS; ///< Transformation of the panel geometry
    vruiTransformNode *myChildDCS; ///< Children origin
    float scale; ///< scale factor, scales all children
    float myX; ///< Panel position X
    float myY; ///< Panel position Y
    float myZ; ///< Panel position Z
    float myWidth; ///< Panel width
    float myHeight; ///< Panel height
    float contentWidth; ///< Content width
    float contentHeight; ///< Content height
    coPanelGeometry *myGeometry; ///< Panel geometry
};
}
#endif
