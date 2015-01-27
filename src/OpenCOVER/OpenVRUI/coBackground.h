/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_BACKGROUND_H
#define CO_BACKGROUND_H

#include <OpenVRUI/coUIContainer.h>

namespace vrui
{

class vruiTransformNode;

/** This class provides background for GUI elements.
  A background should contain only one child, use another container to layout
  multiple chlidren inside the frame.
*/
class OPENVRUIEXPORT coBackground : public coUIContainer
{
public:
    coBackground();
    virtual ~coBackground();
    virtual void addElement(coUIElement *element);
    virtual void removeElement(coUIElement *element);
    virtual float getWidth() const;
    virtual float getHeight() const;
    virtual float getDepth() const;
    virtual float getXpos() const;
    virtual float getYpos() const;
    virtual float getZpos() const;
    virtual void setPos(float x, float y, float z = 0);
    virtual void setWidth(float width);
    virtual void setHeight(float height);
    virtual void setDepth(float depth);
    virtual void setMinWidth(float minWidth);
    virtual void setMinHeight(float minHeight);
    virtual void setMinDepth(float minDepth);
    virtual void setZOffset(float offset);
    virtual void setSize(float size);
    virtual void setSize(float nw, float nh, float nd);
    virtual void setEnabled(bool enabled);
    virtual void setHighlighted(bool highlighted);
    virtual void resizeToParent(float, float, float, bool shrink = true);
    virtual void shrinkToMin();

    /// get the Element's classname
    virtual const char *getClassName() const;
    /// check if the Element or any ancestor is this classname
    virtual bool isOfClassName(const char *) const;

protected:
    virtual void resizeGeometry();

    //void resize();                              ///< resize the background to fit around the child
    void realign(); ///< centers the child
    float myX; ///< Background position X
    float myY; ///< Background position Y
    float myZ; ///< Background position Z
    float myWidth; ///< Background width
    float myHeight; ///< Background height
    float myDepth; ///< Background depth
    float minWidth; ///< Background minimal width
    float minHeight; ///< Background minimal height
    float minDepth; ///< Background minimal depth
    float myZOffset; ///< Z offset of child
};
}
#endif
