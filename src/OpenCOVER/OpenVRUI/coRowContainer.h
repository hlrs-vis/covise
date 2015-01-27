/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_ROW_CONTAINER_H
#define CO_ROW_CONTAINER_H
#include <OpenVRUI/coUIContainer.h>

namespace vrui
{

class vruiTransformNode;

/// Container class that aligns its children in a row
class OPENVRUIEXPORT coRowContainer : public coUIContainer
{
public:
    /// orientation of this container
    enum Orientation
    {
        HORIZONTAL = 0,
        VERTICAL
    };

    coRowContainer(Orientation orientation = HORIZONTAL);
    virtual ~coRowContainer();

    virtual void addElement(coUIElement *element);
    virtual void removeElement(coUIElement *element);
    virtual void insertElement(coUIElement *element, int pos);
    virtual void resizeToParent(float, float, float, bool shrink = true);
    virtual void shrinkToMin();
    void hide(coUIElement *element);
    void show(coUIElement *element);

    void setPos(float x, float y, float z = 0.0f);
    void setOrientation(Orientation orientation);
    int getOrientation() const;
    void setAlignment(int alignment);
    void setHgap(float g);
    void setVgap(float g);
    void setDgap(float g);

    vruiTransformNode *getDCS();

    virtual float getWidth() const
    {
        return myWidth;
    }
    virtual float getHeight() const
    {
        return myHeight;
    }
    virtual float getDepth() const
    {
        return myDepth;
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

    virtual float getVgap() const;
    virtual float getHgap() const;
    virtual float getDgap() const;

    virtual void setAttachment(int attachment);
    virtual int getAttachment() const;

    /// get the Element's classname
    virtual const char *getClassName() const;
    /// check if the Element or any ancestor is this classname
    virtual bool isOfClassName(const char *) const;

protected:
    //virtual void resize();
    /// transformation node to position this container
    vruiTransformNode *myDCS;
    /// position and size
    float myX;
    float myY;
    float myZ;
    float myHeight;
    float myWidth;
    float myDepth;
    /// layout orientation
    Orientation orientation;
    /// alignment of the children
    int alignment;
    /// Horizontal-, vertical- and depth-gap, default is 5mm for H and Vgap, 0 for Dgap
    float Hgap;
    float Vgap;
    float Dgap;

    int attachment;

    /// get my current extension in given direction
    float getExtW() const;
    float getExtH() const;
};
}
#endif
