/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_VOL_ROTBUTTON_H
#define CO_VOL_ROTBUTTON_H

#include <OpenVRUI/coAction.h>
#include <OpenVRUI/coButtonGeometry.h>
#include <OpenVRUI/coUIElement.h>

namespace vrui
{

class coRotButton;

class vruiTransformNode;

/// Action listener for events triggered by coButton.
class OPENVRUIEXPORT coRotButtonActor
{
public:
    virtual ~coRotButtonActor(){};
    virtual void buttonEvent(coRotButton *button) = 0;
};

/** Basic button class providing a selectable rectangular area.
  This class is derived from coAction for input device events, and it
  is derived from coUIElement to inherit the basic UI element functionality.
  Additionally it offers the chance to rotate the content.
*/
class OPENVRUIEXPORT coRotButton : public coAction, public coUIElement
{
public:
    coRotButton(coButtonGeometry *geometry, coRotButtonActor *actor);
    virtual ~coRotButton();

    void setState(bool state, bool generateEvent = false);
    bool getState() const;

    void updateSwitch();
    bool isPressed() const;

    virtual int hit(vruiHit *hit);
    virtual void miss();

    virtual void onPress();
    virtual void onRelease();

    void setPos(float x, float y, float z = 0.0f);
    virtual void setSize(float x, float y, float z);
    virtual void setSize(float size);

    virtual void setRotation(float rotation);

    /// functions activates or deactivates the item
    virtual void setActive(bool a);

    virtual float getWidth() const
    {
        return myGeometry->getWidth() * xScaleFactor;
    }
    virtual float getHeight() const
    {
        return myGeometry->getHeight() * yScaleFactor;
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

    virtual void createGeometry();
    virtual void resizeGeometry();

    virtual vruiTransformNode *getDCS();

    /// get the Element's classname
    virtual const char *getClassName() const;
    /// check if the Element or any ancestor is this classname
    virtual bool isOfClassName(const char *) const;

protected:
    bool selectionState; ///< true if button is selected by the user
    bool pressState; ///< true if the button is currently pressed
    bool wasReleased;
    coButtonGeometry *myGeometry; ///< information about visual appearance
    coRotButtonActor *myActor; ///< action listener, triggered on button press
    float myX; ///< button x position
    float myY; ///< button y position
    float myZ; ///< button z position

    vruiTransformNode *myDCS;
    vruiTransformNode *rotationDCS;
    float rotation; ///< rotation in degrees off ??? orientation
    bool active_; ///< flag if button is active
};

/** This class provides a specific coRotButton to be used as a push button.
  This means it can be pressed and will be released when the pointer
  device releases it.
*/
class coRotPushButton : public coRotButton
{
public:
    coRotPushButton(coButtonGeometry *geometry, coRotButtonActor *actor);
    virtual ~coRotPushButton();
    virtual void onPress();
    virtual void onRelease();
    virtual void miss();

    /// get the Element's classname
    virtual const char *getClassName() const;
    /// check if the Element or any ancestor is this classname
    virtual bool isOfClassName(const char *) const;
};

/** This class provides a specific coRotButton to be used as a toggle button.
  This means it can be pressed and will remain pressed when the pointer
  is released. Only another pointer button press and release will restore
  its original state.
*/
class coRotToggleButton : public coRotButton
{
public:
    coRotToggleButton(coButtonGeometry *geometry, coRotButtonActor *actor);
    virtual ~coRotToggleButton();
    virtual void onPress();

    /// get the Element's classname
    virtual const char *getClassName() const;
    /// check if the Element or any ancestor is this classname
    virtual bool isOfClassName(const char *) const;
};
}
#endif
