/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_BUTTON_H
#define CO_BUTTON_H

#include <OpenVRUI/coAction.h>
#include <OpenVRUI/coButtonGeometry.h>
#include <OpenVRUI/coUIElement.h>
#include <OpenVRUI/coUpdateManager.h>

namespace vrui
{

class coButton;
class coCombinedButtonInteraction;

/// Action listener for events triggered by coButton.
class OPENVRUIEXPORT coButtonActor
{
public:
    virtual ~coButtonActor()
    {
    }
    virtual void buttonEvent(coButton *button) = 0;
};

/** Basic button class providing a selectable rectangular area.
  This class is derived from coAction for input device events, and it
  is derived from coUIElement to inherit the basic UI element functionality.
*/
class OPENVRUIEXPORT coButton : public coAction, public coUIElement, public coUpdateable
{
public:
    coButton(coButtonGeometry *geom, coButtonActor *actor);
    virtual ~coButton();

    void setState(bool state, bool generateEvent = false);
    bool getState() const;

    void updateSwitch();
    bool isPressed() const;

    void setPos(float x, float y, float z = 0);

    virtual int hit(vruiHit *hit);
    virtual void miss();

    virtual void onPress();
    virtual void onRelease();
    virtual void setSize(float, float, float);
    virtual void setSize(float);
    virtual float getWidth() const;
    virtual float getHeight() const;

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

    virtual vruiTransformNode *getDCS();

    /// get the Element's classname
    virtual const char *getClassName() const;
    /// check if the Element or any ancestor is this classname
    virtual bool isOfClassName(const char *) const;

    virtual bool update();
    /// functions activates or deactivates the item
    virtual void setActive(bool a);

protected:
    virtual void createGeometry();
    virtual void resizeGeometry();

    bool selectionState; ///< true if button is selected by the user
    bool pressState; ///< true if the button is currently pressed
    coButtonGeometry *myGeometry; ///< information about visual appearance
    coButtonActor *myActor; ///< action listener, triggered on button press
    float myX; ///< button x position
    float myY; ///< button y position
    float myZ; ///< button z position
    coCombinedButtonInteraction *interactionA; ///< button interaction
    bool unregister; ///< try to unregister interactions
    bool active_; ///< flag if button is active
};

/** This class provides a specific coButton to be used as a push button.
  This means it can be pressed and will be released when the pointer
  device releases it.
*/
class OPENVRUIEXPORT coPushButton : public coButton
{
public:
    coPushButton(coButtonGeometry *geom, coButtonActor *actor);
    virtual ~coPushButton();
    virtual void onPress();
    virtual void onRelease();
    virtual void miss();

    /// get the Element's classname
    virtual const char *getClassName() const;
    /// check if the Element or any ancestor is this classname
    virtual bool isOfClassName(const char *) const;
};

/** This class provides a specific coButton to be used as a toggle button.
  This means it can be pressed and will remain pressed when the pointer
  is released. Only another pointer button press and release will restore
  its original state.
*/
class OPENVRUIEXPORT coToggleButton : public coButton
{
public:
    coToggleButton(coButtonGeometry *geom, coButtonActor *actor);
    virtual ~coToggleButton();
    virtual void onPress();
    virtual void onRelease();

    /// get the Element's classname
    virtual const char *getClassName() const;
    /// check if the Element or any ancestor is this classname
    virtual bool isOfClassName(const char *) const;

protected:
    bool wasReleased;
};
}
#endif
