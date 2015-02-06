/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/coRotButton.h>
#include <OpenVRUI/coUIContainer.h>

#include <OpenVRUI/sginterface/vruiButtons.h>
#include <OpenVRUI/sginterface/vruiHit.h>
#include <OpenVRUI/sginterface/vruiIntersection.h>
#include <OpenVRUI/sginterface/vruiTransformNode.h>
#include <OpenVRUI/sginterface/vruiRendererInterface.h>

#include <OpenVRUI/util/vruiLog.h>

namespace vrui
{

/** Called whenever a button was pressed.
  This method needs to be overwritten by child classes.
  @param button coRotButton which triggered this event
*/
void coRotButtonActor::buttonEvent(coRotButton *)
{
    //  if (cover->debugLevel(2))
    VRUILOG("call to base class function: coRotButtonActor::buttonEvent(coRotButton *)")
}

/** Constructor.
  @param geom  a coButtonGeometry is required to define the button shape
  @param actor the coRotButtonActor is called whenever the button is pressed
*/
coRotButton::coRotButton(coButtonGeometry *geometry, coRotButtonActor *actor)
{
    myDCS = vruiRendererInterface::the()->createTransformNode();
    myDCS->setName("coRotButton");

    rotationDCS = vruiRendererInterface::the()->createTransformNode();
    rotation = 0.0;
    rotationDCS->setRotation(0.0f, 0.0f, 1.0f, rotation);

    myActor = actor;
    myGeometry = geometry;

    vruiTransformNode *centerDCS = geometry->getDCS();

    // link object tree
    myDCS->addChild(rotationDCS);
    rotationDCS->addChild(centerDCS);

    // center icon before rotation
    centerDCS->setTranslation(-myGeometry->getWidth() / 2.0f,
                              -myGeometry->getHeight() / 2.0f,
                              0.0f);

    // move icon origin to lower left corner after rotation
    rotationDCS->setTranslation(myGeometry->getWidth() / 2.0f,
                                myGeometry->getHeight() / 2.0f,
                                0.0f);

    selectionState = false;
    pressState = false;
    wasReleased = false;
    updateSwitch();

    if (myActor != 0)
    {
        vruiIntersection::getIntersectorForAction("coAction")->add(myDCS, this);
    }

    myX = 0.0f;
    myY = 0.0f;
    myZ = 0.0f;

    active_ = true;
}

/// Destructor. All occurences of this button are removed in its parents.
coRotButton::~coRotButton()
{

    vruiIntersection::getIntersectorForAction("coAction")->remove(this);

    rotationDCS->removeAllChildren();

    myDCS->removeAllParents();
    myDCS->removeAllChildren();

    vruiRendererInterface::the()->deleteNode(myDCS);
    vruiRendererInterface::the()->deleteNode(rotationDCS);

    delete myGeometry;
}

// see superclass for comment
void coRotButton::setSize(float s)
{
    xScaleFactor = yScaleFactor = zScaleFactor = s / myGeometry->getWidth();
    getDCS()->setScale(xScaleFactor);
}

// see superclass for comment
void coRotButton::setSize(float xs, float ys, float zs)
{
    xScaleFactor = xs / myGeometry->getWidth();
    yScaleFactor = ys / myGeometry->getHeight();
    zScaleFactor = zs;
    getDCS()->setScale(xScaleFactor, yScaleFactor, zScaleFactor);
}

void coRotButton::setRotation(float newrot)
{
    rotation = newrot;
    // order in DCS always is: Rotate then Translate
    // rotate Icon
    rotationDCS->setRotation(0.0f, 0.0f, 1.0f, rotation);
}

/** Set button press state.
  @param state button press state
  @param generateEvent if true, onPress and onRelease events are generated
*/
void coRotButton::setState(bool state, bool generateEvent)
{
    if (generateEvent)
        onPress();
    pressState = state;
    updateSwitch();
    if (generateEvent)
        onRelease();
}

/** Get button press state.
  @return button press state
*/
bool coRotButton::getState() const
{
    return pressState;
}

//set if item is active
void coRotButton::setActive(bool a)
{
    // if item is activated add background to intersector
    if (!active_ && a)
    {
        vruiIntersection::getIntersectorForAction("coAction")->add(myDCS, this);
        active_ = a;
    }
    // if item is deactivated remove background from intersector
    else if (active_ && !a)
    {
        vruiIntersection::getIntersectorForAction("coAction")->remove(this);
        active_ = a;
        myGeometry->switchGeometry(coButtonGeometry::DISABLED);
    }
}

/** Update the geometry switch.
  The switch state depends on both the button's selection state and press state.
  Thus, four switch states need to be considered.
*/
void coRotButton::updateSwitch()
{

    if (!selectionState)
    {
        if (pressState)
            myGeometry->switchGeometry(coButtonGeometry::PRESSED);
        else
            myGeometry->switchGeometry(coButtonGeometry::NORMAL);
    }
    else
    {
        if (pressState)
            myGeometry->switchGeometry(coButtonGeometry::HIGHLIGHT_PRESSED);
        else
            myGeometry->switchGeometry(coButtonGeometry::HIGHLIGHT);
    }
}

// see superclass for comment
void coRotButton::setPos(float x, float y, float z)
{
    myX = x;
    myY = y;
    myZ = z;
    getDCS()->setTranslation(x, y, z);
}

/** Get button state.
  @return true if button is pressed
*/
bool coRotButton::isPressed() const
{
    return pressState;
}

/** Hit is called whenever the button with this action is intersected
  by the input device.
  @param hitPoint,hit  Performer intersection information
  @return ACTION_CALL_ON_MISS if you want miss to be called,
          otherwise ACTION_DONE is returned
*/
int coRotButton::hit(vruiHit *hit)
{

    //VRUILOG("coRotButton::hit info: called")

    vruiRendererInterface *renderer = vruiRendererInterface::the();

    Result preReturn = renderer->hit(this, hit);
    if (preReturn != ACTION_UNDEF)
        return preReturn;

    vruiButtons *buttons = hit->isMouseHit() ? renderer->getMouseButtons() : renderer->getButtons();

    selectionState = true;
    if (buttons->wasPressed(vruiButtons::ACTION_BUTTON))
    {
        // left Button was pressed
        onPress();
    }
    else if (buttons->wasReleased(vruiButtons::ACTION_BUTTON))
    {
        // left Button was released
        onRelease();
    }

    updateSwitch();
    return ACTION_CALL_ON_MISS;
}

/// Miss is called once after a hit, if the button is not intersected anymore.
void coRotButton::miss()
{
    vruiRendererInterface::the()->miss(this);
    selectionState = false;
    updateSwitch();
}

/// Overwrite this method for action on button presses.
void coRotButton::onPress()
{
    if (!wasReleased)
        return;
    wasReleased = false;
}

/// Overwrite this method for action on button releases.
void coRotButton::onRelease()
{
    if (!wasReleased)
    {
        pressState = !pressState;
        myActor->buttonEvent(this);
    }
    wasReleased = true;
}

void coRotButton::createGeometry()
{
    myGeometry->createGeometry();
}

void coRotButton::resizeGeometry()
{
    createGeometry();
    myGeometry->resizeGeometry();
}

vruiTransformNode *coRotButton::getDCS()
{
    return myDCS;
}

const char *coRotButton::getClassName() const
{
    return "coRotButton";
}

bool coRotButton::isOfClassName(const char *classname) const
{
    // paranoia makes us mistrust the string library and check for NULL.
    if (classname && getClassName())
    {
        // check for identity
        if (!strcmp(classname, getClassName()))
        { // we are the one
            return true;
        }
        else
        { // we are not the wanted one. Branch up to parent class
            return coUIElement::isOfClassName(classname);
        }
    }

    // nobody is NULL
    return false;
}
}
