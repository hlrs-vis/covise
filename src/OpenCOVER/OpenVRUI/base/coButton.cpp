/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/coButton.h>
#include <OpenVRUI/coCombinedButtonInteraction.h>
#include <OpenVRUI/coUIContainer.h>

#include <OpenVRUI/sginterface/vruiButtons.h>
#include <OpenVRUI/sginterface/vruiTransformNode.h>
#include <OpenVRUI/sginterface/vruiRendererInterface.h>

#include <OpenVRUI/sginterface/vruiIntersection.h>
#include <OpenVRUI/sginterface/vruiHit.h>

#include <OpenVRUI/util/vruiLog.h>

#include <string.h>

namespace vrui
{

/** Called whenever a button was pressed.
  This method needs to be overwritten by child classes.
  @param button coButton which triggered this event
*/
#if 0
void coButtonActor::buttonEvent(coButton *)
{
   //if (cover->debugLevel(2))
   VRUILOG("coButtonActor::buttonEvent err: call to base class function")
}
#endif

/** Constructor.
  @param geom  a coButtonGeometry is required to define the button shape
  @param actor the coButtonActor is called whenever the button is pressed
*/
coButton::coButton(coButtonGeometry *geometry, coButtonActor *actor)
{
    myX = 0;
    myY = 0;
    myZ = 0;

    unregister = false;
    myActor = actor;
    myGeometry = geometry;
    selectionState = false;
    pressState = false;
    active_ = true;

    if (myActor != 0)
    {
        vruiIntersection::getIntersectorForAction("coAction")->add(getDCS(), this);
    }

    interactionA = new coCombinedButtonInteraction(coInteraction::ButtonA, "Button", coInteraction::Menu);

    vruiRendererInterface::the()->getUpdateManager()->add(this);
}

/// Destructor. All occurences of this button are removed in its parents.
coButton::~coButton()
{

    vruiIntersection::getIntersectorForAction("coAction")->remove(this);

    vruiNode *myDCS = getDCS();
    myDCS->removeAllParents();
    myDCS->removeAllChildren();

    delete myGeometry;
    delete interactionA;
}

// see superclass for comment
void coButton::setSize(float s)
{
    xScaleFactor = yScaleFactor = zScaleFactor = s / myGeometry->getWidth();
    getDCS()->setScale(xScaleFactor);
}

// see superclass for comment
void coButton::setSize(float xs, float ys, float zs)
{
    xScaleFactor = xs / myGeometry->getWidth();
    yScaleFactor = ys / myGeometry->getHeight();
    zScaleFactor = zs;
    getDCS()->setScale(xScaleFactor, yScaleFactor, zScaleFactor);
}

/** Set button press state.
  @param state button press state
  @param generateEvent if true, onPress and onRelease events are generated
*/
void coButton::setState(bool state, bool generateEvent)
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
bool coButton::getState() const
{
    return pressState;
}

/** Update the geometry switch.
  The switch state depends on both the button's selection state and press state.
  Thus, four switch states need to be considered.
*/
void coButton::updateSwitch()
{

    createGeometry();
    if (!active_)
        myGeometry->switchGeometry(coButtonGeometry::DISABLED);
    else if (!selectionState)
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
void coButton::setPos(float x, float y, float z)
{
    myX = x;
    myY = y;
    myZ = z;
    getDCS()->setTranslation(x, y, z);
}

/** Get button state.
  @return true if button is pressed
*/
bool coButton::isPressed() const
{
    return pressState;
}

/** Hit is called whenever the button with this action is intersected
  by the input device.
  @param hit intersection information
  @return ACTION_CALL_ON_MISS if you want miss to be called,
          otherwise ACTION_DONE is returned
*/
int coButton::hit(vruiHit *hit)
{

    //VRUILOG("coButton::hit info: called")

    vruiRendererInterface *renderer = vruiRendererInterface::the();

    Result preReturn = renderer->hit(this, hit);
    if (preReturn != ACTION_UNDEF)
        return preReturn;

    if (!interactionA->isRegistered())
    {
        coInteractionManager::the()->registerInteraction(interactionA);
        interactionA->setHitByMouse(hit->isMouseHit());
    }

    selectionState = true;

    vruiButtons *buttons = hit->isMouseHit()
                               ? renderer->getMouseButtons()
                               : renderer->getButtons();

    if (buttons->wasPressed() & vruiButtons::ACTION_BUTTON)
    {
        // left Button was pressed
        onPress();
    }
    else if (buttons->wasReleased() & vruiButtons::ACTION_BUTTON)
    {
        // left Button was released
        onRelease();
    }

    updateSwitch();
    return ACTION_CALL_ON_MISS;
}

/// Miss is called once after a hit, if the button is not intersected anymore.
void coButton::miss()
{
    vruiRendererInterface::the()->miss(this);
    selectionState = false;
    updateSwitch();

    unregister = true;
}

bool coButton::update()
{
    if (unregister)
    {
        if (interactionA->isRegistered() && (interactionA->getState() != coInteraction::Active))
        {
            coInteractionManager::the()->unregisterInteraction(interactionA);
            unregister = false;
        }
    }

    return true;
}

/// Overwrite this method for action on button presses.
void coButton::onPress()
{
}

/// Overwrite this method for action on button releases.
void coButton::onRelease()
{
}

const char *coButton::getClassName() const
{
    return "coButton";
}

bool coButton::isOfClassName(const char *classname) const
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

float coButton::getWidth() const
{
    return myGeometry->getWidth() * xScaleFactor;
}

float coButton::getHeight() const
{
    return myGeometry->getHeight() * yScaleFactor;
}

void coButton::createGeometry()
{
    if (myGeometry->getButtonProvider() == 0)
    {
        myGeometry->createGeometry();
        vruiIntersection::getIntersectorForAction("coAction")->add(myGeometry->getDCS(), this);
    }
}

void coButton::resizeGeometry()
{
    createGeometry();
    myGeometry->resizeGeometry();
}

vruiTransformNode *coButton::getDCS()
{

    createGeometry();

    return myGeometry->getDCS();
}

//set if item is active
void coButton::setActive(bool a)
{
    // if item is activated add background to intersector
    if (!active_ && a)
    {
        vruiIntersection::getIntersectorForAction("coAction")->add(getDCS(), this);
        myGeometry->switchGeometry(coButtonGeometry::NORMAL);
    }
    // if item is deactivated remove background from intersector
    else if (active_ && !a)
    {
        vruiIntersection::getIntersectorForAction("coAction")->remove(this);
        myGeometry->switchGeometry(coButtonGeometry::DISABLED);
        selectionState = false;
        pressState = false;
    }
    active_ = a;
}
}
