/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cstdlib>
#include <OpenVRUI/coValuePoti.h>

#include <OpenVRUI/coCombinedButtonInteraction.h>

#include <OpenVRUI/sginterface/vruiIntersection.h>
#include <OpenVRUI/sginterface/vruiRendererInterface.h>
#include <OpenVRUI/sginterface/vruiTransformNode.h>
#include <OpenVRUI/sginterface/vruiUIElementProvider.h>
#include <OpenVRUI/sginterface/vruiHit.h>

#include <OpenVRUI/util/vruiDefines.h>

using namespace std;

namespace vrui
{

void coValuePotiActor::potiPressed(coValuePoti *, int)
{
}

void coValuePotiActor::potiReleased(coValuePoti *, int)
{
}

/** Constructor.
  @param buttonText           button text
  @param actor                action listener for poti value changes
  @param backgroundTexture    background texture file name
  @param cInterfaceManager    collaborative interface manager (optional)
  @param interfaceName        unique ID string for collaboration (optional)
*/
coValuePoti::coValuePoti(const string &buttonText, coValuePotiActor *actor,
                         const string &backgroundTexture, vruiCOIM *cInterfaceManager,
                         const string &interfaceName)
    : vruiCollabInterface(cInterfaceManager, interfaceName, vruiCollabInterface::VALUEPOTI)
{

    unregister = false;
    integer = false;
    discrete = false;
    labelVisible = true;
    increment = 0.0f;
    minValue = 0;
    maxValue = 1;
    lastRoll = 0.0f;
    myActor = actor;

    this->baseButtonText = buttonText;
    this->backgroundTexture = backgroundTexture;

    setValue(0.0f);
    setPos(0.0f, 0.0f);
    setSize(0.4f);

    string nodeName = "coValuePoti(" + buttonText + ")";
    getDCS()->setName(nodeName);

    vruiIntersection::getIntersectorForAction("coAction")->add(getDCS(), this);

    vruiRendererInterface::the()->getUpdateManager()->add(this);

    interactionA = new coCombinedButtonInteraction(coInteraction::ButtonA, "Poti", coInteraction::Menu);
    interactionB = new coCombinedButtonInteraction(coInteraction::ButtonC, "Poti", coInteraction::Menu);
    interactionW = new coCombinedButtonInteraction(coInteraction::WheelVertical, "Poti", coInteraction::Menu);
}

/// Destructor.
coValuePoti::~coValuePoti()
{

    vruiTransformNode *myDCS = getDCS();

    myDCS->removeAllParents();
    myDCS->removeAllChildren();

    vruiIntersection::getIntersectorForAction("coAction")->remove(this);
    // can´t do this here, or we have to set it to 0 as well whereever it is declared because it is otherwise reused later.  in ~coUIElement -> parent->removeElement(this)... vruiRendererInterface::the()->deleteNode(myDCS);

    delete interactionA;
    delete interactionB;
    delete interactionW;
}

/** Set poti location.
  @param x,y,z location of top left corner of poti element
*/
void coValuePoti::setPos(float x, float y, float)
{

    myX = x;
    myY = y;
    getDCS()->setTranslation(myX + (xScaleFactor * 30.0f), myY + (yScaleFactor * 53.0f / 2.0f), 0.0f);
}

/** Set poti size.
  @param s factor relative to original poti VRML file size
*/
void coValuePoti::setSize(float s)
{
    xScaleFactor = yScaleFactor = zScaleFactor = s;
    getDCS()->setScale(s);
    getDCS()->setTranslation(myX + (xScaleFactor * 30.0f), myY + (yScaleFactor * 53.0f / 2.0f), 0.0f);
}

/// Set different sizes for each dimension (see setSize)
void coValuePoti::setSize(float xs, float ys, float zs)
{
    xScaleFactor = xs;
    yScaleFactor = ys;
    zScaleFactor = zs;
    getDCS()->setScale(xs, ys, zs);
    getDCS()->setTranslation(myX + (xScaleFactor * 30.0f), myY + (yScaleFactor * 53.0f / 2.0f), 0.0f);
}

/** Disable poti. This is needed when the element is used and should not
  be available for remote collaborators any more.
  @param message message string passed to collaborators
*/
void coValuePoti::remoteLock(const char *message)
{
    setEnabled(false);
    vruiCollabInterface::remoteLock(message);
    myActor->potiPressed(this, remoteContext);
}

/** Inform collaborators about value changes.
  @param message message string passed to collaborators
*/
void coValuePoti::remoteOngoing(const char *message)
{
    float val = (float)strtod(message, 0);
    float oldValue = value;
    setValue(val);
    // trigger "changed" event
    myActor->potiValueChanged(oldValue, val, this, remoteContext);
}

/** Re-enable poti. This is needed when remote collaborators may continue using the element.
  @param message message string passed to collaborators
*/
void coValuePoti::releaseRemoteLock(const char *message)
{
    myActor->potiReleased(this, remoteContext);
    setEnabled(true);
    vruiCollabInterface::releaseRemoteLock(message);
}

/** Set poti value.
  @param val new poti value [0..1]
*/
void coValuePoti::setValue(float val)
{
    val = coClamp(val, minValue, maxValue);

    value = val;
    displayValue(value);

    if (uiElementProvider)
        uiElementProvider->update();
}

/** Set number type for poti value.
  @param i true = integer, false = floating point
*/
void coValuePoti::setInteger(bool on)
{
    integer = on;
    setValue(value);
}

/** Set increment for discreet poti.
  @param increment increment
*/
void coValuePoti::setIncrement(float increment)
{
    if (increment != 0.0f)
        discrete = true;
    else
        discrete = false;

    this->increment = increment;
    setValue(value);
}

/** Set minimum poti value.
  @param m minimum value
*/
void coValuePoti::setMin(float min)
{
    if (discrete)
        minValue = discreteValue(min);
    else
        minValue = min;

    setValue(value);
}

/** Set maximum poti value.
  @param m maximum value
*/
void coValuePoti::setMax(float max)
{
    if (discrete)
        maxValue = discreteValue(max);
    else
        maxValue = max;

    setValue(value);
}

/** Set maximum poti value.
  @param m maximum value
  @param m minimum value
  @param inc increment
*/
void coValuePoti::setState(float min, float max, float val, bool isInt, float inc)
{

    if (inc == 0.0f)
    {
        discrete = false;
    }
    else
    {
        discrete = true;
        increment = inc;
    }

    minValue = min;
    maxValue = max;
    integer = isInt;
    setValue(val);
}

/** Set the poti value to actually display.
  @param value poti value to display
*/
void coValuePoti::displayValue(float value)
{

    char *description;
    if (baseButtonText.length())
    {
        description = new char[baseButtonText.length() + 64];
        if (integer)
        {
            sprintf(description, "%s: %d", baseButtonText.c_str(), (int)value);
        }
        else if (discrete)
        {
            sprintf(description, "%s: %-6.2f", baseButtonText.c_str(), discreteValue(value));
        }
        else
        {
            sprintf(description, "%s: %-6.1f", baseButtonText.c_str(), value);
        }
    }
    else
    {
        description = new char[64];
        if (integer)
        {
            sprintf(description, "%d", (int)value);
        }
        else if (discrete)
        {
            sprintf(description, "%-6.2f", discreteValue(value));
        }
        else
        {
            sprintf(description, "%-6.1f", value);
        }
    }
    //VRUILOG(description);
    setText(description);
    delete[] description;
}

/** Set button text.
  @param bt button text string
*/
void coValuePoti::setText(const string &bt)
{
    buttonText = bt;
    //VRUILOG("coValuePoti::setText info: " << buttonText)
    if (uiElementProvider)
        uiElementProvider->update();
}

/** Get poti value.
  @return current poti value
*/
float coValuePoti::getValue() const
{
    return value;
}

float coValuePoti::getXSize() const
{
    return xScaleFactor;
}

float coValuePoti::getYSize() const
{
    return yScaleFactor;
}

float coValuePoti::getZSize() const
{
    return zScaleFactor;
}

/** Update poti value depending on user input.
  @return true
*/
bool coValuePoti::update()
{

    if (/*FIXME ((cover->getSyncMode()==MS_COUPLING)&&(!cover->isMaster()))||*/ enabled == false)
        return true;

    float oldValue, newValue;
    oldValue = newValue = value;

    if (interactionW->isRunning() || interactionW->wasStarted())
    {
        if (maxValue - minValue > 0.0)
            newValue += interactionW->getWheelCount() * (maxValue - minValue) / 20.0f;

        newValue = coClamp(newValue, minValue, maxValue);
    }
    else if (interactionA->isRunning() || interactionB->isRunning())
    {
        vruiCoord mouseCoord = interactionA->getHandMatrix();

        float mouseRotation;
        float rotScale;

        if (interactionA->is2D())
        {
            mouseRotation = (float)mouseCoord.hpr[1];
            rotScale = 10.0f;
        }
        else
        {
            mouseRotation = (float)mouseCoord.hpr[2];
            rotScale = 1.0f;
        }

        if (interactionB->isRunning())
        {
            rotScale /= 10.0f;
        }

        if (lastRoll != mouseRotation) // did roll value change?
        {
            while (lastRoll - mouseRotation > 180.f)
                lastRoll -= 360.f;
            while (lastRoll - mouseRotation < -180.0f)
                lastRoll += 360.0f;

            newValue -= rotScale * ((lastRoll - mouseRotation) / 180.0f) * (maxValue - minValue);
            lastRoll = mouseRotation;

            newValue = coClamp(newValue, minValue, maxValue);
        }
    }

    if (oldValue != newValue)
    {
        setValue(newValue);
        // trigger "changed" event
        myActor->potiValueChanged(oldValue, newValue, this);

        static char num[100];
        sprintf(num, "%f", newValue);
        sendOngoingMessage(num);
    }

    if (interactionA->wasStopped() || interactionB->wasStopped())
    {
        sendReleaseMessage("");
        if (myActor)
            myActor->potiReleased(this, remoteContext);
    }

    if (unregister)
    {
        if (interactionA->isRegistered() && (interactionA->getState() != coInteraction::Active))
        {
            coInteractionManager::the()->unregisterInteraction(interactionA);
        }
        if (interactionB->isRegistered() && (interactionB->getState() != coInteraction::Active))
        {
            coInteractionManager::the()->unregisterInteraction(interactionB);
        }
        if (interactionW->isRegistered() && (interactionW->getState() != coInteraction::Active))
        {
            coInteractionManager::the()->unregisterInteraction(interactionW);
        }
        if ((!interactionA->isRegistered()) && (!interactionB->isRegistered()) && (!interactionW->isRegistered()))
        {
            unregister = false;
        }
    }

    return true;
}

/** Set the poti label visibility
*/
void coValuePoti::setLabelVisible(bool visible)
{
    labelVisible = visible;
    if (uiElementProvider)
        uiElementProvider->update();
}

/** This method checks if the label is visible
*/
bool coValuePoti::isLabelVisible()
{
    return labelVisible;
}

/** This method is called on intersections of the input device with the
  poti .
  @return ACTION_CALL_ON_MISS
*/
int coValuePoti::hit(vruiHit *hit)
{

    //VRUILOG("coValuePoti::hit info: called")

    Result preReturn = vruiRendererInterface::the()->hit(this, hit);
    if (preReturn != ACTION_UNDEF)
        return preReturn;

    if (enabled == false)
        return ACTION_DONE;

    setHighlighted(true);
    if (!interactionA->isRegistered())
    {
        coInteractionManager::the()->registerInteraction(interactionA);
        interactionA->setHitByMouse(hit->isMouseHit());
    }
    if (!interactionB->isRegistered())
    {
        coInteractionManager::the()->registerInteraction(interactionB);
        interactionB->setHitByMouse(hit->isMouseHit());
    }
    if (!interactionW->isRegistered())
    {
        coInteractionManager::the()->registerInteraction(interactionW);
        interactionW->setHitByMouse(hit->isMouseHit());
    }

    if (interactionA->wasStarted() || interactionB->wasStarted())
    {
        static char context[100];
        sprintf(context, "%d", myActor->getContext());
        sendLockMessage(context);
        if (myActor)
            myActor->potiPressed(this, remoteContext);

        vruiCoord mouseCoord = interactionA->getHandMatrix();
        if (interactionA->is2D())
            lastRoll = (float)mouseCoord.hpr[1];
        else
            lastRoll = (float)mouseCoord.hpr[2];
    }

    return ACTION_CALL_ON_MISS;
}

/// Called when input device leaves the element.
void coValuePoti::miss()
{
    setHighlighted(false);
    unregister = true;
}

void coValuePoti::joystickUp()
{
    float val = value;
    if (integer)
        val++;
    else
        val += (maxValue - minValue) / 100.0f;
    val = coClamp(val, minValue, maxValue);

    value = val;
    displayValue(value);
}

void coValuePoti::joystickDown()
{
    float val = value;
    if (integer)
        val--;
    else
        val -= (maxValue - minValue) / 100.0f;
    val = coClamp(val, minValue, maxValue);

    value = val;
    displayValue(value);
}

/** Set activation state.
  @param en true = poti enabled
*/
void coValuePoti::setEnabled(bool en)
{
    if (en != enabled && uiElementProvider)
    {
        uiElementProvider->setEnabled(en);
    }
    coUIElement::setEnabled(en);
}

float coValuePoti::getMin() const
{
    return minValue;
}

float coValuePoti::getMax() const
{
    return maxValue;
}

float coValuePoti::getIncrement() const
{
    return increment;
}

bool coValuePoti::isInteger() const
{
    return integer;
}

bool coValuePoti::isDiscrete() const
{
    return discrete;
}

float coValuePoti::discreteValue(float val) const
{
    return (int(val / increment + 0.5f)) * increment;
}

const char *coValuePoti::getClassName() const
{
    return "coValuePoti";
}

bool coValuePoti::isOfClassName(const char *classname) const
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
