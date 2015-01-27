/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cstdlib>
#ifdef _WIN32
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#endif

#include <OpenVRUI/coValuePoti.h>

#include <OpenVRUI/coCombinedButtonInteraction.h>
#include <OpenVRUI/sginterface/vruiRendererInterface.h>
#include <OpenVRUI/sginterface/vruiUIElementProvider.h>

#include <OpenVRUI/util/vruiDefines.h>

using namespace std;

namespace vrui
{

/** Constructor.
  @param buttonText           button text
  @param actor                action listener for poti value changes
  @param backgroundTexture    background texture file name
  @param cInterfaceManager    collaborative interface manager (optional)
  @param interfaceName        unique ID string for collaboration (optional)
*/
coSlopePoti::coSlopePoti(const string &buttonText, coValuePotiActor *actor,
                         const string &backgroundTexture, vruiCOIM *cInterfaceManager,
                         const string &interfaceName)
    : coValuePoti(buttonText, actor, backgroundTexture, cInterfaceManager, interfaceName)
{
    positive = false;
}

/** Set slope poti value.
  @param val new value [-MAX_SLOPE..MAX_SLOPE]
*/
void coSlopePoti::setValue(float val)
{
    if (positive)
        val = fabs(val);
    val = coClamp(val, minValue, maxValue);
    setLinearValue(convertSlopeToLinear(val));
    uiElementProvider->update();
}

/** Converts a slope value to a linear value. The linear value
  can be converted to the current rotational angle by multiplying
  it with 2 Pi.
  @param slope slope value to convert [-MAX_SLOPE..MAX_SLOPE]
  @return corresponding linear value [0..1]
*/
float coSlopePoti::convertSlopeToLinear(float slope) const
{

    float linear;

    if (positive)
        slope = fabs(slope);
    if (slope > 0.0f)
        linear = atan(1.0f / slope);
    else if (slope < 0.0f)
        linear = (float)(M_PI / 2.0f + atan(-slope));
    else
        linear = (float)(M_PI / 2.0f);

    return (float)(linear / (2.0f * M_PI));
}

/** Converts a linear value to a slope value.
  @param  linear value [0..1]
  @return corresponding slope value [-MAX_SLOPE..MAX_SLOPE]
*/
float coSlopePoti::convertLinearToSlope(float linear) const
{

    float slope;

    if (linear >= 0.5f)
        linear -= 0.5f; // constrain to [0..0.5]
    linear *= 360.0f; // convert to degrees [0..180]
    if (linear <= 0.0f)
        slope = (float)MAX_SLOPE;
    else if (linear < 90.0f)
        slope = (float)(1.0f / tan(linear * M_PI / 180.0f));
    else if (linear == 90.0f)
        slope = 0.0f;
    else if (linear < 180.0f)
        slope = (float)(-tan((linear - 90.0f) * M_PI / 180.0f));
    else
        slope = (float)-MAX_SLOPE;

    if (positive)
        slope = fabs(slope);
    return slope;
}

/// Update method, called every overall frame.
bool coSlopePoti::update()
{

    /* FIXME if((cover->getSyncMode()==MS_COUPLING)&&(!cover->isMaster()))
      return true;*/

    float oldValue, newValue;
    oldValue = newValue = value;

    if (interactionW->isRunning() || interactionW->wasStarted())
    {
        newValue += interactionW->getWheelCount() / 300.0f;
        newValue = fmod(newValue, 1.0f);
    }
    else if (interactionA->isRunning() || interactionB->isRunning())
    {
        coCombinedButtonInteraction *inter = interactionA->isRunning() ? interactionA : interactionB;
        vruiCoord mouseCoord = inter->getHandMatrix();

        float mouseRotation;
        float rotScale;

        if (inter->is2D())
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
            newValue += rotScale * (mouseRotation - lastRoll) / 360.0f;
            while (newValue < 0.0f)
                newValue += 1.0f;
            while (newValue > 1.0f)
                newValue -= 1.0f;

            lastRoll = mouseRotation;
            while (lastRoll > 180.0)
                lastRoll -= 360.0;
            while (lastRoll < -180.0)
                lastRoll += 360.0;
        }
    }

    if (oldValue != newValue)
    {
        setLinearValue(newValue);
        // trigger "changed" event
        myActor->potiValueChanged(convertLinearToSlope(oldValue),
                                  convertLinearToSlope(newValue), this);
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

/** Set linear slope poti value.
  @param lin linear value [0..1]
*/
void coSlopePoti::setLinearValue(float linear)
{
    value = linear;
    displayValue(convertLinearToSlope(linear));
}

/** Define if poti should only represent positive values.
  @param newState true = positive values only
*/
void coSlopePoti::setABS(bool newState)
{
    positive = newState;
}

/** Get state of positive values display.
  @return true if positive values only
*/
bool coSlopePoti::getABS() const
{
    return positive;
}

/** Get current poti value.
  @return current poti value
*/
float coSlopePoti::getValue() const
{
    return convertLinearToSlope(value);
}

/** Inform collaborators about value changes.
  @param message message string passed to collaborators
*/
void coSlopePoti::remoteOngoing(const char *message)
{
    float val = (float)strtod(message, 0);
    float oldValue = value;
    setLinearValue(val);
    // trigger "changed" event
    myActor->potiValueChanged(convertLinearToSlope(oldValue), convertLinearToSlope(val), this, remoteContext);
}

const char *coSlopePoti::getClassName() const
{
    return "coSlopePoti";
}

bool coSlopePoti::isOfClassName(const char *classname) const
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
            return coValuePoti::isOfClassName(classname);
        }
    }

    // nobody is NULL
    return false;
}
}
