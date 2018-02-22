/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/coSlider.h>

#include <OpenVRUI/coCombinedButtonInteraction.h>

#include <OpenVRUI/sginterface/vruiTransformNode.h>
#include <OpenVRUI/sginterface/vruiUIElementProvider.h>

#include <OpenVRUI/sginterface/vruiHit.h>
#include <OpenVRUI/sginterface/vruiIntersection.h>
#include <OpenVRUI/sginterface/vruiRendererInterface.h>

#include <OpenVRUI/util/vruiLog.h>
#include <OpenVRUI/util/vruiDefines.h>

#define ZOFFSET 5

#include <math.h>

using namespace std;

namespace vrui
{

/** Called every time the slider value is changed by the user.
  It is also called if setValue() is called with generateEvent==true.
  This method needs to be overwritten by child classes.
  @param slider coSlider which triggered this event
*/
void coSliderActor::sliderEvent(coSlider *)
{
    //if (cover->debugLevel(2))
    VRUILOG("call to base class function: coSliderActor::sliderEvent(coSlider *)")
}

void coSliderActor::sliderReleasedEvent(coSlider *){};

/** Constructor.
  @param actor     pointer to the coSliderActor which is called on slider events
  @param showValue true = slider value is to be displayed next to the slider position indicator (optional)
*/
coSlider::coSlider(coSliderActor *actor, bool showValue)
{

    this->showValue = showValue;

    myActor = actor;

    myX = 0.0f;
    myY = 0.0f;
    myZ = 0.0f;

    numTicks = 10.0f;
    precision = 1;

    integer = false;

    minVal = 0.0f;
    maxVal = 1.0f;
    value = 0.5f;

    myWidth = 300.0f;
    myHeight = 80.0f;
    myDepth = 0.0f;

    unregister = false;
    active_ = true;

    if (showValue)
    {
        dialSize = myHeight / 4.0f;
    }
    else
    {
        dialSize = myHeight / 2.0f;
    }

    if (myActor != 0)
    {
        vruiIntersection::getIntersectorForAction("coAction")->add(getDCS(), this);
    }

    resizeGeometry();

    vruiRendererInterface::the()->getUpdateManager()->add(this);

    interactionA = new coCombinedButtonInteraction(coInteraction::ButtonA, "Slider", coInteraction::Menu);
    interactionWheel[0] = new coCombinedButtonInteraction(coInteraction::WheelVertical, "Slider", coInteraction::Menu);
    interactionWheel[1] = new coCombinedButtonInteraction(coInteraction::WheelHorizontal, "Slider", coInteraction::Menu);

    lastPressAction = 0;
}

/** Destructor.
  The slider is removed from all parents to which it is attached.
*/
coSlider::~coSlider()
{
    if (myActor != 0)
    {
        vruiIntersection::getIntersectorForAction("coAction")->remove(this);
    }

    delete interactionA;
    delete interactionWheel[0];
    delete interactionWheel[1];
}

/////////////////// Better values for the slider
void coSlider::adjustSlider(float &mini, float &maxi, float & /*value*/, float &step, int &digits)
{
    if (mini == maxi)
    {
        step = 1.0f;
        digits = 4;
        return;
    }

    float basis[3] = { 1.0f, 2.0f, 5.0f };
    float tens = 1.0f;

    int i = 0;
    digits = 0;
    step = tens * basis[i];
    float numSteps = (maxi - mini) / step;
    //fprintf(stderr, "adjustSlider: min=%f, max=%f, step=%f, digits=%d", min, max, step, digits);

    while (numSteps > 500 || numSteps < -500)
    {
        ++i;
        if (i > 2)
        {
            i = 0;
            tens *= 10.0f;
            --digits;
        }
        step = tens * basis[i];
        numSteps = (maxi - mini) / step;
    }

    while (numSteps < 100 && numSteps > -100)
    {
        --i;
        if (i < 0)
        {
            i = 2;
            tens /= 10.0f;
            ++digits;
        }
        step = tens * basis[i];
        numSteps = (maxi - mini) / step;
    }

    //adjust all
    if (maxi > mini)
    {
        mini = (float)(step * floor(mini / (double)step));
        maxi = (float)(step * ceil(maxi / (double)step));
    }
    else
    {
        mini = (float)(step * ceil(mini / (double)step));
        maxi = (float)(step * floor(maxi / (double)step));
    }
    //fprintf(stderr, "; min=%f, max=%f, step=%f, digits=%d\n", min, max, step, digits);
    ////   value = step * my_round(value/(double)step);
}

/** Sets the slider size. The slider height is set to 1/3 the slider width,
  the depth is set to zero.
  @param s new slider width [mm]
*/
void coSlider::setSize(float s)
{
    if (s > 0)
    {
        myWidth = s;
        myHeight = s / 3.0f;
        if (showValue)
        {
            dialSize = myHeight / 4.0f;
        }
        else
        {
            dialSize = myHeight / 2.0f;
        }
        myDepth = 0.0f;
        resizeGeometry();
    }
}

/** Sets the slider size, using different values for all dimensions.
  @param xs,ys,zs new slider sizes [mm]
*/
void coSlider::setSize(float xs, float ys, float zs)
{
    if ((xs > 0.0f) && (ys > 0.0f))
    {
        myWidth = xs;
        myHeight = ys;
        if (showValue)
        {
            dialSize = myHeight / 4.0f;
        }
        else
        {
            dialSize = myHeight / 2.0f;
        }
        myDepth = zs;
        resizeGeometry();
    }
}

/// Returns the current slider value.
float coSlider::getValue() const
{
    clamp();
    return value;
}

/// Returns the minimum slider value.
float coSlider::getMin() const
{
    return minVal;
}

/// Returns the maximum slider value.
float coSlider::getMax() const
{
    return maxVal;
}

/// Returns the number of tickmarks used on the dial.
float coSlider::getNumTicks() const
{
    return numTicks;
}

/// Returns the number of decimals used for the slider value display.
int coSlider::getPrecision() const
{
    return precision;
}

/// Returns true if only integer slider values are accepted.
bool coSlider::isInteger() const
{
    return integer;
}

float coSlider::getDialSize() const
{
    return dialSize;
}

/** Sets the integer mode.
  @param i true = slider accepts only integer values, false = floating point values are accepted
*/
void coSlider::setInteger(bool i)
{
    integer = i;
    createGeometry();
    uiElementProvider->update();
}

/** Sets the slider to a new value. The position indicator is adjusted accordingly.
  @param v new slider value
  @param generateEvent true to generate a sliderEvent
*/
void coSlider::setValue(float v, bool generateEvent)
{
    value = v;
    createGeometry();
    uiElementProvider->update();
    if (generateEvent && myActor)
        myActor->sliderEvent(this);
}

/// Sets a new minimum value for the slider range.
void coSlider::setMin(float m)
{
    minVal = m;
    createGeometry();
    uiElementProvider->update();
}

/// Sets a new maximum value for the slider range.
void coSlider::setMax(float m)
{
    maxVal = m;
    createGeometry();
    uiElementProvider->update();
}

/// Sets the number of tickmarks to be used on the dial.
void coSlider::setNumTicks(float t)
{
    numTicks = t;
    createGeometry();
    uiElementProvider->update();
}

/// Sets the number of digits to use for the slider value display.
void coSlider::setPrecision(int nt)
{
    precision = nt;
}

/// Sets the slider position in object space [mm].
void coSlider::setPos(float x, float y, float z)
{
    myX = x;
    myY = y;
    myZ = z;
    getDCS()->setTranslation(x, y, z);
}

/**
  @return true if the slider value is displayed
  */
bool coSlider::getShowValue() const
{
    return showValue;
}

/// functions activates or deactivates the slider
void coSlider::setActive(bool a)
{
    // if item is activated add background to intersector
    if (!active_ && a)
    {
        vruiIntersection::getIntersectorForAction("coAction")->add(getDCS(), this);
        active_ = a;
        createGeometry();
        uiElementProvider->update();
    }
    // if item is deactivated remove background from intersector
    else if (active_ && !a)
    {
        vruiIntersection::getIntersectorForAction("coAction")->remove(this);
        active_ = a;
        createGeometry();
        uiElementProvider->update();
    }
}

/// This routine is called in each frame displayed by the system.

bool coSlider::update()
{

    for (int i=0; i<2; ++i)
    {
        if (interactionWheel[i]->isRunning() || interactionWheel[i]->wasStarted())
        {
            float oldValue = value;

            if (maxVal - minVal > 0.0)
            {
                if (integer && maxVal - minVal < 30.0)
                {
                    value += interactionWheel[i]->getWheelCount();
                    if (value < minVal)
                        value = minVal;
                    if (value > maxVal)
                        value = maxVal;
                }
                else
                {
                    value += interactionWheel[i]->getWheelCount() * (maxVal - minVal) / 30.0f;
                }
            }

            if (integer)
            {
                value = (float)(static_cast<int>(value));
            }

            clamp();

            uiElementProvider->update();

            if (value != oldValue && myActor)
                myActor->sliderEvent(this);
        }
    }
    if (interactionA->isRunning())
    {
        //TODO should use the last intersection point and continue moving the slider
    }
    if (interactionA->wasStopped())
    {
        if (myActor)
        {
            myActor->sliderReleasedEvent(this);
            myActor->sliderEvent(this);
        }
    }
    if (unregister && !interactionA->isRunning() && !interactionA->wasStopped())
    // don't unregister while the interaction is running or was JUST stopped
    {
        if (interactionA->isRegistered() /* && (interactionA->getState() != coInteraction::Active)*/)
        {
            coInteractionManager::the()->unregisterInteraction(interactionA);
        }

        for (int i=0; i<2; ++i)
        {
            if (interactionWheel[i]->isRegistered() /* && (interactionWheel->getState() != coInteraction::Active)*/)
            {
                coInteractionManager::the()->unregisterInteraction(interactionWheel[i]);
            }
        }
        unregister = false;
    }
    return true;
}

void coSlider::setHighlighted(bool hl)
{
    coUIElement::setHighlighted(hl);
    createGeometry();
    uiElementProvider->update();
}

/** This method is called when the input device intersects the slider.
  If it does, the slider is highlighted and the pointer is grabbed and slider
  value changes are computed accordingly.
  @param hit      hit
*/
int coSlider::hit(vruiHit *hit)
{

    //VRUILOG("coSlider::hit info: called")

    Result preReturn = vruiRendererInterface::the()->hit(this, hit);
    if (preReturn != ACTION_UNDEF)
        return preReturn;

    setHighlighted(true);
    unregister = false; // abort a "scheduled" unregister (i.e. miss() during active interaction)

    if (!interactionA->isRegistered())
    {
        coInteractionManager::the()->registerInteraction(interactionA);
        interactionA->setHitByMouse(hit->isMouseHit());
    }

    for (int i=0; i<2; ++i)
    {
        if (!interactionWheel[i]->isRegistered())
        {
            coInteractionManager::the()->registerInteraction(interactionWheel[i]);
            interactionWheel[i]->setHitByMouse(hit->isMouseHit());
        }
    }

    if (interactionA->isRunning() || interactionA->wasStarted())
    {
        vruiMatrix *myTrans = vruiRendererInterface::the()->createMatrix();
        float oldValue = value;
        myTrans->makeInverse(getTransformMatrix());

        coVector localPoint = myTrans->getFullXformPt(hit->getWorldIntersectionPoint());
        if (maxVal - minVal > 0.0)
            value = minVal + (((float)localPoint[0] - dialSize) / (myWidth - (2.0f * dialSize)) * (maxVal - minVal));
        else
            value = minVal;
        if (integer)
        {
            value = (float)(static_cast<int>(value));
        }

        clamp();
        uiElementProvider->update();
        if (value != oldValue && myActor)
            myActor->sliderEvent(this);
    }

    return ACTION_CALL_ON_MISS;
}

/// This method is called when the pointer leaves the slider. Then the highlight is removed.
void coSlider::miss()
{
    vruiRendererInterface::the()->miss(this);
    setHighlighted(false);
    unregister = true; // don't unregister immediately, interaction might still be running
}

void coSlider::joystickUp()
{
    // 	std::cerr << "joystickUp" << std::endl;
    int numSteps = 1;
    if (vruiRendererInterface::the()->getJoystickManager())
    {
        if (vruiRendererInterface::the()->getJoystickManager()->getDeltaSlider() < 0)
            numSteps = 1;
        else
        {
            if (lastPressAction == 0)
            {
                // 				std::cerr << "no last timestamp -> slider first pressed" << std::endl;
                lastPressAction = vruiRendererInterface::the()->getJoystickManager()->getActualTime();
                // 				std::cerr << "lastPressaction ist now: " << lastPressAction << std::endl;
            }
            numSteps = (int)(vruiRendererInterface::the()->getJoystickManager()->getActualTime() - lastPressAction) / vruiRendererInterface::the()->getJoystickManager()->getDeltaSlider();
            // 			std::cerr << "number of steps: " << numSteps << std::endl;
            // 			std::cerr << "calculation: actual time (" << vruiRendererInterface::the()->getJoystickManager()->getActualTime() <<") - last time ("<<lastPressAction<<") / delta ("<<vruiRendererInterface::the()->getJoystickManager()->getDeltaSlider()<< ")" << std::endl;
            if (numSteps > 0)
            {
                // 				std::cerr<< "number of steps bigger 0 set last pressaction to " << vruiRendererInterface::the()->getJoystickManager()->getActualTime() << std::endl;
                lastPressAction = vruiRendererInterface::the()->getJoystickManager()->getActualTime();
            }
        }
    }
    if (integer)
    {
        // 		std::cerr << "value of int-slider: oldvalue("<<value<<") + numSteps("<<numSteps<<")" << std::endl;
        value += numSteps;
        // 		std::cerr << "value = " <<value << std::cerr;
    }
    else
    {
        int step = 100;
        if (vruiRendererInterface::the()->getJoystickManager())
            step = vruiRendererInterface::the()->getJoystickManager()->getAccuracyFloatSlider();
        // 		std::cerr<< "value of float-slider: oldvalue("<<value<<") +  (max-min ("<<maxVal-minVal<<")) / step ("<<step<<") * numSteps("<<numSteps<<")" << std::endl;
        value += (maxVal - minVal) / step * numSteps;
        // 		std::cerr << "value = " <<value << std::cerr;
    }
    if (value > maxVal)
        value = maxVal;

    clamp();
    uiElementProvider->update();
    if (myActor)
        myActor->sliderEvent(this);
}

void coSlider::joystickDown()
{
    // 	std::cerr << "joystickUp" << std::endl;

    int numSteps = 1;
    if (vruiRendererInterface::the()->getJoystickManager())
    {
        if (vruiRendererInterface::the()->getJoystickManager()->getDeltaSlider() < 0)
            numSteps = 1;
        else
        {
            if (lastPressAction == 0)
            {
                // 				std::cerr << "no last timestamp -> slider first pressed" << std::endl;
                lastPressAction = vruiRendererInterface::the()->getJoystickManager()->getActualTime();
                // 				std::cerr << "lastPressaction ist now: " << lastPressAction << std::endl;
            }
            numSteps = (int)(vruiRendererInterface::the()->getJoystickManager()->getActualTime() - lastPressAction) / vruiRendererInterface::the()->getJoystickManager()->getDeltaSlider();
            // 			std::cerr << "number of steps: " << numSteps << std::endl;
            // 			std::cerr << "calculation: actual time (" << vruiRendererInterface::the()->getJoystickManager()->getActualTime() <<") - last time ("<<lastPressAction<<") / delta ("<<vruiRendererInterface::the()->getJoystickManager()->getDeltaSlider()<< ")" << std::endl;
            if (numSteps > 0)
            {
                // 				std::cerr<< "number of steps bigger 0 set last pressAction to " << vruiRendererInterface::the()->getJoystickManager()->getActualTime() << std::endl;
                lastPressAction = vruiRendererInterface::the()->getJoystickManager()->getActualTime();
            }
        }
    }
    if (integer)
    {
        // 		std::cerr << "value of int-slider: oldvalue("<<value<<") - numSteps("<<numSteps<<")" << std::endl;
        value -= numSteps;
        // 		std::cerr << "value = "<< value << std::cerr;
    }
    else
    {
        int step = 100;
        if (vruiRendererInterface::the()->getJoystickManager())
            step = vruiRendererInterface::the()->getJoystickManager()->getAccuracyFloatSlider();
        // 		std::cerr<< "value of float-slider: oldvalue("<<value<<") -  (max-min ("<<maxVal-minVal<<")) / step ("<<step<<") * numSteps("<<numSteps<<")" << std::endl;
        value -= (maxVal - minVal) / step * numSteps;
        // 		std::cerr << "value = "<< value << std::cerr;
    }
    if (value < minVal)
        value = minVal;
    clamp();
    uiElementProvider->update();
    if (myActor)
        myActor->sliderEvent(this);
}

void coSlider::resetLastPressAction()
{
    // 	std::cerr << "reset last press action to zero" << std::endl;
    lastPressAction = 0;
}

void coSlider::clamp() const
{
    value = coClamp(value, minVal, maxVal);
}

const char *coSlider::getClassName() const
{
    return "coSlider";
}

bool coSlider::isOfClassName(const char *classname) const
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
