/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/coSliderMenuItem.h>
#include <OpenVRUI/coLabel.h>
#include <OpenVRUI/coMenuContainer.h>
#include <OpenVRUI/coColoredBackground.h>

#include <OpenVRUI/sginterface/vruiHit.h>
#include <OpenVRUI/sginterface/vruiIntersection.h>
#include <OpenVRUI/sginterface/vruiRendererInterface.h>

#include <OpenVRUI/util/vruiLog.h>
#include <config/CoviseConfig.h>

using namespace std;

namespace vrui
{

/** Constructor, derived from coRowMenuItem.
  @param name     Label text.
  @param min,max  Minimum and maximum slider values.
  @param init     Initial slider value.
  @see coRowMenuItem
*/
coSliderMenuItem::coSliderMenuItem(const string &name, float min, float max, float init)
    : coRowMenuItem(name)
{
    slider = new coSlider(this);

    // Constrain initialization values:
    if (max < min)
        max = min;
    //was: coClamp(init, min, max);
    init = (init < min ? min : (init > max ? max : init));

    setMin(min);
    setMax(max);
    setValue(init);

    container->addElement(slider);
    container->addElement(label);
    vruiIntersection::getIntersectorForAction("coAction")->add(background->getDCS(), this);
}

/** This method is called on intersections of the input device with the
  slider menu item.
  @param hit hit information
  @return ACTION_CALL_ON_MISS
*/

int coSliderMenuItem::hit(vruiHit *hit)
{
    if (!vruiRendererInterface::the()->isRayActive())
        return ACTION_CALL_ON_MISS;

    // update coJoystickManager
    if (vruiRendererInterface::the()->isJoystickActive())
        vruiRendererInterface::the()->getJoystickManager()->selectItem(this, myMenu);

    //VRUILOG("coSliderMenuItem::hit info: called")

    Result preReturn = vruiRendererInterface::the()->hit(this, hit);
    if (preReturn != ACTION_UNDEF)
        return preReturn;

    background->setHighlighted(true);
    slider->hit(hit);
    return ACTION_CALL_ON_MISS;
}

/// Called when input device leaves the element.
void coSliderMenuItem::miss()
{
    if (!vruiRendererInterface::the()->isRayActive())
        return;

    slider->miss();
    background->setHighlighted(false);
}

/**
  * highlight the item
  */
void coSliderMenuItem::selected(bool select)
{
    if (vruiRendererInterface::the()->isJoystickActive())
    {
        coRowMenuItem::selected(select);
        slider->setHighlighted(select);
        label->setHighlighted(select);
    }
}

void coSliderMenuItem::doActionRelease()
{
    slider->resetLastPressAction();
    sliderReleasedEvent(slider);
}

/**
  * move slider to upper
  */
void coSliderMenuItem::doActionPress()
{
    if (coUIElement::LEFT == myMenu->getAttachment())
        slider->joystickDown();
    else
        slider->joystickUp();
}

void coSliderMenuItem::doSecondActionRelease()
{
    slider->resetLastPressAction();
    sliderReleasedEvent(slider);
}

/**
  * move slider to lower
  */
void coSliderMenuItem::doSecondActionPress()
{
    if (coUIElement::LEFT == myMenu->getAttachment())
        slider->joystickUp();
    else
        slider->joystickDown();
}

/// Destructor.
coSliderMenuItem::~coSliderMenuItem()
{
    vruiIntersection::getIntersectorForAction("coAction")->remove(this);
    delete slider;
}

/** Set the slider value.
  @param val new slider value
*/
void coSliderMenuItem::setValue(float val)
{
    slider->setValue(val);
}

/** @return current slider value
 */
float coSliderMenuItem::getValue() const
{
    return slider->getValue();
}

/** Set slider minimum value.
  @param val minimum value
*/
void coSliderMenuItem::setMin(float val)
{
    slider->setMin(val);
}

/** Get current slider minimum value.
  @return current slider minimum value
*/
float coSliderMenuItem::getMin() const
{
    return slider->getMin();
}

/** Set slider maximum value.
  @param val maximum value
*/
void coSliderMenuItem::setMax(float val)
{
    slider->setMax(val);
}

/** Get slider maximum value.
  @return slider maximum value
*/
float coSliderMenuItem::getMax() const
{
    return slider->getMax();
}

/** Set number of tick lines on slider.
  @param nt new tick number
*/
void coSliderMenuItem::setNumTicks(float nt)
{
    slider->setNumTicks(nt);
}

/** Get number of ticks.
  @return number of tick lines
*/
float coSliderMenuItem::getNumTicks() const
{
    return slider->getNumTicks();
}

/** Set value display precision.
  @param nt number of decimals to display
*/
void coSliderMenuItem::setPrecision(int nt)
{
    slider->setPrecision(nt);
}

/** Get precision.
  @return number of decimals displayed
*/
int coSliderMenuItem::getPrecision() const
{
    return slider->getPrecision();
}

/** Get integer number display state.
  @return true if value display is integer
*/
bool coSliderMenuItem::isInteger() const
{
    return slider->isInteger();
}

/** Set integer or floating point value display.
  @param val true = integer display, false = floating point display
*/
void coSliderMenuItem::setInteger(bool val)
{
    slider->setInteger(val);
}

const char *coSliderMenuItem::getClassName() const
{
    return "coSliderMenuItem";
}

bool coSliderMenuItem::isOfClassName(const char *classname) const
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
            return coRowMenuItem::isOfClassName(classname);
        }
    }

    // nobody is NULL
    return false;
}

void coSliderMenuItem::sliderEvent(coSlider *s)
{
    if (s == slider && listener)
        listener->menuEvent(this);
}

void coSliderMenuItem::sliderReleasedEvent(coSlider *s)
{
    if (s == slider && listener)
        listener->menuReleaseEvent(this);
}

//set if item is active
void coSliderMenuItem::setActive(bool a)
{
    // if item is activated add background to intersector
    if (!active_ && a)
    {
        vruiIntersection::getIntersectorForAction("coAction")->add(background->getDCS(), this);
        slider->setActive(a);
    }
    // if item is deactivated remove background from intersector
    else if (active_ && !a)
    {
        vruiIntersection::getIntersectorForAction("coAction")->remove(this);
        slider->setActive(a);
    }
    coRowMenuItem::setActive(a);
}
}
