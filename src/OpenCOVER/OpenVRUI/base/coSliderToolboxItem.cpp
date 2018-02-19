/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */



#include <OpenVRUI/coSliderToolboxItem.h>

#include <OpenVRUI/coSliderMenuItem.h>
#include <OpenVRUI/coSlider.h>
#include <OpenVRUI/coBackground.h>
#include <OpenVRUI/coLabel.h>
#include <OpenVRUI/coMenuContainer.h>
#include <OpenVRUI/coColoredBackground.h>

#include <OpenVRUI/sginterface/vruiButtons.h>
#include <OpenVRUI/sginterface/vruiIntersection.h>
#include <OpenVRUI/sginterface/vruiRendererInterface.h>
#include <OpenVRUI/util/vruiDefines.h>

#include <util/unixcompat.h>
#include <config/CoviseConfig.h>

namespace vrui
{

/** Constructor, derived from coRowToolboxItem.
  @param name     Label text.
  @param min,max  Minimum and maximum slider values.
  @param init     Initial slider value.
  @see coRowToolboxItem
*/
coSliderToolboxItem::coSliderToolboxItem(const std::string &name, float min, float max, float init)
    : coToolboxMenuItem(name)
{
    slider = new coSlider(this);
    minLabel = new coLabel();
    maxLabel = new coLabel();
    label = new coLabel();

    setLabel(name);

    // Constrain initialization values:
    if (max < min)
        max = min;
    init = coClamp(init, min, max);

    setMin(min);
    setMax(max);
    setValue(init);
    menuContainer->setOrientation(coRowContainer::HORIZONTAL);
    menuContainer->addElement(minLabel);
    menuContainer->addElement(slider);
    menuContainer->addElement(maxLabel);
    menuContainer->addElement(label);
    // menuContainer->addElement(label);

    vruiIntersection::getIntersectorForAction("coAction")->add(background->getDCS(), this);
}

/// Destructor.
coSliderToolboxItem::~coSliderToolboxItem()
{
    vruiIntersection::getIntersectorForAction("coAction")->remove(this);

    delete slider;
    delete minLabel;
    delete maxLabel;
    delete label;
}

/** This method is called on intersections of the input device with the
  slider menu item.
  @param p hit position
  @param h hit information
  @return ACTION_CALL_ON_MISS
*/
int coSliderToolboxItem::hit(vruiHit *hit)
{
    if (!vruiRendererInterface::the()->isRayActive())
        return ACTION_CALL_ON_MISS;
    // update coJoystickManager
    if (vruiRendererInterface::the()->isJoystickActive())
        vruiRendererInterface::the()->getJoystickManager()->selectItem(this, myMenu);

    background->setHighlighted(true);

    vruiRendererInterface *renderer = vruiRendererInterface::the();

    Result preReturn = renderer->hit(this, hit);
    if (preReturn != ACTION_UNDEF)
        return preReturn;

    slider->hit(hit);

    return ACTION_CALL_ON_MISS;
}

/// Called when input device leaves the element.
void coSliderToolboxItem::miss()
{
    if (!vruiRendererInterface::the()->isRayActive())
        return;

    slider->miss();
    background->setHighlighted(false);
}

void coSliderToolboxItem::selected(bool select)
{
    if (vruiRendererInterface::the()->isJoystickActive())
    {
        background->setHighlighted(select);
        slider->setHighlighted(select);
    }
}

/**
  * move slider to lower
  */
void coSliderToolboxItem::doActionPress()
{
    std::string attachment = covise::coCoviseConfig::getEntry("COVER.Plugin.AKToolbar.Attachment");
    if (attachment == "BOTTOM")
        slider->joystickUp();
    else
        slider->joystickDown();
}
/**
  * move slider to upper
  */
void coSliderToolboxItem::doSecondActionPress()
{
    std::string attachment = covise::coCoviseConfig::getEntry("COVER.Plugin.AKToolbar.Attachment");
    if (attachment == "BOTTOM")
        slider->joystickDown();
    else
        slider->joystickUp();
}

/** Set the slider value.
  @param val new slider value
*/
void coSliderToolboxItem::setValue(float val)
{
    slider->setValue(val);
}

/** @return current slider value
 */
float coSliderToolboxItem::getValue() const
{
    return slider->getValue();
}

/** Set slider minimum value.
  @param val minimum value
*/
void coSliderToolboxItem::setMin(float val)
{
    slider->setMin(val);

    char string[64];

    if (slider->isInteger())
        snprintf(string, 64, "%d", (int)val);
    else
        snprintf(string, 64, "%.*f", slider->getPrecision(), val);

    minLabel->setString(string);
}

/** Get current slider minimum value.
  @return current slider minimum value
*/
float coSliderToolboxItem::getMin() const
{
    return slider->getMin();
}

/** Set slider maximum value.
  @param val maximum value
*/
void coSliderToolboxItem::setMax(float val)
{
    slider->setMax(val);

    char string[64];

    if (slider->isInteger())
        snprintf(string, 64, "%d", (int)val);
    else
        snprintf(string, 64, "%.*f", slider->getPrecision(), val);

    maxLabel->setString(string);
}

/** Get slider maximum value.
  @return slider maximum value
*/
float coSliderToolboxItem::getMax() const
{
    return slider->getMax();
}

/** Set number of tick lines on slider.
  @param nt new tick number
*/
void coSliderToolboxItem::setNumTicks(float nt)
{
    slider->setNumTicks(nt);
}

/** Get number of ticks.
  @return number of tick lines
*/
float coSliderToolboxItem::getNumTicks() const
{
    return slider->getNumTicks();
}

/** Set value display precision.
  @param nt number of decimals to display
*/
void coSliderToolboxItem::setPrecision(int nt)
{
    slider->setPrecision(nt);
}

/** Get precision.
  @return number of decimals displayed
*/
int coSliderToolboxItem::getPrecision() const
{
    return slider->getPrecision();
}

/** Get integer number display state.
  @return true if value display is integer
*/
bool coSliderToolboxItem::isInteger() const
{
    return slider->isInteger();
}

/** Set integer or floating point value display.
  @param val true = integer display, false = floating point display
*/
void coSliderToolboxItem::setInteger(bool val)
{
    slider->setInteger(val);
}

const char *coSliderToolboxItem::getClassName() const
{
    return "coSliderToolboxItem";
}

bool coSliderToolboxItem::isOfClassName(const char *classname) const
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
            return coToolboxMenuItem::isOfClassName(classname);
        }
    }

    // nobody is NULL
    return false;
}

void coSliderToolboxItem::sliderEvent(coSlider *s)
{
    if (s == slider && listener)
        listener->menuEvent(this);
}

void coSliderToolboxItem::sliderReleasedEvent(coSlider *s)
{
    if (s == slider && listener)
        listener->menuReleaseEvent(this);
}

//set if item is active
void coSliderToolboxItem::setActive(bool a)
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
    coToolboxMenuItem::setActive(a);
}

void coSliderToolboxItem::setLabel(const std::string &name)
{
    std::string labelstr = "[" + name + "]";
    if (name.empty())
        labelstr.clear();
    label->setString(labelstr);
}
}
