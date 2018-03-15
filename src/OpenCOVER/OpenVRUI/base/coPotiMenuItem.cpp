/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/coPotiMenuItem.h>

#include <OpenVRUI/coBackground.h>
#include <OpenVRUI/coColoredBackground.h>
#include <OpenVRUI/coFlatButtonGeometry.h>
#include <OpenVRUI/coLabel.h>
#include <OpenVRUI/coMenuContainer.h>

#include <OpenVRUI/sginterface/vruiIntersection.h>
#include <OpenVRUI/sginterface/vruiRendererInterface.h>

#include <config/CoviseConfig.h>

using namespace std;

namespace vrui
{

/** Constructor.
  @param name           poti description text
  @param min,max        minimum and maximum poti values
  @param defaultValue   default poti value
  @param coim           Covise interaction manager for collaborative usage (optional)
  @param idName         unique ID string for collaboration (optional)
*/
coPotiMenuItem::coPotiMenuItem(const string &name, float min, float max, float defaultValue,
                               vruiCOIM *coim, const string &idName)
    : coRowMenuItem(name)
{
    poti = new coValuePoti("", this, "Volume/valuepoti-bg", coim, idName);
    poti->setSize(2.0);
    poti->setMin(min);
    poti->setMax(max);
    poti->setValue(defaultValue);
    container->addElement(poti);
    container->addElement(label);
    vruiIntersection::getIntersectorForAction("coAction")->add(background->getDCS(), this);
}

/// Destructor.
coPotiMenuItem::~coPotiMenuItem()
{
    delete poti;
}

/** This method is called on intersections of the input device with the
  poti menu item.
  @return ACTION_CALL_ON_MISS
*/
int coPotiMenuItem::hit(vruiHit *hit)
{
    if (!vruiRendererInterface::the()->isRayActive())
        return ACTION_CALL_ON_MISS;
    // update coJoystickManager
    if (vruiRendererInterface::the()->isJoystickActive())
        vruiRendererInterface::the()->getJoystickManager()->selectItem(this, myMenu);

    Result preReturn = vruiRendererInterface::the()->hit(this, hit);
    if (preReturn != ACTION_UNDEF)
        return preReturn;

    background->setHighlighted(true);
    poti->hit(hit);
    return ACTION_CALL_ON_MISS;
}

/// Called when input device leaves the element.
void coPotiMenuItem::miss()
{
    if (!vruiRendererInterface::the()->isRayActive())
        return;

    vruiRendererInterface::the()->miss(this);
    background->setHighlighted(false);
    poti->miss();
}

/**
  * highlight the item
  */
void coPotiMenuItem::selected(bool select)
{
    coRowMenuItem::selected(select);
    poti->setHighlighted(true);
}

/**
  * move slider to lower
  */
void coPotiMenuItem::doActionPress()
{
    float old = poti->getValue();
    std::string attachment = covise::coCoviseConfig::getEntry("COVER.VRUI.MenuAttachment");
    if (attachment == "LEFT")
        poti->joystickUp();
    else
        poti->joystickDown();
    potiValueChanged(old, poti->getValue(), poti);
}

/**
  * move slider to upper
  */
void coPotiMenuItem::doSecondActionPress()
{
    float old = poti->getValue();
    std::string attachment = covise::coCoviseConfig::getEntry("COVER.VRUI.MenuAttachment");
    if (attachment == "LEFT")
        poti->joystickDown();
    else
        poti->joystickUp();
    potiValueChanged(old, poti->getValue(), poti);
}

/// Calls the value changed listener (coValuePotiActor) when poti value was changed.
//  This function is called by the coValuePoti interactor
void coPotiMenuItem::potiValueChanged(float /*oldValue*/, float newValue, coValuePoti *, int)
{
    if (listener)
        listener->menuEvent(this);
}

/// Calls the value changed listener (coValuePotiActor) when button was pressed.
//  This function is called by the coValuePoti interactor
void coPotiMenuItem::potiPressed(coValuePoti *, int)
{
    if (listener)
        listener->menuPressEvent(this);
}

/// Calls the value changed listener (coValuePotiActor) when button was released.
//  This function is called by the coValuePoti interactor
void coPotiMenuItem::potiReleased(coValuePoti *, int)
{
    if (listener)
        listener->menuReleaseEvent(this);
}

/** Set a new poti value.
  @param newValue new poti value, must be within min/max boundaries
*/
void coPotiMenuItem::setValue(float newValue)
{
    poti->setValue(newValue);
}

/** Set minimum poti value.
  @param m minimum value
*/
void coPotiMenuItem::setMin(float min)
{
    poti->setMin(min);
}

/** Set poti value type.
  @param i true = integer value, false = floating point value
*/
void coPotiMenuItem::setInteger(bool i)
{
    poti->setInteger(i);
}

/** Set increment.
  @param incr trincrement
*/
void coPotiMenuItem::setIncrement(float incr)
{
    poti->setIncrement(incr);
}

/** Set maximum poti value.
  @param m maximum value
*/

void coPotiMenuItem::setMax(float m)
{
    poti->setMax(m);
}

/** Get current poti value.
  @return current poti value
*/
float coPotiMenuItem::getValue() const
{
    return poti->getValue();
}

float coPotiMenuItem::getMax() const
{
    return poti->getMax();
}

float coPotiMenuItem::getMin() const
{
    return poti->getMin();
}

bool coPotiMenuItem::isInteger() const
{
    return poti->isInteger();
}

const char *coPotiMenuItem::getClassName() const
{
    return "coPotiMenuItem";
}

bool coPotiMenuItem::isOfClassName(const char *classname) const
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
}
