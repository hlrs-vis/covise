/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/coPotiToolboxItem.h>

#include <OpenVRUI/coBackground.h>
#include <OpenVRUI/coLabel.h>
#include <OpenVRUI/coFlatButtonGeometry.h>
#include <OpenVRUI/coMenuContainer.h>
#include <OpenVRUI/coColoredBackground.h>

#include <OpenVRUI/coPotiMenuItem.h>

#include <OpenVRUI/sginterface/vruiButtons.h>
#include <OpenVRUI/sginterface/vruiIntersection.h>
#include <OpenVRUI/sginterface/vruiRendererInterface.h>

#include <config/CoviseConfig.h>

namespace vrui
{

/** Constructor.
  @param symbolicName   poti description text
  @param min,max        minimum and maximum poti values
  @param defaultValue   default poti value
  @param coim           Covise interaction manager for collaborative usage (optional)
  @param idName         unique ID string for collaboration (optional)
*/
coPotiToolboxItem::coPotiToolboxItem(const std::string &symbolicName,
                                     float min, float max, float defaultValue,
                                     vruiCOIM *coim, const std::string &idName)
    : coToolboxMenuItem(symbolicName)
{

    poti = new coValuePoti("", this, "Volume/valuepoti-bg", coim, idName);

    poti->setSize(2);
    poti->setMin(min);
    poti->setMax(max);
    poti->setValue(defaultValue);

    label = new coLabel();
    label->setString(symbolicName);

    menuContainer->addElement(poti);
    menuContainer->addElement(label);

    vruiIntersection::getIntersectorForAction("coAction")->add(background->getDCS(), this);
}

/// Destructor.
coPotiToolboxItem::~coPotiToolboxItem()
{
    vruiIntersection::getIntersectorForAction("coAction")->remove(this);

    delete label;
    delete poti;
}

/** This method is called on intersections of the input device with the
  poti Toolbox item.
  @return ACTION_CALL_ON_MISS
*/
int coPotiToolboxItem::hit(vruiHit *hit)
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

    poti->hit(hit);

    return ACTION_CALL_ON_MISS;
}

/// Called when input device leaves the element.
void coPotiToolboxItem::miss()
{
    if (!vruiRendererInterface::the()->isRayActive())
        return;

    background->setHighlighted(false);
    poti->miss();
}

/**
  * highlight the item
  */
void coPotiToolboxItem::selected(bool select)
{
    coToolboxMenuItem::selected(select);
    poti->setHighlighted(true);
}

/**
  * move slider to lower
  */
void coPotiToolboxItem::doActionPress()
{
    float old = poti->getValue();
    std::string attachment = covise::coCoviseConfig::getEntry("COVER.Plugin.AKToolbar.Attachment");
    if (attachment == "BOTTOM")
        poti->joystickUp();
    else
        poti->joystickDown();
    potiValueChanged(old, poti->getValue(), poti);
}

/**
  * move slider to upper
  */
void coPotiToolboxItem::doSecondActionPress()
{
    float old = poti->getValue();
    std::string attachment = covise::coCoviseConfig::getEntry("COVER.Plugin.AKToolbar.Attachment");
    if (attachment == "BOTTOM")
        poti->joystickDown();
    else
        poti->joystickUp();
    potiValueChanged(old, poti->getValue(), poti);
}

/// Calls the value changed listener (coValuePotiActor) when poti value was changed.
void coPotiToolboxItem::potiValueChanged(float /*oldValue*/, float newValue, coValuePoti * /*this_poti*/, int /*context*/)
{
    // activate own listener
    if (listener)
        listener->menuEvent(this);
}

/// Calls the value changed listener (coValuePotiActor) when button was pressed.
void coPotiToolboxItem::potiPressed(coValuePoti *, int)
{
    if (listener)
        listener->menuPressEvent(this);
}

/// Calls the value changed listener (coValuePotiActor) when button was released.
void coPotiToolboxItem::potiReleased(coValuePoti *, int)
{
    if (listener)
        listener->menuReleaseEvent(this);
}

/** Get current poti value.
  @return current poti value
*/
float coPotiToolboxItem::getValue() const
{
    return poti->getValue();
}

void coPotiToolboxItem::setIncrement(float incr)
{
    poti->setIncrement(incr);
}

const char *coPotiToolboxItem::getClassName() const
{
    return "coPotiToolboxItem";
}

bool coPotiToolboxItem::isOfClassName(const char *classname) const
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
}
