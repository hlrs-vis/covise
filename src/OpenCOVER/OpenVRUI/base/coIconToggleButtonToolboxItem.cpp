/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/coLabel.h>

#include <OpenVRUI/sginterface/vruiButtons.h>
#include <OpenVRUI/sginterface/vruiIntersection.h>
#include <OpenVRUI/sginterface/vruiRendererInterface.h>
#include <OpenVRUI/sginterface/vruiHit.h>

#include <OpenVRUI/coIconToggleButtonToolboxItem.h>

#include <OpenVRUI/coMenuContainer.h>

#include <OpenVRUI/coToggleButtonGeometry.h>
#include <OpenVRUI/coColoredBackground.h>

namespace vrui
{

/** Constructor.
  @param symbolicName label string
*/

coIconToggleButtonToolboxItem::coIconToggleButtonToolboxItem(const std::string &symbolicName)
    : coToolboxMenuItem(symbolicName)
    , pressed(false)
{
    myButton = new coToggleButton(new coToggleButtonGeometry(symbolicName), this);
    //myButton = new coToggleButton(new coToggleButtonGeometry(symbolicName),this);

    myButton->setSize(100.0f, 100.0f, 1.0f);

    menuContainer->addElement(myButton);

    // make sure icon always is centered in container
    menuContainer->setAlignment(coMenuContainer::CENTER);
    menuContainer->setNumAlignedMin(0);

    vruiIntersection::getIntersectorForAction("coAction")->add(background->getDCS(), this);
}

/// Destructor
coIconToggleButtonToolboxItem::~coIconToggleButtonToolboxItem()
{
    vruiIntersection::getIntersectorForAction("coAction")->remove(this);
    delete myButton;
}

/** This method is called on intersections of the input device with the item.
  @return ACTION_CALL_ON_MISS
*/
int coIconToggleButtonToolboxItem::hit(vruiHit *hit)
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

    vruiButtons *buttons = hit->isMouseHit() ? renderer->getMouseButtons() : renderer->getButtons();

    if (buttons->wasPressed())
    {
        pressed = true;
    }

    // feed through if Button was activated
    if (pressed && buttons->wasReleased())
    {
        pressed = false;
        setState(!getState(), true /* generate event */);
    }

    return ACTION_CALL_ON_MISS;
}

/// Called when input device leaves the element, removes highlight.
void coIconToggleButtonToolboxItem::miss()
{
    if (!vruiRendererInterface::the()->isRayActive())
        return;

    background->setHighlighted(false);
}

/**
  * highlight the item
  */
void coIconToggleButtonToolboxItem::selected(bool select)
{
    if (vruiRendererInterface::the()->isJoystickActive())
        background->setHighlighted(select);
}

/**
  * open or close Submenu
  */
void coIconToggleButtonToolboxItem::doActionRelease()
{
    // perform own action (if existent) as secondary
    if (listener)
        listener->menuEvent(this);
}

const char *coIconToggleButtonToolboxItem::getClassName() const
{
    return "coIconTooggleButtonToolboxItem";
}

bool coIconToggleButtonToolboxItem::isOfClassName(const char *classname) const
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

/** Set new checkbox state.
  @param newState true = checked, false = unchecked
  @param generateEvent if true, a menuEvent is generated
*/
void coIconToggleButtonToolboxItem::setState(bool newState, bool generateEvent)
{
    if (myButton->getState() != newState)
    {
        myButton->setState(newState);
    }

    if (generateEvent)
    {
        if (listener)
            listener->menuEvent(this);
    }
}

/** Get checkbox state.
  @return checkbox state (true = checked, false = unchecked)
*/
bool coIconToggleButtonToolboxItem::getState()
{
    return myButton->getState();
}

//set if item is active
void coIconToggleButtonToolboxItem::setActive(bool a)
{
    // if item is activated add background to intersector
    if (!active_ && a)
    {
        vruiIntersection::getIntersectorForAction("coAction")->add(background->getDCS(), this);
        myButton->setActive(a);
    }
    // if item is deactivated remove background from intersector
    else if (active_ && !a)
    {
        vruiIntersection::getIntersectorForAction("coAction")->remove(this);
        myButton->setActive(a);
    }
    coToolboxMenuItem::setActive(a);
}
}
