/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/coMenuContainer.h>
#include <OpenVRUI/coLabel.h>
#include <OpenVRUI/coColoredBackground.h>

#include <OpenVRUI/sginterface/vruiButtons.h>
#include <OpenVRUI/sginterface/vruiHit.h>
#include <OpenVRUI/sginterface/vruiIntersection.h>
#include <OpenVRUI/sginterface/vruiRendererInterface.h>

#include <OpenVRUI/util/vruiLog.h>

using namespace std;

namespace vrui
{

/** Constructor.
  @param name label string
*/
coButtonMenuItem::coButtonMenuItem(const string &name)
    : coRowMenuItem(name)
{
    lastContent = false;
    pressed = false;
    icon = NULL;
    space = new coBackground();
    space->setMinWidth((float)LEFTMARGIN);
    container->addElement(space);
    container->addElement(label);
    vruiIntersection::getIntersectorForAction("coAction")->add(background->getDCS(), this);
}

/** Constructor
   @param name label string
   @param nameIcon filename of icon needs <filename>.rgb, <filename>-selected.rgb and <filename>-disabled.rgb
   @param texSizeX size of icon, if 0 fit to background
   @param texSizeY size of icon, if 0 fit to background
*/
coButtonMenuItem::coButtonMenuItem(const std::string &name, const std::string &nameIcon, float texSizeX, float texSizeY)
    : coRowMenuItem(name)
{
    lastContent = false;
    pressed = false;
    space = NULL;
    //TODO ACTOR for textured bg and correct name of icon
    iconName_ = nameIcon;
    icon = new coTexturedBackground(nameIcon, nameIcon + "-selected", nameIcon + "-disabled", this);
    icon->setSize((float)LEFTMARGIN);
    icon->setMinWidth((float)LEFTMARGIN);
    icon->setTexSize(texSizeX, texSizeY);
    icon->setRepeat(false);
    container->addElement(icon);
    container->addElement(label);
    vruiIntersection::getIntersectorForAction("coAction")->add(background->getDCS(), this);
}

/// Destructor
coButtonMenuItem::~coButtonMenuItem()
{
    if (vruiIntersection::getIntersectorForAction("coAction"))
        vruiIntersection::getIntersectorForAction("coAction")->remove(this);
    if (space)
        delete space;
    if (icon)
        delete icon;
}

/** This method is called on intersections of the input device with the
  slider menu item.
  @return ACTION_CALL_ON_MISS
*/

int coButtonMenuItem::hit(vruiHit *hit)
{

    //VRUILOG("coButtonMenuItem::hit info: called")

    vruiRendererInterface *renderer = vruiRendererInterface::the();

    // return if pickinteraction via ray is not active
    if (!vruiRendererInterface::the()->isRayActive())
        return ACTION_CALL_ON_MISS;

    // update coJoystickManager
    if (vruiRendererInterface::the()->isJoystickActive())
        vruiRendererInterface::the()->getJoystickManager()->selectItem(this, myMenu);

    Result preReturn = renderer->hit(this, hit);
    if (preReturn != ACTION_UNDEF)
        return preReturn;

    background->setHighlighted(true);

    vruiButtons *buttons = hit->isMouseHit() ? renderer->getMouseButtons() : renderer->getButtons();

    // feed through if Button was activated
    if (buttons->wasPressed())
    {
        pressed = true;
    }

    if (buttons->wasReleased() && !vruiRendererInterface::the()->isJoystickActive())
    {
        if (pressed && listener)
            listener->menuEvent(this);

        pressed = false;
    }
    return ACTION_CALL_ON_MISS;
}

/// Called when input device leaves the element, removes highlight.
void coButtonMenuItem::miss()
{
    if (!vruiRendererInterface::the()->isRayActive())
        return;

    pressed = false;
    vruiRendererInterface::the()->miss(this);
    background->setHighlighted(false);
}

/**
  * if item should do any action if joystick selects the item,
  * overwrite this function
  */
void coButtonMenuItem::selected(bool select)
{
    background->setHighlighted(select);
    coRowMenuItem::selected(select);
}

/**
  * if item should do any action if item is pressed via joystick,
  * overwrite this function
  */
void coButtonMenuItem::doActionRelease()
{
    if (listener)
        listener->menuEvent(this);
}

const char *coButtonMenuItem::getClassName() const
{
    return "coButtonMenuItem";
}

bool coButtonMenuItem::isOfClassName(const char *classname) const
{
    // paranoia makes us mistrust the string library and check for NULL.
    if (classname && getClassName())
    { // check for identity
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

float coButtonMenuItem::getIconSizeX()
{
    return icon->getTexXSize();
}

float coButtonMenuItem::getIconSizeY()
{
    return icon->getTexYSize();
}

void coButtonMenuItem::setIconSizeX(float x)
{
    icon->setTexSize(x, icon->getTexYSize());
}

void coButtonMenuItem::setIconSizeY(float y)
{
    icon->setTexSize(icon->getTexXSize(), y);
}

//set if item is active
void coButtonMenuItem::setActive(bool a)
{
    // if item is activated add background to intersector
    if (!active_ && a)
    {
        vruiIntersection::getIntersectorForAction("coAction")->add(background->getDCS(), this);
        if (icon != NULL)
            icon->setActive(a);
    }
    // if item is deactivated remove background from intersector
    else if (active_ && !a)
    {
        vruiIntersection::getIntersectorForAction("coAction")->remove(this);
        if (icon != NULL)
            icon->setActive(a);
    }
    coRowMenuItem::setActive(a);
}
}
