/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//#include <OpenVRUI/coColoredBackground.h>
//#include <OpenVRUI/coLabel.h>
#include <OpenVRUI/coMenu.h>
#include <OpenVRUI/coMenuItem.h>
#include <OpenVRUI/coMenuContainer.h>
#include <OpenVRUI/coGenericSubMenuItem.h>

#include <OpenVRUI/sginterface/vruiTransformNode.h>

using namespace std;

namespace vrui
{

void coMenuListener::menuEvent(coMenuItem *)
{
}

void coMenuListener::menuPressEvent(coMenuItem *)
{
}

void coMenuListener::menuReleaseEvent(coMenuItem *)
{
}

/** Constructor. Creates a new menu item.
  @param name text to appear on label
*/
coMenuItem::coMenuItem(const string &name)
    : myName(name)
{
    myMenu = 0;
    listener = 0;
    visible = true;
    active_ = true;
}

/// Destructor. Removes this menu item from the parent menu.
coMenuItem::~coMenuItem()
{
    // say bye to our menu
    if (myMenu)
        myMenu->remove(this);
}

/// Set a new menu listener to receive menu item events.
void coMenuItem::setMenuListener(coMenuListener *ml)
{
    listener = ml;
}

/// Get the menu listener to receive menu item events.
coMenuListener *coMenuItem::getMenuListener()
{
    return listener;
}

/// Set the parent menu in which this menu item is listed.
void coMenuItem::setParentMenu(coMenu *menu)
{
    myMenu = menu;

    if (coGenericSubMenuItem *submenu = dynamic_cast<coGenericSubMenuItem *>(this))
    {
        if (submenu->getMenu())
            submenu->getMenu()->setParent(menu);
    }
}

/// returns the symbolic name of this menu item.
const char *coMenuItem::getName() const
{
    return myName.c_str();
}

/// set my name - items with labels must call this when changing label
void coMenuItem::setName(const string &newName, bool updateTwins)
{
    myName = newName;
    setLabel(newName);
}

/// this function may be overloaded - it is called by setName()
void coMenuItem::setLabel(const string & /*newName*/)
{
}

/** show or hide this menu item., just removes it from the scenegraph in the default implementation,
should do something better in the real implementation*/
void coMenuItem::setVisible(bool v)
{
    if (visible == v)
        return; // nothing to do

    vruiTransformNode *dcs = getUIElement()->getDCS();
    visible = v;
    if (v)
    {
        if (getUIElement()->getParent())
        {
            getUIElement()->getParent()->getDCS()->addChild(dcs);
        }
    }
    else
    {
        dcs->removeAllParents();
    }
}

bool coMenuItem::isVisible() const
{
    return visible;
}

const char *coMenuItem::getClassName() const
{
    return "coMenuItem";
}

bool coMenuItem::isOfClassName(const char *classname) const
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
            return false;
        }
    }

    // nobody is NULL
    return false;
}

/// return the actual UI Element that represents this menu.
coUIElement *coMenuItem::getUIElement()
{
    return 0;
}

/**
  * if item should do any action if joystick selects the item,
  * overwrite this function
  */
void coMenuItem::selected(bool)
{
}

/**
  * if item should do any action if item is pressed via joystick,
  * overwrite this function
  */
void coMenuItem::doActionPress()
{
}

/**
  * if item should do any action if item is pressed via joystick,
  * overwrite this function
  */
void coMenuItem::doActionRelease()
{
}

/**
  * if item should do any action if item is pressed second action via joystick,
  * overwrite this function
  * if not overwritten calls first action (doAction)
  */
void coMenuItem::doSecondActionPress()
{
    doActionPress();
}

/**
  * if item should do any action if item is pressed second action via joystick,
  * overwrite this function
  * closes the menu if it has a parent menu
  */
void coMenuItem::doSecondActionRelease()
{
    if (myMenu)
        myMenu->closeMenu();
}
}
