/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/coMenu.h>
#include <OpenVRUI/coMenuItem.h>
#include <OpenVRUI/coGenericSubMenuItem.h>
#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coSubMenuToolboxItem.h>
#include <OpenVRUI/coCombinedButtonInteraction.h>
#include <OpenVRUI/coUIContainer.h>

#include <OpenVRUI/sginterface/vruiRendererInterface.h>
#include <OpenVRUI/sginterface/vruiHit.h>

#include <OpenVRUI/util/vruiLog.h>

#include <OpenVRUI/sginterface/vruiIntersection.h>

#include <algorithm>
#include <config/CoviseConfig.h>

using namespace std;

namespace vrui
{

/** Constructor.
  @param parentMenu pointer to parent menu class, NULL if this is the topmost menu
*/
coMenu::coMenu(coMenu *parentMenu, const std::string &namePtr)
    : name(namePtr)
{
    if (parentMenu)
        visible = false;
    else
    {
        visible = false;
        //register
        if (vruiRendererInterface::the()->getJoystickManager())
            vruiRendererInterface::the()->getJoystickManager()->registerMenu(this);
    }
    parent = parentMenu;
    vruiRendererInterface::the()->getUpdateManager()->add(this);
    myMenuItem = 0;
    showMode = MENU_SLIDE;
    listener = 0;

    // remember menu's name - better in base class for debugging
    interaction = new coCombinedButtonInteraction(coInteraction::ButtonA, "Menu", coInteraction::Menu);
    moveInteraction = new coCombinedButtonInteraction(coInteraction::ButtonC, "Menu", coInteraction::Medium);
    wheelInteraction = new coCombinedButtonInteraction(coInteraction::WheelVertical, "Menu", coInteraction::Medium);
}

/// Destructor.
coMenu::~coMenu()
{
    this->closeMenu();

    if (vruiRendererInterface::the()->getJoystickManager())
        vruiRendererInterface::the()->getJoystickManager()->unregisterMenu(this);

    removeAll();
    if (myMenuItem)
    {
        myMenuItem->setMenu(0);
    }

    delete interaction;
    delete moveInteraction;
    delete wheelInteraction;
}

/// Get submenu item from which this menu can be accessed
coGenericSubMenuItem *coMenu::getSubMenuItem()
{
    return myMenuItem;
}

/// Set submenu item from which this menu can be accessed
void coMenu::setSubMenuItem(coGenericSubMenuItem *m)
{
    myMenuItem = m;
}

/** Add a new menu item to this menu. The menu item is appended
  to the end of the local linked list of menu items.
*/
void coMenu::add(coMenuItem *item)
{
    // deselect all items - need for joystickmanager
    for (std::list<coMenuItem *>::iterator it = items.begin(); it != items.end(); it++)
    {
        (*it)->selected(false);
    }
    item->selected(false);

    items.push_back(item);
    item->setParentMenu(this);
}

/** Insert a menu item into the linked menu item list.
  @param item menu item to insert
  @param pos  menu item index before which to insert the new item
*/
void coMenu::insert(coMenuItem *item, int pos)
{
    // deselect all items - need for joystickmanager
    for (std::list<coMenuItem *>::iterator it = items.begin(); it != items.end(); it++)
    {
        (*it)->selected(false);
    }
    item->selected(false);

    if (pos > (int)items.size())
        pos = (int)items.size();

    list<coMenuItem *>::iterator i = items.begin();
    for (int ctr = 0; ctr < pos; ++ctr)
        i++;
    items.insert(i, item);
    item->setParentMenu(this);
}

/// Remove a menu item from the linked list.
void coMenu::remove(coMenuItem *item)
{
    if (!item)
        return;
    item->selected(false);
    items.remove(item);

    if (dynamic_cast<coGenericSubMenuItem *>(item))
        (dynamic_cast<coGenericSubMenuItem *>(item))->closeSubmenu();
    if (item->getParentMenu() != NULL)
    {
        item->setParentMenu(0);
    }
}

/// Remove all menu items from the linked list.
void coMenu::removeAll()
{
    while (!items.empty())
    {
        coMenuItem *item = items.front();
        item->selected(false);

        item->setParentMenu(0);
        coUIElement *uie = item->getUIElement();
        if (uie && uie->getParent())
            uie->getParent()->removeElement(uie);
        items.pop_front();
    }
}

/// Return the number of menu items in the linked list.
int coMenu::getItemCount() const
{
    return (int)items.size();
}

/** This method is called when the menu is closed. If this menu is not
  the topmost menu, it closes the submenu entry by which it was opened previously.
*/
void coMenu::closeMenu()
{
    if (myMenuItem)
    {
        myMenuItem->closeSubmenu();
        // tell the joystickmanager
        if (vruiRendererInterface::the()->getJoystickManager())
            vruiRendererInterface::the()->getJoystickManager()->closedMenu(this, parent);
    }
}

/// This methode sets the desired showMode.
void coMenu::setShowMode(int newShowMode)
{
    showMode = newShowMode;
}

/// This methode returns the showMode currently set.
int coMenu::getShowMode() const
{
    return showMode;
}

/// Set a new menu listener to receive menu item events.
void coMenu::setMenuListener(coMenuFocusListener *ml)
{
    listener = ml;
}

/** Hit is called whenever the menu is intersected
  by the input device.
*/
int coMenu::hit(vruiHit *hit)
{

    //VRUILOG("coMenu::hit info: called")

    Result preReturn = vruiRendererInterface::the()->hit(this, hit);
    if (preReturn != ACTION_UNDEF)
        return preReturn;

    coInteractionManager::the()->registerInteraction(interaction);
    if (covise::coCoviseConfig::isOn("COVER.Menu.MoveMenus", true))
        coInteractionManager::the()->registerInteraction(moveInteraction);
    coInteractionManager::the()->registerInteraction(wheelInteraction);
    interaction->setHitByMouse(hit->isMouseHit());
    moveInteraction->setHitByMouse(hit->isMouseHit());
    wheelInteraction->setHitByMouse(hit->isMouseHit());

    if (listener)
        listener->focusEvent(true, this);

    return ACTION_CALL_ON_MISS;
}

/** 
  * select is called if the menu is selected via joystick
  * overwrite this function for RowMenu and ToolboxMenu 
  */
void coMenu::selected(bool)
{
}

/**
  * the menu is pressed via joystick
  */
void coMenu::doActionRelease()
{
    if (parent)
        closeMenu();
}

/**
  * the menu is pressed via joystick
  */
void coMenu::doActionPress()
{
}

/**
  * the menu is pressed via joystick
  */
void coMenu::doSecondActionRelease()
{
    doActionRelease();
}

/**
  * the menu is pressed via joystick
  */
void coMenu::doSecondActionPress()
{
}

void coMenu::makeVisible(coMenuItem *)
{
}

/// Miss is called once after a hit, if the menu is not intersected anymore.
void coMenu::miss()
{

    vruiRendererInterface::the()->miss(this);

    coInteractionManager::the()->unregisterInteraction(interaction);
    coInteractionManager::the()->unregisterInteraction(moveInteraction);
    coInteractionManager::the()->unregisterInteraction(wheelInteraction);

    if (listener)
        listener->focusEvent(false, this);
}

bool coMenu::isInteractionActive() const
{
    return (interaction->getState() == coInteraction::Active);
}

coCombinedButtonInteraction *coMenu::getInteraction() const
{
    return interaction;
}

coCombinedButtonInteraction *coMenu::getMoveInteraction() const
{
    return moveInteraction;
}

/// return the actual UI Element that represents this menu.
coUIElement *coMenu::getUIElement()
{
    return 0;
}

void coMenu::setVisible(bool newState)
{
    if (visible == newState)
        return; // state is already ok
    visible = newState;

    if (vruiRendererInterface::the()->getJoystickManager() && visible)
    {
        vruiRendererInterface::the()->getJoystickManager()->openedSubMenu(NULL, this);
    }
    else if (vruiRendererInterface::the()->getJoystickManager() && !visible)
    {
        vruiRendererInterface::the()->getJoystickManager()->closedMenu(this, parent);
    }
}

const char *coMenu::getClassName() const
{
    return "coMenu";
}

bool coMenu::isOfClassName(const char *classname) const
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

/// close all menus below a given item
void coMenu::removeAllAfter(coMenuItem *last)
{
    list<coMenuItem *>::iterator item = std::find(items.begin(), items.end(), last);
    if (item != items.end())
        item++;
    //std::list<coMenuItem*> itemsToBeRemoved;
    while (item != items.end() && (*item))
    {
        remove(*item);
        item = std::find(items.begin(), items.end(), last);
        if (item != items.end())
            item++;
    }
    /*   coMenuItem *item = NULL;
   if(items.size() > 0)
      item = items.back();
   while(item != last)
   {
      if(coMenu *p = item->getParentMenu())
      {
         coGenericSubMenuItem *sm = dynamic_cast<coGenericSubMenuItem *>(item);
         if(sm)
            sm->closeSubmenu();
         p->remove(item);
      }
      item = items.back();
   }*/
}

/// close all menus except a given item
void coMenu::closeAllOtherMenus(coMenuItem *leave)
{
    for (list<coMenuItem *>::iterator item = items.begin();
         item != items.end(); ++item)
    {
        if (*item != leave)
        {
            // dynamic_cast does not work on all platforms, so don´t use if not necessary

            if (dynamic_cast<coSubMenuItem *>(*item))
            {
                coSubMenuItem *subMenu = static_cast<coSubMenuItem *>(*item);
                subMenu->closeSubmenu();
            }
            else if (dynamic_cast<coSubMenuToolboxItem *>(*item))
            {
                coSubMenuToolboxItem *subMenuT = static_cast<coSubMenuToolboxItem *>(*item);
                subMenuT->closeSubmenu();
            }
            else
            {
                // This is not a menu - leave untouched
            }
        }
    }
}

/// check whether menu is currently visible
bool coMenu::isVisible() const
{
    return visible;
}

coMenuItemVector coMenu::getAllItems()
{
    coMenuItemVector res(items.begin(), items.end());
    return res;
}

int coMenu::index(const coMenuItem *item)
{
    coMenuItemVector allItems = getAllItems();
    size_t i = 0;
    while (i < allItems.size())
    {
        if (allItems[i] == item)
        {
            return (int)i;
        }
        ++i;
    }
    return -1;
}

coMenuItem *coMenu::getItemByName(const char *itemname)
{
    coMenuItem *found = NULL;

    for (list<coMenuItem *>::iterator item = items.begin(); item != items.end(); ++item)
    {
        if (strcmp((*item)->getName(), itemname) == 0)
        {
            return ((*item));
        }
        else if (strcmp((*item)->getClassName(), "coSubMenuItem") == 0)
        {
            coMenu *submenu = ((coSubMenuItem *)(*item))->getMenu();
            if (submenu)
            {

                found = submenu->getItemByName(itemname);
                if (found)
                    return (found);
            }
        }
    }

    return (NULL);
}

const char *coMenu::getName() const
{
    return name.c_str();
}

void coMenu::positionAllSubmenus()
{
    // reposition submenus
    for (std::list<coMenuItem *>::iterator it = items.begin(); it != items.end(); it++)
    {
        if ((*it)->isOfClassName("coSubMenuItem"))
        {
            ((coSubMenuItem *)(*it))->positionSubmenu();

            ((coSubMenuItem *)(*it))->getMenu()->positionAllSubmenus();
        }
    }
}

void coMenu::setAttachment(int attachment)
{
    attachment_ = attachment;

    for (std::list<coMenuItem *>::iterator it = items.begin(); it != items.end(); it++)
    {
        if ((*it)->isOfClassName("coSubMenuItem"))
        {
            ((coSubMenuItem *)(*it))->setAttachment(attachment);
            if (((coSubMenuItem *)(*it))->getMenu())
                ((coSubMenuItem *)(*it))->getMenu()->setAttachment(attachment);
        }
    }
}

bool coMenu::wasMoved() const
{
    return wasMoved_;
}

void coMenu::setMoved(bool flag)
{
    wasMoved_ = flag;
}

}
