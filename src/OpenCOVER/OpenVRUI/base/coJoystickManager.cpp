/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

#include "coSliderMenuItem.h"
#include "coSliderToolboxItem.h"
#include "coJoystickManager.h"
#include "coSubMenuItem.h"
#include "coSubMenuToolboxItem.h"
#include "coToolboxMenu.h"
#include "coRowMenu.h"
#include "coLabelMenuItem.h"
#include "coPotiMenuItem.h"
#include "coMovableBackgroundMenuItem.h"

#ifndef _WINDOWS
#include <sys/time.h>
#endif

using namespace std;

namespace vrui
{

// ----------------------------------------------------------------------------
// Construction / Destruction
// ----------------------------------------------------------------------------

coJoystickManager *coJoystickManager::joystickManager = 0;

/**
  * for pattern singelton
  * use instead of constructor
  */
coJoystickManager *coJoystickManager::instance()
{
    if (joystickManager == 0)
        joystickManager = new coJoystickManager();
    return joystickManager;
}

coJoystickManager::~coJoystickManager()
{
    joystickManager = 0;
}

coJoystickManager::coJoystickManager()
{
    activeMenu = NULL;
    oldActiveMenu = NULL;
    attachment = coUIElement::RIGHT;
    active = false;
    selectOnRelease = false;
    oldX = 0;
    oldY = 0;
    oldButton = 0;
    barrierXValue = 0.9f;
    barrierYValue = 0.5f;
    barrierMilliSeconds = 1000;
    firstPressed = -1;
    deltaSlider = -1;
    accuracyFloatSlider = 100;
    actualTime = -1;
    lastTime = -1;
    rootMenu = 0;
    actionRelease = false;
    actionPress = false;
    secondActionRelease = false;
    secondActionPress = false;
    menuTypeChanged = false;
}

/** 
  * activate or deactivate the joystick interaction
  */
void coJoystickManager::setActive(bool a)
{
    active = a;
    /*if (!active)
	{
		activeItem.clear();
		activeMenu = NULL;
	} else*/ if (active && activeMenu != NULL)
        setItemActive(true);

    //OutputDebugString("coJoystickManager::setActive ENDE\n");
}

/**
  * a root menu of the application has to be registered 
  * if it is the first rootMenu it will be the active one
  */
void coJoystickManager::registerMenu(coMenu *m, bool force)
{
    std::vector<coMenu *> test;
    if (menus.end() == std::find(menus.begin(), menus.end(), m))
    {
        menus.push_back(m);
    }

    if (rootMenu == 0 || force || dynamic_cast<coToolboxMenu *>(m) != NULL)
    {
        rootMenu = m;
        activeMenu = m;
        // clear all saved menupositions
        activeItem.clear();
        // set the menu as active menu
        setMenuActive(m);
        // choose the first item in menu
        activeItem[m] = 1;

        if (force && attachment != coUIElement::TOP && attachment != coUIElement::BOTTOM)
        {
            attachment = coUIElement::TOP;
        }

        if (active)
            setItemActive(true);
    }
}

/**
  * a root menu of the application has to be unregistered 
  */
void coJoystickManager::unregisterMenu(coMenu *m)
{
    if (m == rootMenu)
        rootMenu = NULL;
    if (m == activeMenu)
        activeMenu = NULL;
    menus.remove(m);
}

/**
  * is called if a submenu is closed via pick interaction 
  * set the parent menu as active 
  */
void coJoystickManager::closedMenu(coMenu *oldMenu, coMenu *newMenu)
{
    if (!newMenu || dynamic_cast<coToolboxMenu *>(rootMenu) != NULL)
        newMenu = rootMenu;
    if (active)
        setItemActive(false);
    setMenuActive(newMenu);
    if (oldMenu)
        oldActiveMenu = oldMenu;
    if (active)
        setItemActive(true);
}

/**
  * is called form coSubMenuItem if it opens the submenu
  * activate the submenu and the first item in it
  */
void coJoystickManager::openedSubMenu(coGenericSubMenuItem *, coMenu *subMenu)
{
    if (!active)
        return;

    if (!subMenu)
        return;

    setItemActive(false);
    activeMenu = subMenu;
    setItemActive(false);
    activeMenu = subMenu;
    activeItem[subMenu] = 1;
    setItemActive(true);
}

/** 
  * is called from coMenuItem if it is hit via ray
  * selects the item 
  */
void coJoystickManager::selectItem(coMenuItem *item, coMenu *parentMenu)
{
    oldActiveMenu = activeMenu;
    if (active)
        setItemActive(false);

    if (!parentMenu)
    {
        activeMenu = NULL;
        return;
    }

    // get the number of items in menu
    int numItems = parentMenu->getItemCount();

    // return if no item in menu
    if (numItems == 0)
    {
        activeItem[parentMenu] = 0;
        activeMenu = parentMenu;
        if (active)
            activeMenu->selected(true);
        return;
    }

    // get the items of menu
    coMenuItemVector items = parentMenu->getAllItems();
    // get all items which are active and no separator
    vector<coMenuItem *> goodItems;
    for (int i = 0; i < numItems; i++)
    {
        if (items[i]->getActive())
        {
            goodItems.push_back(items[i]);
            if (items[i] == item)
            {
                activeItem[parentMenu] = (int)goodItems.size();
                activeMenu = parentMenu;
                break;
            }
        }
    }
}

bool coJoystickManager::isDocument(coMenu *m)
{
    if (m->getItemCount() == 1)
    {
        if ((dynamic_cast<coMovableBackgroundMenuItem *>(m->getAllItems()[0]) != NULL) || (strcmp(m->getName(), "Object Name") == 0))
            return true;
    }
    return false;
}

void coJoystickManager::selectNextMenu(bool forward)
{
    // get the next menu
    for (list<coMenu *>::iterator it = menus.begin(); it != menus.end(); it++)
    {
        if (forward)
        {
            if ((*it) == activeMenu)
            {
                if (activeMenu != *(--menus.end()))
                    activeMenu = (*(++it));
                else
                {
                    it = menus.begin();
                    activeMenu = (*(it));
                }
                coMenu *original = activeMenu;
                while (!activeMenu->isVisible() || isDocument(activeMenu) || activeMenu->getItemCount() == 0)
                {
                    if (activeMenu != *(--menus.end()))
                        activeMenu = (*(++it));
                    else
                    {
                        it = menus.begin();
                        activeMenu = (*(it));
                    }
                    if (activeMenu == original)
                        break; // no active menu found
                }
                break;
            }
        }
        else
        {
            if ((*it) == activeMenu)
            {
                if (it != menus.begin())
                    activeMenu = (*(--it));
                else
                {
                    it = --menus.end();
                    activeMenu = (*(it));
                }
                coMenu *original = activeMenu;
                while (!activeMenu->isVisible() || isDocument(activeMenu) || activeMenu->getItemCount() == 0)
                {
                    if (it != menus.begin())
                        activeMenu = (*(--it));
                    else
                    {
                        it = --menus.end();
                        activeMenu = (*(it));
                    }
                    if (activeMenu == original)
                        break; // no active menu found
                }
                break;
            }
        }
    }
}

/*
 * selects or deselects the active item 
 */
void coJoystickManager::setItemActive(bool b)
{
    if (b && !active) // setting items inactive should always be possible (so never return if !b) (in case of activeMenu change while not active)
        return;

    if (activeMenu == NULL)
    {
        //fprintf(stderr, "\t setItem return because no activeMenu or activeItem\n");
        return;
    }

    if (b && !activeMenu->isVisible()) // setting items inactive should always be possible (so never return if !b) (in case of activeMenu change while not active)
        return;

    if (activeItem.empty())
        activeItem[activeMenu] = 1;

    // select the handle if menu is not a toolboxMenu
    if (activeItem[activeMenu] == 0 && dynamic_cast<coToolboxMenu *>(activeMenu) == NULL)
    {
        //activeItem[activeMenu]=1;
        //activeMenu->selected(b);
        //return;
    }

    int aItem = activeItem[activeMenu];
    // get the number of items in menu
    int numItems = activeMenu->getItemCount();

    //if (b)
    //	fprintf(stderr, "setItemActive %s %d\n", activeMenu->getName(), aItem);

    // return if no item in menu
    if (numItems == 0)
    {
        //if ( dynamic_cast<coToolboxMenu*>(activeMenu)==NULL)
        //{
        //   activeItem[activeMenu] = 0;
        //   activeMenu->selected(b);
        //}
        return;
    }

    vector<coMenuItem *> goodItems = getGoodItems(activeMenu);
    numItems = (int)goodItems.size();

    // check if number of items is smaller than active item
    if (numItems < aItem) //|| ((numItems-1 < aItem) && (dynamic_cast<coToolboxMenu*>(activeMenu)==NULL)))
    {
        // get the next menu
        selectNextMenu(true);
        goodItems = getGoodItems(activeMenu);
        numItems = (int)goodItems.size();
        aItem = 1;
        activeItem[activeMenu] = 1;
    }

    // check if active item is smaller 0 or 0 if it is a toolboxMenu
    if (aItem == 0) //(((aItem == 0) && (dynamic_cast<coToolboxMenu*>(activeMenu)==NULL)) || (aItem==0 && (dynamic_cast<coToolboxMenu*>(activeMenu)!=NULL)))
    {
        // get the menu before the active one
        selectNextMenu(false);
        goodItems = getGoodItems(activeMenu);
        numItems = (int)goodItems.size();
        activeItem[activeMenu] = numItems;
        aItem = numItems;
    }

    goodItems[aItem - 1]->selected(b);
    activeMenu->makeVisible(goodItems[aItem - 1]);
}

/**
  * sets the active menu
  */
void coJoystickManager::setMenuActive(coMenu *m)
{
    if (activeMenu != m)
        setItemActive(false);
    activeMenu = m;
}

void coJoystickManager::selectAction(int action)
{
    // which action is performed
    // switch left and right for attachment LEFT and for ToolboxMenus with attachment top

    bool switchActions = false;
    //attachment = activeMenu->getAttachment();

    if (attachment == coUIElement::RIGHT)
        switchActions = true;
    if (attachment == coUIElement::BOTTOM && dynamic_cast<coToolboxMenu *>(activeMenu) != NULL)
        switchActions = true;
    if ((attachment == coUIElement::BOTTOM || attachment == coUIElement::TOP) && (dynamic_cast<coToolboxMenu *>(activeMenu) == NULL))
        switchActions = true;

    if (switchActions)
    {
        actionRelease = (oldX == ACTION && action == 0);
        actionPress = (action == ACTION);
        secondActionRelease = (oldX == SECOND_ACTION && action == 0);
        secondActionPress = (action == SECOND_ACTION);
    }
    else
    {
        actionRelease = (oldX == SECOND_ACTION && action == 0);
        actionPress = (action == SECOND_ACTION);
        secondActionRelease = (oldX == ACTION && action == 0);
        secondActionPress = (action == ACTION);
    }
}

void coJoystickManager::selectActionButton(int action)
{
    // which action is performed
    // switch left and right for attachment LEFT and for ToolboxMenus with attachment top

    bool switchActions = false;
    //attachment = activeMenu->getAttachment();

    if (attachment == coUIElement::RIGHT)
        switchActions = true;
    if (attachment == coUIElement::BOTTOM && dynamic_cast<coToolboxMenu *>(activeMenu) != NULL)
        switchActions = true;
    if ((attachment == coUIElement::BOTTOM || attachment == coUIElement::TOP) && (dynamic_cast<coToolboxMenu *>(activeMenu) == NULL))
        switchActions = true;

    if (switchActions)
    {
        actionRelease = (oldButton == ACTION && action == 0);
        actionPress = (action == ACTION);
        secondActionRelease = (oldButton == SECOND_ACTION && action == 0);
        secondActionPress = (action == SECOND_ACTION);
    }
    else
    {
        actionRelease = (oldButton == SECOND_ACTION && action == 0);
        actionPress = (action == SECOND_ACTION);
        secondActionRelease = (oldButton == ACTION && action == 0);
        secondActionPress = (action == ACTION);
    }
}

/*
 * calls the action of the item 
 */
void coJoystickManager::pressActiveItem(int action)
{
    (void)action;

    if (!active)
        return;

    if (!activeMenu->isVisible())
        return;

    int selectItem = activeItem[activeMenu];

    //fprintf(stderr, "actRel=%d actPress=%d scdAcRel=%d scdAcPress=%d\n", actionRelease, actionPress, secondActionRelease, secondActionPress);

    // actions for handle
    if (selectItem == 0)
    {
        if (actionRelease)
            activeMenu->doActionRelease();
        else if (actionPress)
            activeMenu->doActionPress();
        else if (secondActionRelease)
            activeMenu->doSecondActionRelease();
        else if (secondActionPress)
            activeMenu->doSecondActionPress();
        // if menu closed set parentmenu active
        if (actionRelease)
        {
            coGenericSubMenuItem *subItem = activeMenu->getSubMenuItem();
            if (subItem)
            {
                // deselect item in sub menu
                setItemActive(false);
                // activate item in parent menu
                coSubMenuItem *sub = dynamic_cast<coSubMenuItem *>(subItem);
                if (sub)
                    setMenuActive(sub->getParentMenu());
                else
                {
                    coSubMenuToolboxItem *tool = dynamic_cast<coSubMenuToolboxItem *>(subItem);
                    if (tool)
                        setMenuActive(tool->getParentMenu());
                }
                setItemActive(true);
            }
        }
        return;
    }

    int numItems = activeMenu->getItemCount();

    // check if any items in menu
    if (numItems == 0)
        return;

    // get all items which are active and no separator
    vector<coMenuItem *> goodItems = getGoodItems(activeMenu);
    numItems = (int)goodItems.size();

    // action for items
    if (actionRelease)
        goodItems[selectItem - 1]->doActionRelease();
    else if (actionPress)
        goodItems[selectItem - 1]->doActionPress();
    else if (secondActionRelease)
        goodItems[selectItem - 1]->doSecondActionRelease();
    else if (secondActionPress)
        goodItems[selectItem - 1]->doSecondActionPress();

    // if item is a submenuItem change focus of menu
    coGenericSubMenuItem *sub = dynamic_cast<coGenericSubMenuItem *>(goodItems[selectItem - 1]);
    if (sub && actionRelease)
    {
        setItemActive(false);
        setMenuActive(sub->getMenu());
        setItemActive(true);
    }
}

/** 
  * takes the values of the valuators of the joystick (Flystick 2)
  * values between -1 and 1
  */
void coJoystickManager::update(float x, float y, long timeStamp)
{
    // if joystick is not active do nothing
    if (!active)
        return;

    int ix = 0;
    int iy = 0;
    if (x < -barrierXValue)
        ix = -1;
    else if (x > barrierXValue)
        ix = 1;
    if (y < -barrierYValue)
        iy = -1;
    else if (y > barrierYValue)
        iy = 1;
    update(ix, iy, timeStamp);
}

/** 
  * takes the values of the valuators of the joystick (Flystick 1)
  * values are -1 or 1
  */
void coJoystickManager::update(int x, int y, long timeStamp)
{
    // if joystick is not active do nothing
    if (!active)
        return;

    if ((activeMenu == NULL || !activeMenu->isVisible() || isDocument(activeMenu) || activeMenu->getItemCount() == 0) && rootMenu)
    {
        activeMenu = rootMenu;
        selectNextMenu(true);
        activeItem[activeMenu] = 1;
    }
    else if (activeMenu == NULL || !activeMenu->isVisible() || isDocument(activeMenu) || activeMenu->getItemCount() == 0)
    {
        activeMenu = *menus.begin();
        selectNextMenu(true);
        activeItem[activeMenu] = 1;
    }
    //fprintf(stderr, "activeMenu = %s item=%d\n", activeMenu->getName(), activeItem[activeMenu]);
    setItemActive(true);

    bool upDown = false;
    long act;
    if (timeStamp > 0)
        act = timeStamp;
    else
    {
#ifdef _WINDOWS
        act = GetTickCount();
#else
        struct timeval tv;
        gettimeofday(&tv, NULL);
        act = (long)(tv.tv_sec * 1000 + (tv.tv_usec / 1000));
#endif
    }

    lastTime = actualTime;
    actualTime = act;

    // for attachmennt top or bottom switch x and y
    bool switchedXY = false;
    if (activeMenu && dynamic_cast<coToolboxMenu *>(activeMenu) != NULL)
    {
        int tmpX = -1 * x;
        x = y;
        y = tmpX;
        switchedXY = true;
    }

    if (selectOnRelease)
    {
        // up down for item selection
        // left/right for action
        // deactivate the old item if a new one is selected (react on release)
        if (y == 0 && oldY != 0)
        {
            firstPressed = -1;
            setItemActive(false);
        }
        // pressed up / down
        // get the new index of item (react on release)
        if (y == 0 && oldY > 0)
        {
            // if item is justed decreased because of time do not decrease
            if (firstPressed < 0 || act - firstPressed >= 1000)
            {
                upDown = true;
                activeItem[activeMenu]--;
            }
        }
        else if (y == 0 && oldY < 0)
        {
            // if item is justed increased because of time do not increase
            if (firstPressed < 0 || act - firstPressed >= 1000)
            {
                upDown = true;
                activeItem[activeMenu]++;
            }
        }
        // activate the new item
        if (y == 0 && oldY != 0)
        {
            setItemActive(true);
        }

        // call release action after barrierSeconds
        // item is first pressed -> get the time
        if (oldY == 0 && y != 0)
        {
            firstPressed = act;
        }
        else if (y != 0 && oldY == y && (act - firstPressed >= barrierMilliSeconds))
        {
            setItemActive(false);
            if (y > 0)
            {
                upDown = true;
                activeItem[activeMenu]--;
            }
            else
            {
                upDown = true;
                activeItem[activeMenu]++;
            }
            setItemActive(true);
            firstPressed = act;
        }
    }
    else // select on button down
    {
        if (y != 0 && (oldY == 0 || act - firstPressed >= barrierMilliSeconds))
        {
            firstPressed = act;
            setItemActive(false);
            activeItem[activeMenu] -= y;
            setItemActive(true);
            upDown = true;
        }
    }

    // if menu type changed switch back x and y
    // toolbox but was no toolbox before
    if (activeMenu && dynamic_cast<coToolboxMenu *>(activeMenu) != NULL && !switchedXY)
    {
        int tmpY = y;
        y = x;
        x = tmpY;
        menuTypeChanged = true;
    }
    // no toolbox but was a toolbox before
    else if (activeMenu && dynamic_cast<coToolboxMenu *>(activeMenu) == NULL && switchedXY)
    {
        int tmpX = x;
        x = y;
        y = tmpX;
        menuTypeChanged = true;
    }
    // possibly pressed left / right
    if (!upDown && activeItem[activeMenu] >= 0 && !menuTypeChanged)
    {
        selectAction(x);
        pressActiveItem(x);
    }
    if (menuTypeChanged && x == 0 && oldX != 0)
        menuTypeChanged = false;

    oldX = x;
    oldY = y;
}

/** 
  * takes the values of the valuators of the joystick (Flystick 2)
  * values between -1 and 1
  * button for selection in menuModus
  */
void coJoystickManager::newUpdate(float x, float y, int button, long timeStamp)
{
    // if joystick is not active do nothing
    if (!active)
        return;

    int ix = 0;
    int iy = 0;
    if (x < -barrierXValue)
        ix = -1;
    else if (x > barrierXValue)
        ix = 1;
    if (y < -barrierYValue)
        iy = -1;
    else if (y > barrierYValue)
        iy = 1;
    newUpdate(ix, iy, button, timeStamp);
}

/** 
  * takes the values of the valuators of the joystick (Flystick 1)
  * values are -1 or 1
  * button for selection in menuModus
  */
void coJoystickManager::newUpdate(int x, int y, int button, long timeStamp)
{
    // if joystick is not active do nothing
    if (!active)
        return;

    setItemActive(false);

    // select menu if no menu is selected
    if ((activeMenu == NULL || !activeMenu->isVisible() || isDocument(activeMenu) || activeMenu->getItemCount() == 0) && rootMenu)
    {
        activeMenu = rootMenu;
        selectNextMenu(true);
        activeItem[activeMenu] = 1;
    }
    else if (activeMenu == NULL || !activeMenu->isVisible() || isDocument(activeMenu) || activeMenu->getItemCount() == 0)
    {
        activeMenu = *menus.begin();
        selectNextMenu(true);
        activeItem[activeMenu] = 1;
    }
    //fprintf(stderr, "activeMenu = %s item=%d\n", activeMenu->getName(), activeItem[activeMenu]);
    setItemActive(true);

    bool upDown = false;
    long act;
    if (timeStamp > 0)
        act = timeStamp;
    else
    {
#ifdef _WINDOWS
        act = GetTickCount();
#else
        struct timeval tv;
        gettimeofday(&tv, NULL);
        act = (long)(tv.tv_sec * 1000 + (tv.tv_usec / 1000));
#endif
    }

    lastTime = actualTime;
    actualTime = act;

    // for attachmennt top or bottom switch x and y
    bool switchedXY = false;
    if (activeMenu && dynamic_cast<coToolboxMenu *>(activeMenu) != NULL)
    {
        int tmpX = -1 * x;
        x = y;
        y = tmpX;
        switchedXY = true;
    }

    if (selectOnRelease)
    {
        // up down for item selection
        // left/right for action
        // deactivate the old item if a new one is selected (react on release)
        if (y == 0 && oldY != 0)
        {
            firstPressed = -1;
            setItemActive(false);
        }
        // pressed up / down
        // get the new index of item (react on release)
        if (y == 0 && oldY > 0)
        {
            // if item is justed decreased because of time do not decrease
            if (firstPressed < 0 || act - firstPressed >= 1000)
            {
                upDown = true;
                activeItem[activeMenu]--;
            }
        }
        else if (y == 0 && oldY < 0)
        {
            // if item is justed increased because of time do not increase
            if (firstPressed < 0 || act - firstPressed >= 1000)
            {
                upDown = true;
                activeItem[activeMenu]++;
            }
        }
        // activate the new item
        if (y == 0 && oldY != 0)
        {
            setItemActive(true);
        }

        // call release action after barrierSeconds
        // item is first pressed -> get the time
        if (oldY == 0 && y != 0)
        {
            firstPressed = act;
        }
        else if (y != 0 && oldY == y && (act - firstPressed >= barrierMilliSeconds))
        {
            setItemActive(false);
            if (y > 0)
            {
                upDown = true;
                activeItem[activeMenu]--;
            }
            else
            {
                upDown = true;
                activeItem[activeMenu]++;
            }
            setItemActive(true);
            firstPressed = act;
        }
    }
    else // select on button down
    {
        if (y != 0 && (oldY == 0 || act - firstPressed >= barrierMilliSeconds))
        {
            firstPressed = act;
            setItemActive(false);
            activeItem[activeMenu] -= y;
            setItemActive(true);
            upDown = true;
        }
    }

    // if menu type changed switch back x and y
    // toolbox but was no toolbox before
    if (activeMenu && dynamic_cast<coToolboxMenu *>(activeMenu) != NULL && !switchedXY)
    {
        int tmpY = y;
        y = x;
        x = tmpY;
        menuTypeChanged = true;
    }
    // no toolbox but was a toolbox before
    else if (activeMenu && dynamic_cast<coToolboxMenu *>(activeMenu) == NULL && switchedXY)
    {
        int tmpX = x;
        x = y;
        y = tmpX;
        menuTypeChanged = true;
    }

    // possibly pressed left / right
    if (!upDown && activeItem[activeMenu] >= 0 && !menuTypeChanged)
    {
        vector<coMenuItem *> goodItems = getGoodItems(activeMenu);
        if (goodItems.size() > 0 && ssize_t(goodItems.size()) >= activeItem[activeMenu])
        {
            coMenuItem *item = goodItems[activeItem[activeMenu] - 1];
            // if item is slider we need to use left and right
            if (dynamic_cast<coSliderMenuItem *>(item) || dynamic_cast<coSliderToolboxItem *>(item))
                selectAction(x);
            else
                selectActionButton(button);
        }
        else
            selectActionButton(button);
        pressActiveItem(button);
    }
    if (menuTypeChanged && x == 0 && oldX != 0)
        menuTypeChanged = false;

    oldX = x;
    oldY = y;
    oldButton = button;
}

vector<coMenuItem *> coJoystickManager::getGoodItems(coMenu *m)
{
    (void)m;

    // get the items of menu
    coMenuItemVector items = activeMenu->getAllItems();
    // get all items which are active and no separator
    vector<coMenuItem *> goodItems;
    int numItems = activeMenu->getItemCount();
    for (int i = 0; i < numItems; i++)
    {
        if (items[i]->getActive() && (strcmp(items[i]->getClassName(), "coSeparatorMenuItem") != 0) && (strcmp(items[i]->getClassName(), "coSeparatorToolboxMenuItem") != 0))
            goodItems.push_back(items[i]);
    }
    return goodItems;
}
}
