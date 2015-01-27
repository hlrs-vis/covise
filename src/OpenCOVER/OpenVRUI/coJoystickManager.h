/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COJOYSTICKMANAGER_H
#define COJOYSTICKMANAGER_H

#include <util/coTypes.h>

#include <stdio.h>
#include <string>
#include <vector>
#include <map>
#include <time.h>

#include "coMenu.h"

#define FORWARD 1
#define BACKWARD -1
#define ACTION 1
#define SECOND_ACTION -1

namespace vrui
{

/**
  * This class manages the selection of the menu via joystick
  */
/// Manages the joystick as input device
class OPENVRUIEXPORT coJoystickManager
{

public:
    /// get the singleton instance of coJoystickManager
    static coJoystickManager *instance(); //get instance of coJoystickManager
    ~coJoystickManager();

protected:
    coJoystickManager(); //because of singleton
    static coJoystickManager *joystickManager; ///< static singleton variable

    void setMenuActive(coMenu *m);
    void setItemActive(bool b);
    void pressActiveItem(int action);
    void selectAction(int action);
    void selectActionButton(int action);
    std::vector<coMenuItem *> getGoodItems(coMenu *m);
    bool isDocument(coMenu *m);
    void selectNextMenu(bool forward);

    coMenu *activeMenu; ///< the selected menu
    coMenu *oldActiveMenu; ///< selected menu before
    std::map<coMenu *, int> activeItem; ///< list of the selected items in menu and in parent menus
    std::list<coMenu *> menus;
    coMenu *rootMenu; ///<root menu

    bool active; ///< flag if the joystick manager is active

    bool selectOnRelease;

    int oldX; ///< save old value of x valuator
    int oldY; ///< save old value of y valuator
    int oldButton;

    float barrierXValue;
    float barrierYValue;
    int barrierMilliSeconds;
    long firstPressed;
    long actualTime;
    long lastTime;

    long deltaSlider;
    int accuracyFloatSlider;

    int attachment;

    bool actionRelease;
    bool actionPress;
    bool secondActionRelease;
    bool secondActionPress;

    bool menuTypeChanged;

public:
    /// activate the selectionmode via joystick
    void setActive(bool a);
    bool getActive()
    {
        return active;
    };
    ///set select on release
    void setSelectOnRelease(bool b)
    {
        selectOnRelease = b;
    };
    bool getSelectOnRelease()
    {
        return selectOnRelease;
    };
    ///set barrier value for x value of slider
    void setBarrierXValue(float f)
    {
        barrierXValue = f;
    };
    ///set barrier value for y value of slider
    void setBarrierYValue(float f)
    {
        barrierYValue = f;
    };
    float getBarrierXValue()
    {
        return barrierXValue;
    };
    float getBarrierYValue()
    {
        return barrierYValue;
    };
    /// set milliseconds to wait until a action is called (release action)
    void setBarrierMilliSeconds(int sec)
    {
        barrierMilliSeconds = sec;
    };
    int getBarrierMilliSeconds()
    {
        return barrierMilliSeconds;
    };
    /// set milliseconds after a slider updates its value
    void setDeltaSlider(long delta)
    {
        deltaSlider = delta;
    }
    long getDeltaSlider()
    {
        return deltaSlider;
    };
    /// set accuracy of float-slider (one step is maxValue-minValue/accuracy)
    void setAccuracyFloatSlider(int accuracy)
    {
        accuracyFloatSlider = accuracy;
    };
    int getAccuracyFloatSlider()
    {
        return accuracyFloatSlider;
    }
    /// get time of the last update
    long getLastTime()
    {
        return lastTime;
    };
    /// get time of the actual update
    long getActualTime()
    {
        return actualTime;
    };

    /// register a rootMenu
    void registerMenu(coMenu *, bool force = false);
    /// unregister a rootMenu
    void unregisterMenu(coMenu *);
    /// a submenu has been closed select the parentMenu
    void closedMenu(coMenu *oldMenu, coMenu *newMenu);
    /// a item has been selected via ray
    void selectItem(coMenuItem *item, coMenu *parentMenu);
    /// opend a submenu via pickray
    void openedSubMenu(coGenericSubMenuItem *item, coMenu *subMenu);
    /// set the values of the valuator (new flystick)
    void update(float x, float y, long timeStamp = -1);
    /// set the values of the valuator (old flystick)
    void update(int x, int y, long timeStamp = -1);

    /// set the values
    void newUpdate(float x, float y, int button, long timeStamp = -1);
    /// set the values
    void newUpdate(int x, int y, int button, long timeStamp = -1);

    void setAttachment(int att)
    {
        attachment = att;
    };
};
}

#endif
