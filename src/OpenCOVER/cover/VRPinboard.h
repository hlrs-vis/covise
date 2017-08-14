/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __VR_PINBOARD_H
#define __VR_PINBOARD_H

/*! \file
 \brief  menu system

 \author Frank Foehl
 \author Uwe Woessner <woessner@hlrs.de> (based on VRUI)
 \author (C) 1996
         Computer Centre University of Stuttgart,
         Allmandring 30,
         D-70550 Stuttgart,
         Germany

 \date   20.08.1997
 */

#include <util/common.h>

#include <OpenVRUI/coMenu.h>

#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coButtonMenuItem.h>
namespace vrui
{
class coMenu;
class coRowMenuItem;
class coMenuItem;
class coCheckboxGroup;
class coToolboxMenu;
class coCheckboxGroup;
}

namespace opencover
{
class buttonSpecCell;
class VRPinboard;
class VRMenu;
class VRButton;

typedef void (*ButtonCallback)(void *calledClass, buttonSpecCell *spec);

extern COVEREXPORT vrui::coCheckboxGroup *groupPointerArray[100];

class COVEREXPORT buttonSpecCell /* button specifier struct */
{
public:
    char name[256];
    char *myMenu;
    int actionType; /* BUTTON_FUNCTION|_SWITCH|_SUBMENU|_SLIDER */
    ButtonCallback callback; /* function called on button state change */
    void *calledClass; /* class object where callback fun resides */
    float state; /* (0 | 1) for switches, or slider float value */
    float oldState; /* (0 | 1) for switches, or slider float value */
    int dragState; /*  PRESSED|DRAGGED|RELEASED */
    VRMenu *subMenu; /* menu invoked for submenu buttons */
    char subMenuName[256]; /* name of invoked submenu */
    int dashed; /* true => button gets a dash below */
    int group; /* group number (only 1 switch of a group can */
    /*   have state 1 at the same time) */
    /*   group=-1 means no group specified */
    float sliderMin; /* min and max states for sliders */
    float sliderMax;
    void *userData;
    buttonSpecCell();
    buttonSpecCell(const buttonSpecCell &); // copy constructor
    ~buttonSpecCell();
    buttonSpecCell &operator=(const buttonSpecCell &s);
    void setMenu(const char *n);
};

class COVEREXPORT VRPinboard : public vrui::coMenuListener
{

private:
    static VRPinboard *s_singleton;
    std::list<VRMenu *> menuList;
    // check if the entry belongs to the permanent entries
    int isPermanentEntry(const char *functionName);
    vrui::coRowMenu *quitMenu_;
    vrui::coButtonMenuItem *yesButton_;
    vrui::coButtonMenuItem *cancelButton_;
    void makeQuitMenu();
    void showQuitMenu();
    int num_perm_functions;

public:
    static VRPinboard *instance();
    enum
    {
        BTYPE_NAVGROUP = 0,
        BTYPE_TOGGLE = 1,
        BTYPE_SYNCGROUP = 2,
        BTYPE_FUNC = 3,
        BTYPE_SLIDER = 4,
        BTYPE_CUSTOMGROUP = 5
    };

    struct PinboardFunction
    {
        const char *functionName;
        int functionType;
        const char *defButtonName;
        const char *defMenuName;
        ButtonCallback callback;
        void *callbackClass;
        bool isInPinboard;
        char *customButtonName;
        char *customMenuName;

        PinboardFunction(const char *functionName,
                         int functionType,
                         const char *buttonName,
                         const char *menuName,
                         ButtonCallback callback,
                         void *callbackClass);
    };

    std::vector<PinboardFunction> functions;
    void addFunction(const char *functionName,
                     int functionType,
                     const char *buttonName,
                     const char *menuName,
                     ButtonCallback callback,
                     void *callbackClass);

    bool customPinboard;

    // Toolbox control
    vrui::coToolboxMenu *theToolbox;

    // create button, if menuName, create also submenu
    void makeButton(const char *functionName, const char *buttonName, const char *menuName, int
                                                                                                type,
                    ButtonCallback callback, void *inst, void *userData = NULL);

    // read entry in config file and copy the button name and menu name to customButtonNames...
    void readCustomEntry(const char *functionName);

    VRMenu *mainMenu;

    VRPinboard();

    ~VRPinboard();

    void configInteraction();

    VRButton *addButtonToMainMenu(buttonSpecCell *spec);

    void removeButtonFromMainMenu(const char *name);
    void removeButtonFromNamedMenu(const char *name, const char *Menuname);

    VRButton *addButtonToNamedMenu(buttonSpecCell *spec, const char *menuName);

    VRMenu *addMenu(const char *name, vrui::coMenu *myMenu);

    void removeMenuFromList(VRMenu *menu);

    VRMenu *namedMenu(const char *name);

    // set the appearance of the button
    bool setButtonState(const char *buttonName, float state);

    // call the button callback
    int callButtonCallback(const char *buttonName);

    // go through all buttons abd submenues and set state to 0
    void clearGroupState(int groupId);

    // return index of function
    int getIndex(const char *functionName);

    // return name
    const char *getCustomName(const char *functionName);

    // returns the VRUI menu item
    vrui::coRowMenuItem *getMenuItem(const char *name);

    // return string to filename
    const char *getFileName(const char *string);

    // return Button given by name
    VRButton *getButtonByName(const char *buttonName);

    virtual void menuEvent(vrui::coMenuItem *menuItem);

    static void quitCallback(void *sceneGraph, buttonSpecCell *spec);
    void hideQuitMenu();

};

class COVEREXPORT VRMenu : public vrui::coMenuFocusListener
{
private:
    char *name;

    std::list<VRButton *> buttonList;

public:
    VRMenu(const char *name);

    virtual ~VRMenu();
    virtual void focusEvent(bool focus, vrui::coMenu *menu);
    vrui::coMenu *myMenu;

    VRButton *addButton(buttonSpecCell *spec);

    void removeButton(const char *name);

    VRButton *namedButton(const char *name);
    int isNamedMenu(const char *name);
    vrui::coMenu *getCoMenu()
    {
        return myMenu;
    };
};

class COVEREXPORT VRButton : public vrui::coMenuListener
{

private:
    vrui::coRowMenuItem *myMenuItem;

public:
    VRButton(buttonSpecCell *spec, vrui::coMenu *myMenu);

    virtual ~VRButton();
    buttonSpecCell spec; /* button specification */

    void setState(float state);
    int isNamedButton(const char *name);
    void menuEvent(vrui::coMenuItem *);
    void menuReleaseEvent(vrui::coMenuItem *);
    void setText(const char *s);

    // return the VRUI menu item
    vrui::coRowMenuItem *getMenuItem();
};
}
#endif
