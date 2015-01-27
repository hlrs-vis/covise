/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _AK_SUB_MENU_H_
#define _AK_SUB_MENU_H_
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS AkSubMenu
//
// This class represents a menu in the new menu system
//
// Initial version: 2003-06-23 [we]
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2001 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:
//

#include <util/common.h>

#include <OpenVRUI/coMenuItem.h>
#include <OpenVRUI/coMenu.h>
#include <OpenVRUI/coMenuChangeListener.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/coToolboxMenuItem.h>

namespace vrui
{
class coMenuItem;
class coLabelSubMenuToolboxItem;
class coRowMenu;
class coToolboxMenu;
class coSubMenuItem;
}

using namespace vrui;

/**
 * Class for AK-VR style Sub-menus
 *
 */
class AkSubMenu : public coMenuListener,
                  public coMenuChangeListener,
                  public coMenuFocusListener
{
public:
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++ Constructors / Destructor
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    /** Constructor
       *  @param  name        Menu name = title of Menu, trigger & toolbar button
       *  @param triggerMenu  menu level above
       *  @param triggerItem  button in menu level above
       *  @param toolbar      the toolbar
       */
    AkSubMenu(const char *name, AkSubMenu *triggerMenu,
              coButtonMenuItem *triggerItem, coToolboxMenu *toolbar);

    /** Cloning Constructor
       *  @param  cloneMenu   Menu to clone
       *  @param triggerMenu  menu level above
       *  @param triggerItem  button in menu level above
       *  @param toolbar      the toolbar
       */
    AkSubMenu(coMenu *cloneMenu, AkSubMenu *triggerMenu,
              coButtonMenuItem *triggerItem, coToolboxMenu *toolbar);

    /// Destructor : virtual in case we derive objects
    virtual ~AkSubMenu();

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++ Operations
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    /** add an item to this menu
       *  @param    item    item to be added
       *  @param    pos     position in menu, -1 = at end
       */
    void add(coMenuItem *item, int pos);

    /** delete an item from this menu
       *  @param    item    item to be deleted
       */
    void remove(coMenuItem *item);

    /// add this menu to the toolbar
    void addToToolbar();

    /// monitor this menu's events
    void monitorMenu(coMenu *menu);

    /// remove all items after my toolbar button
    void cleanToolbarRest();

    /// de-activate my menu
    void deactivate();

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++ Virtual Functions from coMenuFocusListener
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    /// react on menu focus events
    virtual void focusEvent(bool focus, coMenu *menu);

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++ Virtual Functions from coMenuListener
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    /// react on menu events
    virtual void menuEvent(coMenuItem *menuItem);

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++ Virtual Functions from coMenuChangeListener
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    /// react on new menu items in menus we did not create ourself
    virtual void itemAdded(coMenu *menu, coMenuItem *item, int pos);

    /// react on removed menu items in menus we did not create ourself
    virtual void itemRemoved(coMenu *menu, coMenuItem *item);

    // react on changed menu items in menus we did not create ourself
    virtual void itemRenamed(coMenu *menu, coMenuItem *item);

    /** the subMenuItem controlling this menu was set
         @param subMenu pointer to the added submenu
      */
    virtual void subMenuAdded(coMenu *submenu, coGenericSubMenuItem *m);

    /** a new submenu was deleted with this menu as parent
         @param subMenu pointer to the deleted submenu
      */
    virtual void subMenuDeleted(coMenu *submenu);

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++ Attribute request/set functions
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    coMenu *getMenu()
    {
        return theMenu_;
    };

    const char *getName()
    {
        return name_.c_str();
    }

    void rename(const char *newName);

protected:
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++  Attributes
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    /// Name = Title of the menu
    string name_;

    /// Menu holding the trigger button
    AkSubMenu *triggerMenu_;

    /// Trigger button for this menu: either in parent menu
    coButtonMenuItem *triggerItem_;

    /// Button to be put in the toolbar
    coLabelSubMenuToolboxItem *toolbarButton_;

    /// Twin of button required for RowMenu startup
    coSubMenuItem *toolbarButtonTwin_;

    /// Our own menu
    //coRowMenu *theMenu_;
    coMenu *theMenu_;

    /// The toolbar we work with
    coToolboxMenu *toolbar_;

    /// List my sub-Menus
    typedef map<coMenu *, AkSubMenu *> SubMenuMap;
    SubMenuMap subMenuMap_;

    /// Map: toolbar items fired by my menu buttons, e.g. Sliders
    typedef map<coMenuItem *, coToolboxMenuItem *> InlayMap;
    InlayMap inlayMap_;

    /// If the user configures AutoClose, we automatically close menus
    static const bool autoClose_;

private:
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++  Internally used functions
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    /** Copy a menu item and setup as twin. Only implemented for
       *  types that can be directly copied into toolbar menus, so
       *  not for subMenuItems of Sliders
       *  @param   item    item to be copied
       *  @return  true if ok, false otherwise
       */
    bool copyItem(coMenuItem *copyitem,
                  coMenuItem *&menuItem,
                  coToolboxMenuItem *&toolItem);

    /** Slider copy: returns a pushbutton for my menu and a slider
       *  to be added into the toolbar
       *  @param   item        item to be copied
       *  @param   resSlider   slider for menu bar
       */

    /// monitor this item's events
    void monitorSubMenuItem(coSubMenuItem *item);

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++  prevent auto-generated bit copy routines by default
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    /// Copy-Constructor: NOT IMPLEMENTED, checked by assert
    AkSubMenu(const AkSubMenu &);

    /// Assignment operator: NOT IMPLEMENTED, checked by assert
    AkSubMenu &operator=(const AkSubMenu &);

    /// Default constructor: NOT IMPLEMENTED, checked by assert
    AkSubMenu();

    typedef map<coMenu *, coMenu *> MonitoredMenus;
    MonitoredMenus monitoredMenus_;

    typedef map<coSubMenuItem *, coSubMenuItem *> MonitoredSubMenuItems;
    MonitoredSubMenuItems monitoredSubMenuItems_;
};
#endif
