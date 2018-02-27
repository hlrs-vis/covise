/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_MENU_H
#define CO_MENU_H

#include <OpenVRUI/coUpdateManager.h>
#include <OpenVRUI/coMenuItem.h>
#include <OpenVRUI/coAction.h>

#include <util/coTypes.h>

#include <list>
#include <vector>
#include <string>

namespace vrui
{

class coMenuItem;
class coGenericSubMenuItem;
class coCombinedButtonInteraction;

class vruiMatrix;

typedef std::vector<coMenuItem *> coMenuItemVector;

/// Action listener for events triggered by any coMenuItem.
class OPENVRUIEXPORT coMenuFocusListener
{
public:
    virtual ~coMenuFocusListener()
    {
    }
    /** Action listener for events triggered by coMenuItem.
          This method is called whenever a coMenuItem was selected.
          @param menuItem pointer to menu item which triggered this event
      */
    virtual void focusEvent(bool focus, coMenu *menu) = 0;
};

/** This class provides a general menu for use in 3D space. It does not
  impose any restrictions on the layout of the menu items. A possible layout
  is defined in the subclass coRowMenu.
*/
class OPENVRUIEXPORT coMenu : public coAction, public coUpdateable
{
public:
    coMenu(coMenu *parentMenu, const std::string &name);
    virtual ~coMenu();

    ///< Sets the matrix which defines location and orientation
    virtual void setTransformMatrix(vruiMatrix *matrix) = 0;
    /// of the menu.
    ///< Sets the matrix which defines location orientation and scale of the menu and the scale factor included in the matrix.
    virtual void setTransformMatrix(vruiMatrix *matrix, float scale) = 0;
    /// Returns the matrix which defines location and orientation of the menu
    //       virtual vruiMatrix* getTransformMatrixLocal() = 0;
    /// Sets an offset matrix which is added to the current transformation
    //       virtual void setOffsetMatrix(vruiMatrix * matrix) = 0;

    virtual void setScale(float s) = 0; ///< Sets the menu size by applying a scaling factor to the default size.
    virtual float getScale() const = 0; ///< Gets the current scaling factor.
    virtual void add(coMenuItem *item);
    virtual void insert(coMenuItem *item, int location);
    virtual void setMenuListener(coMenuFocusListener *listener);
    virtual int hit(vruiHit *hit);
    virtual void miss();
    virtual coUIElement *getUIElement();
    virtual void setVisible(bool);
    /// reposition all submenus, for example after the main menu was repositioned with setTransform matrix
    virtual void positionAllSubmenus();
    virtual void remove(coMenuItem *item);
    virtual void removeAll();
    virtual int getItemCount() const;
    virtual void closeMenu();

    virtual void selected(bool select); ///< Menu is selected via joystick
    virtual void doActionPress(); ///< Action is called via joystick
    virtual void doActionRelease(); ///< Action is called via joystick
    virtual void doSecondActionPress(); ///< Action is called via joystick
    virtual void doSecondActionRelease(); ///< Action is called via joystick
    virtual void makeVisible(coMenuItem *); ///< makes the item visible for joystick interaction

    virtual vruiTransformNode *getDCS() = 0; ///< Get Transformation Node

    virtual bool isInteractionActive() const;
    coCombinedButtonInteraction *getInteraction() const;
    coCombinedButtonInteraction *getMoveInteraction() const;

    // Functions to Show and Hide the Menu
    virtual void show() = 0; ///< makes the menu visible
    virtual void hide() = 0; ///< hides this menu

    enum /// Menu show modes
    {
        MENU_ONOFF = 0,
        MENU_SLIDE
    };

    void setShowMode(int mode);
    int getShowMode() const;

    virtual void setSubMenuItem(coGenericSubMenuItem *item);
    coGenericSubMenuItem *getSubMenuItem();

    /// get the Element's classname
    virtual const char *getClassName() const;
    /// check if the Element or any ancestor is this classname
    virtual bool isOfClassName(const char *) const;

    /// close all menus: optionally leave one open
    void closeAllOtherMenus(coMenuItem *leave = 0);

    /// remove all items behind a given one
    void removeAllAfter(coMenuItem *last);

    /// check whether menu is currently visible
    bool isVisible() const;

    /// get my menu's name
    const char *getName() const;

    // get my menu's items
    coMenuItemVector getAllItems();

    coMenuItem *getItemByName(const char *name);

    // check if menu contains item, return position in menu or -1
    int index(const coMenuItem *item);

    // test change parent
    void setParent(coMenu *newParent)
    {
        parent = newParent;
    };
    const coMenu *getParent()
    {
        return parent;
    };

    virtual void setAttachment(int att);
    virtual int getAttachment() const
    {
        return attachment_;
    };

    bool wasMoved() const;
    void setMoved(bool flag);

protected:
    std::list<coMenuItem *> items; ///< list of menu items which can be accessed from this menu
    coMenu *parent; ///< parent menu, NULL if this is the topmost menu

    coGenericSubMenuItem *myMenuItem;
    coMenuFocusListener *listener; ///< menu event listener, triggered on menu selection

    int showMode; ///< how to show and hide this Menu
    bool visible;
    std::string name;
    coCombinedButtonInteraction *interaction;
    coCombinedButtonInteraction *moveInteraction;
    coCombinedButtonInteraction *wheelInteraction; ///< interaction for preventing accidental global wheel interaction

    float scale_;
    vruiMatrix *matrix_;

    int attachment_;
    bool wasMoved_ = false;
};
}
#endif
