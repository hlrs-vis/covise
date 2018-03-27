/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_MENUITEM_H
#define CO_MENUITEM_H

#include <OpenVRUI/coUIElement.h>
#include <string>

namespace vrui
{

class coMenuContainer;
class coColoredBackground;
class coMenu;
class coLabel;
class coMenuItem;

/// Action listener for events triggered by any coMenuItem.
class OPENVRUIEXPORT coMenuListener
{
public:
    /** Action listener for events triggered by coMenuItem.
          This method is called whenever a coMenuItem was selected.
          @param menuItem pointer to menu item which triggered this event
      */
    virtual ~coMenuListener()
    {
    }
    virtual void menuEvent(coMenuItem *menuItem);
    virtual void menuPressEvent(coMenuItem *menuItem);
    virtual void menuReleaseEvent(coMenuItem *menuItem);
};

/** This is the base class of all menu items.
  It stores the action listener which is to be called when a menu item is
  selected by the user.
  @see coMenuListener
*/
class OPENVRUIEXPORT coMenuItem
{
protected:
    bool visible;
    coMenu *myMenu; ///< parent menu which contains
    ///< this menu item
    coMenuListener *listener; ///< menu event listener, triggered
    ///<  on menu selection
    std::string myName; ///< name of the item (internal
    ///< symbolic name, or displayed as label)

    bool active_; ///< flag if item is active

public:
    coMenuItem(const std::string &name);
    virtual ~coMenuItem();

    virtual void setMenuListener(coMenuListener *listener);
    virtual coMenuListener *getMenuListener();

    virtual void setParentMenu(coMenu *parent);
    virtual void setVisible(bool visible);
    virtual bool isVisible() const;
    virtual void setAttachment(int /*attachment*/)
    {
    } ///< sets the attachment border

    virtual const char *getName() const;
    virtual coUIElement *getUIElement();

    /// get the Element's classname
    virtual const char *getClassName() const;
    /// check if the Element or any ancestor is this classname
    virtual bool isOfClassName(const char *) const;

    /// get my parent menu
    coMenu *getParentMenu()
    {
        return myMenu;
    }

    /// set my name - items with labels must call this when changing label
    void setName(const std::string &newName, bool updateTwins = true);

    /// functions activates or deactivates the item
    virtual void setActive(bool a)
    {
        active_ = a;
    };
    ///return if item is active
    virtual bool getActive()
    {
        return active_;
    };

    /// this function may be overloaded - it is called by setName()
    virtual void setLabel(const std::string &newName);

    virtual void selected(bool select); ///< MenuItem is selected via joystick
    virtual void doActionPress(); ///< Action is called via joystick
    virtual void doActionRelease(); ///< Action is called via joystick
    virtual void doSecondActionPress(); ///< second Action for Item
    virtual void doSecondActionRelease(); ///< second Action for Item
};
}
#endif
