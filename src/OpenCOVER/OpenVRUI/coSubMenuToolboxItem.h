/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_SUBMENU_TOOLBOXITEM_H
#define CO_SUBMENU_TOOLBOXITEM_H

#include <OpenVRUI/coAction.h>
#include <OpenVRUI/coToolboxMenuItem.h>
#include <OpenVRUI/coGenericSubMenuItem.h>
#include <OpenVRUI/coRotButton.h>

#include <string>

namespace vrui
{

class vruiHit;

/** Menu item which can be used to open and close submenus
    menu events are generated when the user opens or closes the submenu
    In difference to a normal Menu subwindow, orientation (attachement) changes
    have to be handled properly.
*/

class OPENVRUIEXPORT coSubMenuToolboxItem
    : public coToolboxMenuItem,
      public coGenericSubMenuItem,
      public coRotButtonActor,
      public coAction
{
private:
    bool pressed; ///< true if ACTION_BUTTON was pressed on this menu item

protected:
    coRotButton *subMenuIcon; ///< arrow button which is used for interaction
    int attachment;

public:
    coSubMenuToolboxItem(const std::string &name);
    virtual ~coSubMenuToolboxItem();

    int hit(vruiHit *hit);
    void miss();

    virtual void selected(bool select); ///< MenuItem is selected via joystick
    virtual void doActionRelease(); ///< Action is called via joystick

    virtual void closeSubmenu();
    virtual void openSubmenu();

    virtual void setAttachment(int attachment) = 0;

    /// sets the icon states
    virtual bool updateContentBool(bool);

    /// moves the pointed submenu to new position and scale
    virtual bool updateContentPointer(void *);

    /// functions activates or deactivates the item
    virtual void setActive(bool a);

    /// get the Element's classname
    virtual const char *getClassName() const;
    /// check if the Element or any ancestor is this classname
    virtual bool isOfClassName(const char *) const;

    void positionSubmenu();
};
}
#endif
