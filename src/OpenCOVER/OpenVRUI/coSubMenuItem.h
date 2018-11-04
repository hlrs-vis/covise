/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_SUB_MENUITEM_H
#define CO_SUB_MENUITEM_H

#include <OpenVRUI/coAction.h>
#include <OpenVRUI/coRowMenuItem.h>
#include <OpenVRUI/coRotButton.h>
#include <OpenVRUI/coGenericSubMenuItem.h>
#include <string>

namespace vrui
{

class coRotButton;
class coBackground;
class coCombinedButtonInteraction;

class vruiHit;

/** Menu item which can be used to open and close submenus
    menu events are generated when the user opens or closes the submenu
*/
class OPENVRUIEXPORT coSubMenuItem
    : public coRowMenuItem,
      public coGenericSubMenuItem,
      public coRotButtonActor,
      public coAction
{
private:
    bool pressed; ///< true if ACTION_BUTTON was pressed on this menu item

protected:
    coRotButton *subMenuIcon; ///< arrow button which is used for interaction
    coBackground *space; ///< blank space left of label text, used as a margin
    int attachment;
    coMenuItem *secondaryItem; ///< item that is triggered on right-button clicks
    coCombinedButtonInteraction *preventMoveInteraction;

public:
    coSubMenuItem(const std::string &name);
    virtual ~coSubMenuItem();

    int hit(vruiHit *hit);
    void miss();

    virtual void selected(bool select); ///< MenuItem is selected via joystick
    virtual void doActionRelease(); ///< Action is called via joystick
    virtual void doSecondActionRelease(); ///< second Action for Item

    virtual void closeSubmenu();
    virtual void openSubmenu();
    virtual void positionSubmenu();

    virtual void buttonEvent(coRotButton *button);

    /// Set the attachment of the submenu respective to the menu item.
    virtual void setAttachment(int attachment);

    /// Get the attachment.
    virtual int getAttachment()
    {
        return (attachment);
    };

    /// functions activates or deactivates the item
    virtual void setActive(bool a);

    /// get the Element's classname
    virtual const char *getClassName() const;
    /// check if the Element or any ancestor is this classname
    virtual bool isOfClassName(const char *) const;

    /// trigger doActionRelease (i.e. toggle) this item on XFORM button release
    void setSecondaryItem(coMenuItem *item);
};
}
#endif
