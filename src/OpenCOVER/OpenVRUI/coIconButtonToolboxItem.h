/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_ICONBUTTON_TOOLBOXITEM_H
#define CO_ICONBUTTON_TOOLBOXITEM_H

#include <OpenVRUI/coToolboxMenuItem.h>
#include <OpenVRUI/coAction.h>

#include <OpenVRUI/coButton.h>

#include <string>

namespace vrui
{

/** This class defines a toolbox item which can be used as a button to trigger an action.
 The event states are fed to the Twin Item (actual state: on, off). Additionally specified own menuListeners are performed _after_ the twin item! The button works as 'key' button not as 'toggle' button.
*/
class OPENVRUIEXPORT coIconButtonToolboxItem
    : public coToolboxMenuItem,
      public coButtonActor,
      public coAction
{
private:
    bool pressed; ///< true if ACTION_BUTTON was pressed on this menu item

protected:
    coPushButton *myButton;
    vruiNode *lastHit;

public:
    coIconButtonToolboxItem(const std::string &);
    virtual ~coIconButtonToolboxItem();
    virtual int hit(vruiHit *);
    virtual void miss();

    virtual void selected(bool select); ///< MenuItem is selected via joystick
    virtual void doActionRelease(); ///< Action is called via joystick

    // not really used but there because of coButton Constructor :-/
    virtual void buttonEvent(coButton *)
    {
    }

    /// get the Element's classname
    virtual const char *getClassName() const;
    /// check if the Element or any ancestor is this classname
    virtual bool isOfClassName(const char *) const;

    /// functions activates or deactivates the item
    virtual void setActive(bool a);
};
}
#endif
