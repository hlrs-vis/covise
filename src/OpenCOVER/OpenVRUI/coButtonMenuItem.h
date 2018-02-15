/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_BUTTON_MENUITEM_H
#define CO_BUTTON_MENUITEM_H

#include <OpenVRUI/coRowMenuItem.h>
#include <OpenVRUI/coDragNDrop.h>
#include <OpenVRUI/coTexturedBackground.h>

#include <OpenVRUI/coAction.h>

namespace vrui
{

class vruiHit;

class coBackground;
class coRowContainer;

#include <string>

/** This class defines a menu item which can be used as a button to trigger an action.
  The events are processed by coAction.
*/
class OPENVRUIEXPORT coButtonMenuItem : public coRowMenuItem, public coAction, public coDragNDrop, public coTexturedBackgroundActor
{
private:
    bool lastContent; ///< last content sent by twin
    bool pressed; ///< true = input device button press happened on this menu item
    std::string iconName_; ///< name of the icon

protected:
    coBackground *space; ///< blank space left of label text, used as a margin
    coTexturedBackground *icon;

public:
    coButtonMenuItem(const std::string &);
    coButtonMenuItem(const std::string &name, const std::string &nameIcon, float texSizeX = 0, float texSizeY = 0); ///< Button with icon
    virtual ~coButtonMenuItem();
    int hit(vruiHit *hit);
    void miss();

    /// get the Element's classname
    virtual const char *getClassName() const;
    /// check if the Element or any ancestor is this classname
    virtual bool isOfClassName(const char *) const;

    virtual bool dropOperation(coDragNDrop * /*item*/)
    {
        return false;
    }

    virtual const std::string &getIconName()
    {
        return iconName_;
    };
    virtual float getIconSizeX();
    virtual float getIconSizeY();
    virtual void setIconSizeX(float x);
    virtual void setIconSizeY(float y);

    /// functions activates or deactivates the item
    virtual void setActive(bool a);

    virtual void selected(bool select); ///< MenuItem is selected via joystick
    virtual void doActionRelease(); ///< ActionPress is called via joystick
};
}
#endif
