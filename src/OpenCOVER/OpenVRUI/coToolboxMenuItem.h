/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_TOOLBOXMENUITEM_H
#define CO_TOOLBOXMENUITEM_H

#include <OpenVRUI/coMenuItem.h>
#include <string>

namespace vrui
{

class coMenuContainer;
class coColoredBackground;

/** This is the base class of all toolbox menu items.
  It provides a container for menu elements and a label string.
*/

class OPENVRUIEXPORT coToolboxMenuItem
    : public coMenuItem
{
protected:
    coMenuContainer *menuContainer; ///< container to store menu elements
    coColoredBackground *background; ///< menu item background which changes its color when menu item is selected

public:
    coToolboxMenuItem(const std::string &name);
    virtual ~coToolboxMenuItem();
    virtual coUIElement *getUIElement();

    /// get the Element's classname
    virtual const char *getClassName() const;
    /// check if the Element or any ancestor is this classname
    virtual bool isOfClassName(const char *) const;

    /// activates or deactivates the item
    virtual void setActive(bool a);
    virtual void doSecondActionRelease(); ///< Action is called via joystick
};
}
#endif
