/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_SUBMENU_H
#define CO_SUBMENU_H

#include <util/coTypes.h>

/** Menu item which can be used to open and close submenus
    menu events are generated when the user opens or closes the submenu
*/
namespace vrui
{

class coMenu;
class coMenuItem;

class OPENVRUIEXPORT coGenericSubMenuItem
{
protected:
    coMenu *subMenu; ///< the subMenu which is opened and closed by thes button
    bool open; ///< current state of the menu (open or closed)
    coMenuItem *container_; ///< containing menu item
public:
    coGenericSubMenuItem(coMenuItem *container);
    virtual ~coGenericSubMenuItem();

    virtual void setMenu(coMenu *menu);

    virtual void closeSubmenu() = 0;
    virtual void openSubmenu() = 0;
    virtual void positionSubmenu() = 0;
    bool isOpen() const;

    coMenu *getMenu(); ///< get my subMenu
private:
    coGenericSubMenuItem();
};
}
#endif
