/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/coGenericSubMenuItem.h>

#include <OpenVRUI/coMenu.h>

namespace vrui
{

/** Constructor.
  @param name           displayed menu item text
*/
coGenericSubMenuItem::coGenericSubMenuItem(coMenuItem *container)
{
    subMenu = 0;
    container_ = container;
    open = false;
}

/// Destructor.
coGenericSubMenuItem::~coGenericSubMenuItem()
{
    if (subMenu)
    {
        subMenu->setSubMenuItem(0);
    }
}

/** set the submenu to handle.
  @param menu submenu
*/
void coGenericSubMenuItem::setMenu(coMenu *menu)
{
    subMenu = menu;

    if (subMenu)
    {
        subMenu->setSubMenuItem(this);
        if (container_)
            subMenu->setParent(container_->getParentMenu());
    }
}

/** Get menu state.
  @return true if the submenu is open
*/
bool coGenericSubMenuItem::isOpen() const
{
    return open;
}

coMenu *coGenericSubMenuItem::getMenu()
{
    return subMenu;
}

}
