/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_MENU_CHANGE_LISTENER_H_
#define _CO_MENU_CHANGE_LISTENER_H_
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS coMenuChangeListener
//
// This class @@@
//
// Initial version: 2003-06-27 [we]
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2001 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:
//
#include <util/coTypes.h>
/**
 * Class to monitot coMenu classes externally
 *
 */

namespace vrui
{

/// Listener triggered whenever a Menu is changed.
class OPENVRUIEXPORT coMenuChangeListener
{
public:
    virtual ~coMenuChangeListener()
    {
    }

    enum
    {
        RENAME,
        STATE
    };

    /** Action listener for events triggered by coMenuItem.
          This method is called after a coMenuItem was added to the menu.
          @param menu     pointer to menu
          @param menuItem pointer to menu item which triggered this event
          @param pos      position in menu, -1 = at end
      */
    virtual void itemAdded(coMenu * /*menu*/, coMenuItem * /*item*/, int /*pos*/)
    {
    }

    /** Action listener for events triggered by coMenuItem.
          This method is called before a coMenuItem is removed from the menu.
          @param menu     pointer to menu
          @param menuItem pointer to menu item which triggered this event
      */
    virtual void itemRemoved(coMenu * /*menu*/, coMenuItem * /*item*/)
    {
    }

    /** Action listener for events triggered by coMenuItem.
          This method is called when an item changed internal state
          @param menu     pointer to menu
          @param menuItem pointer to menu item which triggered this event
      */
    virtual void itemRenamed(coMenu * /*menu*/, coMenuItem * /*item*/)
    {
    }

    /** the subMenuItem controlling this menu was set
          @param subMenu  pointer to the added submenu
          @param button   menuItem responsible for doing this
      */
    virtual void subMenuAdded(coMenu * /*submenu*/, coGenericSubMenuItem * /*button*/)
    {
    }

    /** a new submenu was deleted with this menu as parent
          @param subMenu pointer to the deleted submenu
      */
    virtual void subMenuDeleted(coMenu * /*submenu*/)
    {
    }
};
}
#endif
