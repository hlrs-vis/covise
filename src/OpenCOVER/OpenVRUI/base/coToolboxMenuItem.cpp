/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <OpenVRUI/coToolboxMenuItem.h>

#include <OpenVRUI/coColoredBackground.h>
#include <OpenVRUI/coMenu.h>

#include <OpenVRUI/coMenuContainer.h>

using namespace std;

namespace vrui
{

/** Constructor. Creates a new menu item.
  @param symbolicName symbolic name and text to appear on label
*/

coToolboxMenuItem::coToolboxMenuItem(const string &symbolicName)
    : coMenuItem(symbolicName)
{
    background = new coColoredBackground(coUIElement::ITEM_BACKGROUND_NORMAL, coUIElement::ITEM_BACKGROUND_HIGHLIGHTED, coUIElement::ITEM_BACKGROUND_DISABLED);
    background->setXAlignment(coUIContainer::MIN);

    menuContainer = new coMenuContainer(coMenuContainer::VERTICAL);
    menuContainer->setVgap(10);
    menuContainer->setNumAlignedMin(1);

    background->addElement(menuContainer);
}

/// Destructor. Removes this menu item from the parent menu.
coToolboxMenuItem::~coToolboxMenuItem()
{
    if (myMenu)
        myMenu->remove(this);
    delete background;
    delete menuContainer;
}

/// return the actual UI Element that represents this menu.
coUIElement *coToolboxMenuItem::getUIElement()
{
    return background;
}

const char *coToolboxMenuItem::getClassName() const
{
    return "coToolboxMenuItem";
}

bool coToolboxMenuItem::isOfClassName(const char *classname) const
{
    // paranoia makes us mistrust the string library and check for NULL.
    if (classname && getClassName())
    {
        // check for identity
        if (!strcmp(classname, getClassName()))
        { // we are the one
            return true;
        }
        else
        { // we are not the wanted one. Branch up to parent class
            return coMenuItem::isOfClassName(classname);
        }
    }

    // nobody is NULL
    return false;
}

/// activates or deactivates the item and deactivates the highlight
void coToolboxMenuItem::setActive(bool a)
{
    active_ = a;
    background->setHighlighted(false);
}

void coToolboxMenuItem::doSecondActionRelease()
{
    doActionRelease();
}
}
