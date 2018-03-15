/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/coRowMenuItem.h>

#include <OpenVRUI/coColoredBackground.h>
#include <OpenVRUI/coLabel.h>
#include <OpenVRUI/coMenu.h>
#include <OpenVRUI/coMenuContainer.h>
#include <OpenVRUI/coUIElement.h>
#include <OpenVRUI/sginterface/vruiRendererInterface.h>

using namespace std;

namespace vrui
{

const int coRowMenuItem::LEFTMARGIN = 30;

/** Constructor. Creates a new menu item.
  @param labelString symbolic name and text to appear on label
*/
coRowMenuItem::coRowMenuItem(const string &labelString)
    : coMenuItem(labelString)
{
    background = new coColoredBackground(coUIElement::ITEM_BACKGROUND_NORMAL, coUIElement::ITEM_BACKGROUND_HIGHLIGHTED, coUIElement::ITEM_BACKGROUND_DISABLED);
    background->setXAlignment(coUIContainer::MIN);
    container = new coMenuContainer();
    container->setVgap(10);
    container->setHgap(15);
    background->addElement(container);
    label = new coLabel(labelString);
}

/** Constructor. Creates a new menu item.
  @param symbolicName symbolic name
  @param labelString text to appear on label
*/
coRowMenuItem::coRowMenuItem(const string &symbolicName, const string &labelString)
    : coMenuItem(symbolicName)
{
    background = new coColoredBackground(coUIElement::ITEM_BACKGROUND_NORMAL, coUIElement::ITEM_BACKGROUND_HIGHLIGHTED, coUIElement::ITEM_BACKGROUND_DISABLED);
    background->setXAlignment(coUIContainer::MIN);
    container = new coMenuContainer();
    container->setVgap(10);
    background->addElement(container);
    label = new coLabel(labelString);
}

/// Destructor. Removes this menu item from the parent menu.
coRowMenuItem::~coRowMenuItem()
{

    if (background->getParent())
    {
        background->getParent()->removeElement(background);
    }

    if (label->getParent())
    {
        label->getParent()->removeElement(label);
    }
    if (myMenu)
        myMenu->remove(this);
    delete background;
    delete label;
    delete container;
}

/** Set a new label for the menu item. The label
  must already be of type coLabel.
  @param newLabel
*/
void coRowMenuItem::setLabel(coLabel *newLabel)
{
    label = newLabel;
}

/** Get the current label.
  @return label
*/
coLabel *coRowMenuItem::getLabel()
{
    return label;
}

void coRowMenuItem::setLabel(const string &labelString)
{
    label->setString(labelString);
}

/// return the actual UI Element that represents this menu.
coUIElement *coRowMenuItem::getUIElement()
{
    return background;
}

const char *coRowMenuItem::getClassName() const
{
    return "coRowMenuItem";
}

bool coRowMenuItem::isOfClassName(const char *classname) const
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
void coRowMenuItem::setActive(bool a)
{
    active_ = a;
    background->setHighlighted(false);
}

void coRowMenuItem::selected(bool selected)
{
    //if (vruiRendererInterface::the()->isJoystickActive())
    background->setHighlighted(selected);
}
}
