/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/coIconSubMenuToolboxItem.h>
#include <OpenVRUI/coMenuContainer.h>

#include <OpenVRUI/coToggleButtonGeometry.h>

#include <OpenVRUI/util/vruiLog.h>

namespace vrui
{

/** Constructor.
  @param name           displayed menu item text
*/
coIconSubMenuToolboxItem::coIconSubMenuToolboxItem(const std::string &symbolicName)
    : coSubMenuToolboxItem(symbolicName)
{
    myButton = new coToggleButton(new coToggleButtonGeometry(symbolicName), this);

    menuContainer->addElement(myButton);
    menuContainer->setAlignment(coMenuContainer::CENTER);
    // Problem: wir wollen eigentlich '0' oder '2' je nach Ausrichtung.
    // aber '0' zentriert :(
    menuContainer->setNumAlignedMin(1);

    setAttachment(coUIElement::BOTTOM);
}

/// Destructor.
coIconSubMenuToolboxItem::~coIconSubMenuToolboxItem()
{
    delete myButton;
}

void coIconSubMenuToolboxItem::setAttachment(int newatt)
{
    float l_button = 0.0;

    // update own icon orientation
    // and type
    switch (newatt)
    {
    case coUIElement::LEFT:
        subMenuIcon->setRotation(0.0);
        menuContainer->setOrientation(coMenuContainer::HORIZONTAL);
        // label,icon
        menuContainer->removeElement(myButton);
        menuContainer->insertElement(myButton, 0);
        // adjust Icon Size
        l_button = 100.0f - subMenuIcon->getWidth() - menuContainer->getHgap();
        break;

    case coUIElement::BOTTOM:
        subMenuIcon->setRotation(90.0);
        menuContainer->setOrientation(coMenuContainer::VERTICAL);
        // icon,label
        menuContainer->removeElement(myButton);
        menuContainer->insertElement(myButton, 1);
        // adjust Icon Size
        l_button = 100.0f - subMenuIcon->getHeight() - menuContainer->getVgap();
        break;

    case coUIElement::RIGHT:
        subMenuIcon->setRotation(180.0);
        menuContainer->setOrientation(coMenuContainer::HORIZONTAL);
        // icon,label
        menuContainer->removeElement(myButton);
        menuContainer->insertElement(myButton, 1);
        // adjust Icon Size
        l_button = 100.0f - subMenuIcon->getWidth() - menuContainer->getHgap();
        break;

    case coUIElement::TOP:
        subMenuIcon->setRotation(270.0);
        menuContainer->setOrientation(coMenuContainer::VERTICAL);
        //label,icon
        menuContainer->removeElement(myButton);
        menuContainer->insertElement(myButton, 0);
        // adjust Icon Size
        l_button = 100.0f - subMenuIcon->getHeight() - menuContainer->getVgap();
        break;

    default:
        VRUILOG("coIconSubMenuToolboxItem::setAttachment: l_button is used uninitialized");
        break;
    }

    myButton->setSize(l_button, l_button, 1.0);

    // copy new attachment
    attachment = newatt;
}

const char *coIconSubMenuToolboxItem::getClassName() const
{
    return "coIconSubMenuToolboxItem";
}

bool coIconSubMenuToolboxItem::isOfClassName(const char *classname) const
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
            return coToolboxMenuItem::isOfClassName(classname);
        }
    }

    // nobody is NULL
    return false;
}
}
