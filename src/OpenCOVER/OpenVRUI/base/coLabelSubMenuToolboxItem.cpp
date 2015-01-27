/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/coLabelSubMenuToolboxItem.h>
#include <OpenVRUI/coMenuContainer.h>

#include <OpenVRUI/coLabel.h>

namespace vrui
{

/** Constructor.
  @param name           displayed menu item text
*/
coLabelSubMenuToolboxItem::coLabelSubMenuToolboxItem(const std::string &symbolicName)
    : coSubMenuToolboxItem(symbolicName)
{
    label = new coLabel();
    label->setString(symbolicName);

    menuContainer->addElement(label);

    setAttachment(coUIElement::BOTTOM);
}

/// Destructor.
coLabelSubMenuToolboxItem::~coLabelSubMenuToolboxItem()
{
    delete label;
}

void coLabelSubMenuToolboxItem::setAttachment(int newatt)
{
    // update own icon orientation
    // and type
    switch (newatt)
    {
    case coUIElement::LEFT:
        subMenuIcon->setRotation(0.0);
        menuContainer->setOrientation(coMenuContainer::HORIZONTAL);
        // label,icon
        menuContainer->removeElement(label);
        menuContainer->insertElement(label, 0);
        break;

    case coUIElement::BOTTOM:
        subMenuIcon->setRotation(90.0);
        menuContainer->setOrientation(coMenuContainer::VERTICAL);
        // icon,label
        menuContainer->removeElement(label);
        menuContainer->insertElement(label, 1);
        break;

    case coUIElement::RIGHT:
        subMenuIcon->setRotation(180.0);
        menuContainer->setOrientation(coMenuContainer::HORIZONTAL);
        // icon,label
        menuContainer->removeElement(label);
        menuContainer->insertElement(label, 1);
        break;

    case coUIElement::TOP:
        subMenuIcon->setRotation(270.0);
        menuContainer->setOrientation(coMenuContainer::VERTICAL);
        //label,icon
        menuContainer->removeElement(label);
        menuContainer->insertElement(label, 0);
        break;
    }

    // copy new attachment
    attachment = newatt;
}

const char *coLabelSubMenuToolboxItem::getClassName() const
{
    return "coLabelSubMenuToolboxItem";
}

bool coLabelSubMenuToolboxItem::isOfClassName(const char *classname) const
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
            return coSubMenuToolboxItem::isOfClassName(classname);
        }
    }

    // nobody is NULL
    return false;
}

void coLabelSubMenuToolboxItem::setLabel(const std::string &labelString)
{
    label->setString(labelString);
}
}
