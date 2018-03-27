/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <OpenVRUI/coSubMenuToolboxItem.h>

#include <OpenVRUI/coColoredBackground.h>
#include <OpenVRUI/coFlatButtonGeometry.h>
#include <OpenVRUI/coMenu.h>
#include <OpenVRUI/coMenuContainer.h>
#include <OpenVRUI/coRotButton.h>

#include <OpenVRUI/sginterface/vruiButtons.h>
#include <OpenVRUI/sginterface/vruiHit.h>
#include <OpenVRUI/sginterface/vruiIntersection.h>
#include <OpenVRUI/sginterface/vruiMatrix.h>
#include <OpenVRUI/sginterface/vruiRendererInterface.h>
#include <OpenVRUI/sginterface/vruiTransformNode.h>

#include <OpenVRUI/util/vruiLog.h>

using namespace std;

namespace vrui
{

/** Constructor.
  @param name           displayed menu item text
*/
coSubMenuToolboxItem::coSubMenuToolboxItem(const string &symbolicName)
    : coToolboxMenuItem(symbolicName)
    , coGenericSubMenuItem(this)
    , pressed(false)
{
    subMenuIcon = new coRotButton(new coFlatButtonGeometry("UI/submenu"), this);

    menuContainer->addElement(subMenuIcon);

    subMenu = 0;
    open = false;
    vruiIntersection::getIntersectorForAction("coAction")->add(background->getDCS(), this);

    attachment = coUIElement::BOTTOM;
}

/// Destructor.
coSubMenuToolboxItem::~coSubMenuToolboxItem()
{
    vruiIntersection::getIntersectorForAction("coAction")->remove(this);
    delete subMenuIcon;
}

/** This method is called on intersections of the input device with this menu item.
  @return ACTION_CALL_ON_MISS
*/
int coSubMenuToolboxItem::hit(vruiHit *hit)
{
    if (!vruiRendererInterface::the()->isRayActive())
        return ACTION_CALL_ON_MISS;
    // update coJoystickManager
    if (vruiRendererInterface::the()->isJoystickActive())
        vruiRendererInterface::the()->getJoystickManager()->selectItem(this, myMenu);

    //VRUILOG("coSubMenuToolboxItem::hit info: called");

    Result preReturn = vruiRendererInterface::the()->hit(this, hit);
    if (preReturn != ACTION_UNDEF)
        return preReturn;

    background->setHighlighted(true);

    vruiButtons *buttons = hit->isMouseHit()
                               ? vruiRendererInterface::the()->getMouseButtons()
                               : vruiRendererInterface::the()->getButtons();

    if (buttons->wasPressed(vruiButtons::ACTION_BUTTON))
    {
        pressed = true;
    }

    if (pressed && (buttons->wasReleased(vruiButtons::ACTION_BUTTON)))
    {
        pressed = false;

        open = !open;
        if (open)
        {
            openSubmenu();
        }
        else
        {
            closeSubmenu();
        }
        if (listener)
            listener->menuEvent(this);
    }
    return ACTION_CALL_ON_MISS;
}

/// Called when input device leaves the element.
void coSubMenuToolboxItem::miss()
{
    if (!vruiRendererInterface::the()->isRayActive())
        return;

    background->setHighlighted(false);
}

/**
  * highlight the item
  */
void coSubMenuToolboxItem::selected(bool select)
{
    if (vruiRendererInterface::the()->isJoystickActive())
        background->setHighlighted(select);
}

/**
  * open or close Submenu
  */
void coSubMenuToolboxItem::doActionRelease()
{
    open = !open;
    if (open)
    {
        openSubmenu();
    }
    else
        closeSubmenu();
    if (listener)
        listener->menuEvent(this);
}

void coSubMenuToolboxItem::setMenu(coMenu *menu)
{
    subMenu = menu;
}

/// Close the submenu.
void coSubMenuToolboxItem::closeSubmenu()
{
    // mark submenu as closed
    open = false;

    // set Icon to appropriate state
    subMenuIcon->setState(open);

    if (vruiRendererInterface::the()->getJoystickManager())
        vruiRendererInterface::the()->getJoystickManager()->closedMenu(subMenu, myMenu);
}

/// Open the submenu.
void coSubMenuToolboxItem::openSubmenu()
{
    // if submenuicon is inactive cannot open the submenu
    if (!getActive())
        return;
    // set own state
    open = true;

    // set Icon to appropriate state
    subMenuIcon->setState(open);

    if (vruiRendererInterface::the()->getJoystickManager())
        vruiRendererInterface::the()->getJoystickManager()->openedSubMenu(this, subMenu);
}

void coSubMenuToolboxItem::positionSubmenu()
{
    if (subMenu && myMenu)
    {
        float sm_x = 0.f;
        float sm_y = 0.f;
        float sm_z = 0.f;

        vruiMatrix *transMatrix = vruiRendererInterface::the()->createMatrix();
        vruiMatrix *menuPosition = vruiRendererInterface::the()->createMatrix();

        // translation to zero
        transMatrix->makeIdentity();

        vruiTransformNode *node = background->getDCS();

        // set up global matrix for submenu positioning
        node->convertToWorld(transMatrix);

        // determine submenu position
        switch (attachment)
        {
        // use gaps?
        case coUIElement::LEFT:
            sm_x = background->getWidth();
            sm_y = background->getHeight();
            break;
        case coUIElement::BOTTOM:
            sm_x = 0.0f;
            sm_y = background->getHeight() + subMenu->getUIElement()->getHeight();
            break;
        case coUIElement::RIGHT:
            sm_x = -subMenu->getUIElement()->getWidth();
            sm_y = background->getHeight();
            break;
        case coUIElement::TOP:
            sm_x = 0.0f;
            sm_y = 0.0f;
            break;
        }
        // generally set new menus to front
        sm_z = 10.0f;

        // now multiply submenu's position (in menu coords)
        // to world coords using transmat
        menuPosition->preTranslated(sm_x, sm_y, sm_z, transMatrix);

        // set menu position and scale
        if (myMenu)
            subMenu->setTransformMatrix(menuPosition, myMenu->getScale());

        vruiRendererInterface::the()->deleteMatrix(transMatrix);
        vruiRendererInterface::the()->deleteMatrix(menuPosition);
    }
}

//set if item is active
void coSubMenuToolboxItem::setActive(bool a)
{
    // if item is activated add background to intersector
    if (!active_ && a)
    {
        vruiIntersection::getIntersectorForAction("coAction")->add(background->getDCS(), this);
        coToolboxMenuItem::setActive(a);
        subMenuIcon->setActive(a);
    }
    // if item is deactivated remove background from intersector
    else if (active_ && !a)
    {
        if (isOpen())
            closeSubmenu();
        vruiIntersection::getIntersectorForAction("coAction")->remove(this);
        coToolboxMenuItem::setActive(a);
        subMenuIcon->setActive(a);
    }
}

const char *coSubMenuToolboxItem::getClassName() const
{
    return "coSubMenuToolboxItem";
}

bool coSubMenuToolboxItem::isOfClassName(const char *classname) const
{
    // paranoia makes us mistrust the string library and check for 0.
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
