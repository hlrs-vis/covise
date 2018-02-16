/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/coRowMenu.h>

#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coFlatButtonGeometry.h>
#include <OpenVRUI/coMenuContainer.h>
#include <OpenVRUI/coButton.h>
#include <OpenVRUI/coLabel.h>
#include <OpenVRUI/coBackground.h>
#include <OpenVRUI/coColoredBackground.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coInteractionManager.h>
#include <OpenVRUI/coCombinedButtonInteraction.h>

#include <OpenVRUI/sginterface/vruiButtons.h>
#include <OpenVRUI/sginterface/vruiIntersection.h>
#include <OpenVRUI/sginterface/vruiMatrix.h>
#include <OpenVRUI/sginterface/vruiRendererInterface.h>
#include <OpenVRUI/sginterface/vruiTransformNode.h>
#include <OpenVRUI/sginterface/vruiHit.h>

#include <OpenVRUI/util/vruiLog.h>

#include <util/unixcompat.h>
#include <config/CoviseConfig.h>

using namespace std;

namespace vrui
{

/** Constructor.
  @param name           displayed menu item text
*/
coSubMenuItem::coSubMenuItem(const string &name)
    : coRowMenuItem(name)
    , coGenericSubMenuItem(this)
    , secondaryItem(NULL)
{
    pressed = false;
    space = new coBackground();
    space->setMinWidth((float)LEFTMARGIN);
    subMenuIcon = new coRotButton(new coFlatButtonGeometry("UI/submenu"), this);
    subMenuIcon->setSize((float)LEFTMARGIN);

    preventMoveInteraction = new coCombinedButtonInteraction(coInteraction::ButtonC, "Menu", coInteraction::Menu);

    attachment = coUIElement::RIGHT;

    std::string menuAttachment = covise::coCoviseConfig::getEntry("COVER.VRUI.MenuAttachment");
    if (!menuAttachment.empty())
    {
        if (strcasecmp(menuAttachment.c_str(), "LEFT") == 0)
            attachment = coUIElement::LEFT;
        else if (strcasecmp(menuAttachment.c_str(), "RIGHT") == 0)
            attachment = coUIElement::RIGHT;
        else if (strcasecmp(menuAttachment.c_str(), "BOTTOM") == 0)
            attachment = coUIElement::BOTTOM;
        else if (strcasecmp(menuAttachment.c_str(), "TOP") == 0)
            attachment = coUIElement::TOP;
        else if (strcasecmp(menuAttachment.c_str(), "REPLACE") == 0)
            attachment = coUIElement::REPLACE;
        else
            cerr << "coSubMenuItem: unknown MenuAttachment \"" << menuAttachment << "\"" << endl;
        if (vruiRendererInterface::the()->getJoystickManager())
            vruiRendererInterface::the()->getJoystickManager()->setAttachment(attachment);
    }

    switch (attachment)
    {
    case coUIElement::LEFT:
        subMenuIcon->setRotation(0.0f);
        container->setOrientation(coMenuContainer::HORIZONTAL);
        // label,icon
        container->addElement(space);
        container->addElement(label);
        container->addElement(subMenuIcon);
        break;

    case coUIElement::BOTTOM:
        subMenuIcon->setRotation(90.0f);
        container->setOrientation(coMenuContainer::VERTICAL);
        // icon,label
        container->addElement(subMenuIcon);
        container->addElement(label);
        container->addElement(space);
        break;

    case coUIElement::RIGHT:
        subMenuIcon->setRotation(180.0f);
        container->setOrientation(coMenuContainer::HORIZONTAL);
        // icon,label
        container->addElement(subMenuIcon);
        container->addElement(label);
        container->addElement(space);
        break;

    case coUIElement::TOP:
        subMenuIcon->setRotation(270.0f);
        container->setOrientation(coMenuContainer::VERTICAL);
        //label,icon
        container->addElement(space);
        container->addElement(label);
        container->addElement(subMenuIcon);
        break;

    case coUIElement::REPLACE:
        subMenuIcon->setRotation(180.0f);
        container->setOrientation(coMenuContainer::HORIZONTAL);
        // icon,label
        container->addElement(subMenuIcon);
        container->addElement(label);
        container->addElement(space);
        break;
    }

    subMenu = 0;
    open = false;
    vruiIntersection::getIntersectorForAction("coAction")->add(background->getDCS(), this);
}

/// Destructor.
coSubMenuItem::~coSubMenuItem()
{
    if (myMenu)
        myMenu->remove(this);
    vruiIntersection::getIntersectorForAction("coAction")->remove(this);
    delete preventMoveInteraction;
    delete subMenuIcon;
    delete space;
}

void coSubMenuItem::setSecondaryItem(coMenuItem *item)
{
    secondaryItem = item;
}

/** This method is called on intersections of the input device with this menu item.
  @return ACTION_CALL_ON_MISS
*/
int coSubMenuItem::hit(vruiHit *hit)
{
    if (!vruiRendererInterface::the()->isRayActive())
        return ACTION_CALL_ON_MISS;
    // update coJoystickManager
    if (vruiRendererInterface::the()->isJoystickActive())
        vruiRendererInterface::the()->getJoystickManager()->selectItem(this, myMenu);

    //VRUILOG("coSubMenuItem::hit info: called")

    Result preReturn = vruiRendererInterface::the()->hit(this, hit);
    if (preReturn != ACTION_UNDEF)
        return preReturn;

    if (secondaryItem)
        coInteractionManager::the()->registerInteraction(preventMoveInteraction);

    background->setHighlighted(true);

    vruiButtons *buttons = hit->isMouseHit()
                               ? vruiRendererInterface::the()->getMouseButtons()
                               : vruiRendererInterface::the()->getButtons();

    if (buttons->wasPressed(vruiButtons::ACTION_BUTTON))
    {
        pressed = true;
    }

    if (!pressed && (buttons->wasReleased(vruiButtons::XFORM_BUTTON)))
    {
        if (secondaryItem)
            secondaryItem->doActionRelease();
    }

    if (pressed && buttons->wasReleased(vruiButtons::ACTION_BUTTON))
    {
        // left Button was pressed
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

        pressed = false;
    }

    return ACTION_CALL_ON_MISS;
}

/// Called when input device leaves the element.
void coSubMenuItem::miss()
{
    if (!vruiRendererInterface::the()->isRayActive())
        return;

    coInteractionManager::the()->unregisterInteraction(preventMoveInteraction);
    background->setHighlighted(false);
}

/**
  * highlight the item
  */
void coSubMenuItem::selected(bool select)
{
    if (vruiRendererInterface::the()->isJoystickActive())
    {
        coRowMenuItem::selected(select);
        background->setHighlighted(select);
    }
}

/**
  * open or close Submenu
  */
void coSubMenuItem::doActionRelease()
{
    if (!open)
    {
        open = true;
        openSubmenu();
    }
    if (listener)
        listener->menuEvent(this);
}

void coSubMenuItem::doSecondActionRelease()
{
    if (open)
    {
        open = false;
        closeSubmenu();
    }
    if (listener)
        listener->menuEvent(this);
}

void coSubMenuItem::buttonEvent(coRotButton *button)
{
    (void)button;
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

/// Close the submenu.
void coSubMenuItem::closeSubmenu()
{
    if (attachment == coUIElement::REPLACE && subMenu->isVisible())
    {
        //we have to show the menu again
        if (myMenu)
            myMenu->setVisible(true);

        // we have to position the parent menu where the submenu was
        vruiTransformNode *node;
        vruiMatrix *toWorld = vruiRendererInterface::the()->createMatrix();
        toWorld->makeIdentity();
        node = subMenu->getDCS();
        node->convertToWorld(toWorld);

        vruiMatrix *menuPosition = vruiRendererInterface::the()->createMatrix();
        float sm_x = 0;
        float sm_y = subMenu->getUIElement()->getHeight();
        float sm_z = 0;
        menuPosition->preTranslated(sm_x, sm_y, sm_z, toWorld);
        if (myMenu)
            myMenu->setTransformMatrix(menuPosition, myMenu->getScale());
    }
    open = false;
    subMenuIcon->setState(open);
    if (subMenu)
    {
        subMenu->setVisible(open);
    }

    if (vruiRendererInterface::the()->getJoystickManager())
        vruiRendererInterface::the()->getJoystickManager()->closedMenu(subMenu, myMenu);
}

/// Open the submenu.
void coSubMenuItem::openSubmenu()
{
    //cannnot open the submenu if item is inacitve
    if (!getActive())
        return;

    // set own state
    open = true;

    // set Icon to appropriate state
    subMenuIcon->setState(open);

    if (subMenu)
    {
        // show submenu
        subMenu->setVisible(open);
        positionSubmenu();

        if (attachment == coUIElement::REPLACE)
        {
            if (myMenu)
                myMenu->setVisible(false);
        }
    }

    if (vruiRendererInterface::the()->getJoystickManager())
        vruiRendererInterface::the()->getJoystickManager()->openedSubMenu(this, subMenu);
}

void coSubMenuItem::positionSubmenu()
{
    float sm_x = 0.f;
    float sm_y = 0.f;
    float sm_z = 0.f;

    vruiMatrix *transMatrix = vruiRendererInterface::the()->createMatrix();
    vruiMatrix *menuPosition = vruiRendererInterface::the()->createMatrix();

    transMatrix->makeIdentity();

    vruiTransformNode *node = background->getDCS();
    if (attachment == coUIElement::REPLACE)
    {
        if (myMenu)
            node = myMenu->getDCS();
    }

    node->convertToWorld(transMatrix);

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
    case coUIElement::REPLACE:
        sm_x = 0;
        sm_y = 0;
        if (myMenu)
            sm_y = myMenu->getUIElement()->getHeight();
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
    else
        subMenu->setTransformMatrix(menuPosition);

    vruiRendererInterface::the()->deleteMatrix(transMatrix);
    vruiRendererInterface::the()->deleteMatrix(menuPosition);
}

void coSubMenuItem::setAttachment(int newatt)
{
    // update own icon orientation
    // and type
    switch (newatt)
    {
    case coUIElement::LEFT:
        subMenuIcon->setRotation(0.0f);
        container->setOrientation(coMenuContainer::HORIZONTAL);
        // label,icon
        container->removeElement(space);
        container->removeElement(label);
        container->removeElement(subMenuIcon);
        container->addElement(space);
        container->addElement(label);
        container->addElement(subMenuIcon);
        break;

    case coUIElement::BOTTOM:
        subMenuIcon->setRotation(90.0f);
        container->setOrientation(coMenuContainer::VERTICAL);
        // icon,label
        container->removeElement(space);
        container->removeElement(label);
        container->removeElement(subMenuIcon);
        container->addElement(subMenuIcon);
        container->addElement(label);
        container->addElement(space);
        break;

    case coUIElement::RIGHT:
        subMenuIcon->setRotation(180.0f);
        container->setOrientation(coMenuContainer::HORIZONTAL);
        // icon,label
        container->removeElement(space);
        container->removeElement(label);
        container->removeElement(subMenuIcon);
        container->addElement(subMenuIcon);
        container->addElement(label);
        container->addElement(space);
        break;

    case coUIElement::TOP:
        subMenuIcon->setRotation(270.0f);
        container->setOrientation(coMenuContainer::VERTICAL);
        //label,icon
        container->removeElement(space);
        container->removeElement(label);
        container->removeElement(subMenuIcon);
        container->addElement(space);
        container->addElement(label);
        container->addElement(subMenuIcon);
        break;

    case coUIElement::REPLACE:
        subMenuIcon->setRotation(180.0f);
        container->setOrientation(coMenuContainer::HORIZONTAL);
        // icon,label
        container->removeElement(space);
        container->removeElement(label);
        container->removeElement(subMenuIcon);
        container->addElement(subMenuIcon);
        container->addElement(label);
        container->addElement(space);
        break;
    }

    // update submenu position if opened

    // copy new attachment
    attachment = newatt;
}

//set if item is active
void coSubMenuItem::setActive(bool a)
{
    // if item is activated add background to intersector
    if (!active_ && a)
    {
        vruiIntersection::getIntersectorForAction("coAction")->add(background->getDCS(), this);
        coMenuItem::setActive(a);
        subMenuIcon->setActive(a);
    }
    // if item is deactivated remove background from intersector
    else if (active_ && !a)
    {
        if (isOpen())
            closeSubmenu();
        vruiIntersection::getIntersectorForAction("coAction")->remove(this);
        coMenuItem::setActive(a);
        subMenuIcon->setActive(a);
    }
    coRowMenuItem::setActive(a);
}

const char *coSubMenuItem::getClassName() const
{
    return "coSubMenuItem";
}

bool coSubMenuItem::isOfClassName(const char *classname) const
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
            return coRowMenuItem::isOfClassName(classname);
        }
    }

    // nobody is NULL
    return false;
}
}
