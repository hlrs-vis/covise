/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/coFrame.h>
#include <OpenVRUI/coToolboxMenuHandle.h>
#include <OpenVRUI/coRowContainer.h>
#include <OpenVRUI/coToolboxMenu.h>
#include <OpenVRUI/coColoredBackground.h>
#include <OpenVRUI/coIconButtonToolboxItem.h>
#include <OpenVRUI/coSubMenuToolboxItem.h>
#include <OpenVRUI/coSubMenuItem.h>

#include <OpenVRUI/coMenuContainer.h>

#include <OpenVRUI/sginterface/vruiButtons.h>
#include <OpenVRUI/sginterface/vruiIntersection.h>
#include <OpenVRUI/sginterface/vruiRendererInterface.h>

#include <config/CoviseConfig.h>

#include <stdlib.h>

namespace vrui
{

coToolboxMenu::coToolboxMenu(const std::string &name, coMenu *parent,
                             coRowContainer::Orientation ori, int attach, int maxItems)
    : coMenu(parent, name)
{
    // structure with handle:
    // handleContainer( handleFrame(...), itemsFrame( itemsContainer( Item,Item,Item,... ) ), )

    // all item operations are fed into the container,
    // all geometric operations are handed over to the handle

    // create elements
    // Create menu items area:
    itemsContainer = new coRowContainer(ori);
    itemsContainer->setHgap(0);
    itemsContainer->setVgap(0);
    itemsContainer->setDgap(0);

    // insert items container into items frame
    itemsFrame = new coFrame("UI/Frame");

    itemsFrame->addElement(itemsContainer);
    itemsFrame->fitToParent();

    // insert frame into handle
    handle = new coToolboxMenuHandle(name, this);
    handle->addElement(itemsFrame);

    attachment = attach;
    stateChangeRunning = false;
    showOperation = false;
    isHidden = false;
    t_start = t_end = t_now = 0;
    stateDelay = 0.3;

    vruiIntersection::getIntersectorForAction("coAction")->add(handle->getDCS(), this);

    maxItems_ = maxItems;
    if (maxItems == 0)
    {
        std::string menuMaxItems = covise::coCoviseConfig::getEntry("COVER.Plugin.AKToolbar.MenuMaxItems");
        if (!menuMaxItems.empty())
        {
            maxItems_ = atoi(menuMaxItems.c_str());
        }
    }

    upItem_ = NULL;
    downItem_ = NULL;
    startPos_ = 0;
    if (maxItems_ != 0)
    {
        // create button menu items for scrolling
        upItem_ = new coIconButtonToolboxItem("UI/scrollLeft");
        upItem_->setMenuListener(this);
        upItem_->setActive(false);

        downItem_ = new coIconButtonToolboxItem("UI/scrollRight");
        downItem_->setMenuListener(this);
        downItem_->setActive(false);
        itemsContainer->addElement(upItem_->getUIElement());
        itemsContainer->addElement(downItem_->getUIElement());
    }
    else
    {
        upItem_ = 0;
        downItem_ = 0;
    }
    visible = false;

    if (vruiRendererInterface::the()->getJoystickManager())
        vruiRendererInterface::the()->getJoystickManager()->registerMenu(this, true);
}

coToolboxMenu::~coToolboxMenu()
{

    vruiIntersection::getIntersectorForAction("coAction")->remove(this);

    delete itemsContainer;
    delete itemsFrame;
    delete handle;
}

void coToolboxMenu::add(coMenuItem *item)
{
    coMenu::add(item);
    item->setAttachment(getAttachment());

    if (maxItems_ != 0)
    {
        bool inserted = false;
        if (ssize_t(items.size()) <= startPos_ + maxItems_)
        {
            itemsContainer->insertElement(item->getUIElement(), itemsContainer->getSize() - 1);
            inserted = true;
        }
        if (startPos_ > 0 && ssize_t(items.size()) > maxItems_)
            upItem_->setActive(true);
        if (startPos_ + maxItems_ < ssize_t(items.size()))
            downItem_->setActive(true);
        if (!inserted)
            menuEvent(downItem_);
    }
    else
        itemsContainer->addElement(item->getUIElement());
}

void coToolboxMenu::insert(coMenuItem *item, int pos)
{
    coMenu::insert(item, pos);
    if (maxItems_ != 0)
    {
        if ((pos >= startPos_) && (pos <= startPos_ + maxItems_ - 1))
        {
            itemsContainer->insertElement(item->getUIElement(), pos + 1);
        }
        else if (pos < startPos_)
            startPos_++;

        if (itemsContainer->getSize() > maxItems_ + 2)
        {
            int i = 0;
            for (std::list<coMenuItem *>::iterator it = items.begin(); it != items.end(); it++)
            {
                i++;
                if (i == maxItems_ + 1) // insert already happened
                    itemsContainer->removeElement((*it)->getUIElement());
            }
        }
        if (startPos_ > 0 && ssize_t(items.size()) > maxItems_)
            upItem_->setActive(true);
        if (startPos_ + maxItems_ < ssize_t(items.size()))
            downItem_->setActive(true);
    }
    else
        itemsContainer->insertElement(item->getUIElement(), pos);
}

void coToolboxMenu::remove(coMenuItem *item)
{
    if (dynamic_cast<coGenericSubMenuItem *>(item))
        ((coGenericSubMenuItem *)item)->closeSubmenu();

    if (maxItems_ != 0)
    {
        int numItem = 0;
        for (std::list<coMenuItem *>::iterator it = items.begin(); it != items.end(); it++)
        {
            if (item == *it)
                break;
            numItem++;
        }
        coMenu::remove(item);
        if (item && item->getUIElement())
            itemsContainer->removeElement(item->getUIElement());

        // decrease startpos if deleted in front
        if (numItem < startPos_)
            startPos_--;
        // insert item in back
        else if (ssize_t(items.size()) >= startPos_ + maxItems_ && itemsContainer->getSize() < maxItems_ + 2)
        {
            int i = 0;
            for (std::list<coMenuItem *>::iterator it = items.begin(); it != items.end(); it++)
            {
                i++;
                if (i == startPos_ + maxItems_) // insert already happened
                    itemsContainer->insertElement((*it)->getUIElement(), maxItems_);
            }
        }
        // insert item in front
        else if ((ssize_t(items.size()) >= maxItems_) && (ssize_t(items.size()) < startPos_ + maxItems_) && (itemsContainer->getSize() < maxItems_ + 2))
        {
            int i = 0;
            for (std::list<coMenuItem *>::iterator it = items.begin(); it != items.end(); it++)
            {
                i++;
                if (i == startPos_) // insert already happened
                    itemsContainer->insertElement((*it)->getUIElement(), 1);
            }
            startPos_--;
        }
        if (startPos_ == 0)
            upItem_->setActive(false);
        if (startPos_ + maxItems_ == ssize_t(items.size()))
            downItem_->setActive(false);
    }
    else
    {
        if (item && item->getUIElement())
            itemsContainer->removeElement(item->getUIElement());
        coMenu::remove(item);
    }
}

void coToolboxMenu::removeAll()
{
    coMenu::removeAll();
}

int coToolboxMenu::getItemCount() const
{
    return coMenu::getItemCount();
}

void coToolboxMenu::setVisible(bool newState)
{
    if (visible == newState)
        return; // state is already ok
    visible = newState;
    if (visible)
    {
        vruiRendererInterface::the()->getMenuGroup()->addChild(getDCS());
    }
    else
    {
        while (getDCS() && getDCS()->getParent(0))
        {
            getDCS()->getParent(0)->removeChild(getDCS());
        }
    }
}

void coToolboxMenu::setScale(float s)
{
    handle->setScale(s);
}

float coToolboxMenu::getScale() const
{
    return handle->getScale();
}

vruiTransformNode *coToolboxMenu::getDCS()
{
    return handle->getDCS();
}

void coToolboxMenu::setTransformMatrix(vruiMatrix *mat)
{
    vruiMatrix *trans = mat;

    //    trans.preTrans(0,0,0,mat);  // Why?

    handle->setTransformMatrix(trans);
}

void coToolboxMenu::setTransformMatrix(vruiMatrix *mat, float scale)
{
    handle->setTransformMatrix(mat, scale);
}

/// This function frequently updates the menu scale when activated
bool coToolboxMenu::update()
{
    double t_i;

    if (stateChangeRunning)
    {

        // get current time index
        t_now = vruiRendererInterface::the()->getFrameTime();
        t_i = (t_now - t_start) / (t_end - t_start);

        // slide up or down linear and stay below 100%
        if (t_i < 0.99)
        {

            switch (showMode)
            {

            case MENU_SLIDE:
                // adjust t_i for close operations
                if (!showOperation)
                    t_i = 1.0 - t_i;

                // translate to proper position if necessary
                switch (attachment)
                {
                case coUIElement::TOP:
                case coUIElement::BOTTOM:
                    itemsFrame->getDCS()->setScale((float)t_i, 1.0, 1.0);
                    break;

                case coUIElement::LEFT:
                case coUIElement::RIGHT:
                    // icon bar on top, items hanging downwards
                    itemsFrame->getDCS()->setScale(1.0, (float)t_i, 1.0);

                    itemsFrame->setPos(itemsFrame->getXpos(),
                                       (float)itemsFrame->getHeight() * (1.0f - t_i)
                                       + handle->getVgap(),
                                       itemsFrame->getZpos());
                    break;
                }

                break;

            default:
                // just switch on or off
                if (showOperation)
                {
                    itemsFrame->getDCS()->setScale(1.0, 1.0, 1.0);
                }
                else
                {
                    //itemsFrame->getDCS()->setScale(1.0, 0.001, 0.001);
                    handle->getDCS()->removeChild(itemsFrame->getDCS());
                    isHidden = true;
                }
                // and stop operation
                stateChangeRunning = false;
                break;
            }
        }
        else
        {

            if (!showOperation)
            {
                // make sure that scale is not zero but hide...
                //itemsFrame->getDCS()->setScale(1.0, 0.001, 0.001);
                handle->getDCS()->removeChild(itemsFrame->getDCS());
                isHidden = true;
            }
            else
            {
                itemsFrame->getDCS()->setScale(1.0, 1.0, 1.0);
            }

            switch (attachment)
            {
            case coUIElement::TOP:
            case coUIElement::BOTTOM:
                //  itemsFrame->getDCS()->setTrans(0.0, 0.0, 0.0);
                //  itemsFrame->setPos(0.0, 0.0, 0.0);
                break;

            case coUIElement::LEFT:
            case coUIElement::RIGHT:
                itemsFrame->setPos(itemsFrame->getXpos(),
                                   handle->getVgap(),
                                   itemsFrame->getZpos());
                break;
            }

            stateChangeRunning = false;

        } // eo 99% boundary

    } // eo stateChangeRunning

    handle->update();

    return true;
}

void coToolboxMenu::show()
{
    double t_j;

    if (isHidden)
    {
        handle->getDCS()->addChild(itemsFrame->getDCS());
        isHidden = false;
    }

    if (!stateChangeRunning)
    {
        stateChangeRunning = true;

        showOperation = true;
        t_start = vruiRendererInterface::the()->getFrameTime();
        t_end = t_start + stateDelay;

        t_now = t_start;
    }
    else
    {
        if (!showOperation)
        {
            // make sure t_i is at inverse position in frame...
            t_j = (t_end - t_now) / (t_end - t_start);

            showOperation = true;
            t_start = t_now - t_j * stateDelay;
            t_end = t_start + stateDelay;
        }
    }
}

void coToolboxMenu::hide()
{
    double t_j;

    if (!stateChangeRunning)
    {
        stateChangeRunning = true;

        showOperation = false;
        t_start = vruiRendererInterface::the()->getFrameTime();
        t_end = t_start + stateDelay;

        t_now = t_start;
    }
    else
    {
        if (showOperation)
        {
            // make sure t_i is at inverse position in frame...
            t_j = (t_end - t_now) / (t_end - t_start);

            showOperation = false;
            t_start = t_now - t_j * stateDelay;
            t_end = t_start + stateDelay;
        }
    }
}

// the toolbox's attachment should be changed to 'ori'
void coToolboxMenu::setAttachment(int newatt)
{
    float oldX = 0.0f, oldY = 0.0f, oldZ = 0.0f;
    float newX = 0.0f, newY = 0.0f, newZ = 0.0f;

    vruiMatrix *oldMat;
    vruiMatrix *newMat = vruiRendererInterface::the()->createMatrix();

    // get old relative position of lower right corner of titleFrame
    if ((attachment == coUIElement::LEFT) || (attachment == coUIElement::RIGHT))
    {
        oldX = itemsFrame->getXpos() + itemsFrame->getWidth();
        oldY = itemsFrame->getYpos() + itemsFrame->getHeight()
               + handle->getVgap();
        oldZ = itemsFrame->getZpos();
    }
    else if ((attachment == coUIElement::TOP) || (attachment == coUIElement::BOTTOM))
    {
        oldX = itemsFrame->getXpos() - handle->getHgap();
        oldY = itemsFrame->getYpos();
        oldZ = itemsFrame->getZpos();
    }

    // switch the container orientation
    // Turn around and adjust size
    // Containers have to care about re-attaching their own children.
    switch (newatt)
    {
    case coUIElement::TOP:
    case coUIElement::BOTTOM:
        handle->setOrientation(coRowContainer::HORIZONTAL);
        break;

    case coUIElement::LEFT:
    case coUIElement::RIGHT:
        handle->setOrientation(coRowContainer::VERTICAL);
        break;
    }

    // now re-attach the container children
    itemsContainer->setAttachment(newatt);

    // now re-attach all items
    for (std::list<coMenuItem *>::iterator i = items.begin(); i != items.end(); ++i)
    {
        (*i)->setAttachment(newatt);
    }

    // copy attachment
    attachment = newatt;

    // get new relative position of lower right corner of titleFrame
    if ((attachment == coUIElement::LEFT) || (attachment == coUIElement::RIGHT))
    {
        newX = itemsFrame->getXpos() + itemsFrame->getWidth();
        newY = itemsFrame->getYpos() + itemsFrame->getHeight()
               + handle->getVgap();
        newZ = itemsFrame->getZpos();
    }
    else if ((attachment == coUIElement::TOP) || (attachment == coUIElement::BOTTOM))
    {
        newX = itemsFrame->getXpos() - handle->getHgap();
        newY = itemsFrame->getYpos();
        newZ = itemsFrame->getZpos();
    }

    // now set position
    oldMat = handle->getDCS()->getMatrix();
    newMat->preTranslated(oldX - newX,
                          oldY - newY,
                          oldZ - newZ, oldMat);
    handle->getDCS()->setMatrix(newMat);
}

int coToolboxMenu::getAttachment() const
{
    return attachment;
}

/// return the actual UI Element that represents this menu.
coUIElement *coToolboxMenu::getUIElement()
{
    return handle;
}

const char *coToolboxMenu::getClassName() const
{
    return "coToolboxMenu";
}

bool coToolboxMenu::isOfClassName(const char *classname) const
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
            return coMenu::isOfClassName(classname);
        }
    }

    // nobody is NULL
    return false;
}

void coToolboxMenu::fixPos(bool doFix)
{
    handle->fixPos(doFix);
}

void
coToolboxMenu::menuEvent(coMenuItem *item)
{
    if (item == upItem_)
    {

        startPos_--;

        if (startPos_ == 0)
            upItem_->setActive(false);

        if (startPos_ + maxItems_ == ssize_t(items.size()) - 1)
            downItem_->setActive(true);

        // remove last
        int i = 0;
        for (std::list<coMenuItem *>::iterator it = items.begin(); it != items.end(); it++)
        {
            if (i == startPos_ + maxItems_)
            {
                itemsContainer->removeElement((*it)->getUIElement());
            }
            i++;
        }
        // add first
        i = 0;
        for (std::list<coMenuItem *>::iterator it = items.begin(); it != items.end(); it++)
        {
            if (i == startPos_)
                itemsContainer->insertElement((*it)->getUIElement(), 1);
            i++;
        }

        // reposition submenus
        i = 0;
        for (std::list<coMenuItem *>::iterator it = items.begin(); it != items.end(); it++)
        {
            if (dynamic_cast<coSubMenuToolboxItem *>(*it))
            {
                if ((i >= startPos_) && (i < startPos_ + maxItems_))
                {
                    if (((coSubMenuToolboxItem *)(*it))->isOpen())
                        ((coSubMenuToolboxItem *)(*it))->positionSubmenu();
                }
                else
                {
                    ((coSubMenuToolboxItem *)(*it))->closeSubmenu();
                }
            }
            i++;
        }
    }
    else if (item == downItem_)
    {
        startPos_++;

        if (startPos_ == 1)
            upItem_->setActive(true);

        if (startPos_ + maxItems_ == ssize_t(items.size()))
            downItem_->setActive(false);

        // remove first in container
        int i = 0;
        for (std::list<coMenuItem *>::iterator it = items.begin(); it != items.end(); it++)
        {
            i++;
            if (i == startPos_)
            {
                itemsContainer->removeElement((*it)->getUIElement());
                break;
            }
        }
        // add to container
        i = 0;
        for (std::list<coMenuItem *>::iterator it = items.begin(); it != items.end(); it++)
        {
            i++;
            if (i == startPos_ + maxItems_)
            {
                itemsContainer->insertElement((*it)->getUIElement(), maxItems_);
                break;
            }
        }

        // reposition submenus
        i = 0;
        for (std::list<coMenuItem *>::iterator it = items.begin(); it != items.end(); it++)
        {
            if (dynamic_cast<coSubMenuToolboxItem *>(*it))
            {
                if ((i >= startPos_) && (i < startPos_ + maxItems_))
                {
                    if (((coSubMenuToolboxItem *)(*it))->isOpen())
                        ((coSubMenuToolboxItem *)(*it))->positionSubmenu();
                }
                else
                {
                    ((coSubMenuToolboxItem *)(*it))->closeSubmenu();
                }
            }
            i++;
        }
    }
}

/** 
  * select is called if the menu is selected via joystick
  * highlight the handle
  */
void coToolboxMenu::selected(bool select)
{
    if (vruiRendererInterface::the()->isJoystickActive())
        handle->highlight(select);
}

/**
  * scrolls to the correct Item
  */
void coToolboxMenu::makeVisible(coMenuItem *item)
{
    if (maxItems_ == 0)
        return;

    std::list<coMenuItem *>::iterator it;
    int i = 0;
    for (it = items.begin(); it != items.end(); it++)
    {
        if ((*it) == item)
        {
            if (startPos_ + maxItems_ <= i)
                menuEvent(downItem_);
            else if (startPos_ > i)
                menuEvent(upItem_);
            break;
        }
        i++;
    }
}
}
