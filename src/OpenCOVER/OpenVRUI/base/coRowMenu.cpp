/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/coRowMenu.h>

#include <OpenVRUI/coFrame.h>
#include <OpenVRUI/coRowMenuHandle.h>
#include <OpenVRUI/coMenu.h>
#include <OpenVRUI/coMenuItem.h>
#include <OpenVRUI/coRowContainer.h>
#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coRowMenuItem.h>
#include <OpenVRUI/coLabel.h>
#include <OpenVRUI/coButton.h>
#include <OpenVRUI/coColoredBackground.h>
#include <OpenVRUI/coCombinedButtonInteraction.h>
#include <OpenVRUI/coIconButtonToolboxItem.h>

#include <OpenVRUI/sginterface/vruiButtons.h>
#include <OpenVRUI/sginterface/vruiIntersection.h>
#include <OpenVRUI/sginterface/vruiRendererInterface.h>
#include <OpenVRUI/sginterface/vruiGroupNode.h>
#include <OpenVRUI/sginterface/vruiTransformNode.h>
#include <OpenVRUI/util/vruiLog.h>

#include <config/CoviseConfig.h>

#include <stdlib.h>

namespace vrui
{

/** Constructor.
  @param name name of the Menu which is displayed on the titlebar
  @param parent pointer to parent menu class, NULL if this is the topmost menu
*/
coRowMenu::coRowMenu(const char *name, coMenu *parent, int maxItems, bool inScene)
    : coMenu(parent, name)
{

    inScene_ = inScene;

    // Create menu items area:
    itemsContainer = new coRowContainer(coRowContainer::VERTICAL);
    itemsContainer->setHgap(0);
    itemsContainer->setVgap(0);
    itemsContainer->setDgap(0);

    itemsFrame = new coFrame("UI/Frame");
    itemsFrame->addElement(itemsContainer);
    itemsFrame->fitToParent();

    handle = new coRowMenuHandle(name, this);
    handle->addElement(itemsFrame);

    vruiIntersection::getIntersectorForAction("coAction")->add(handle->getDCS(), this);
    myMenuItem = 0;

    stateChangeRunning = false;
    showOperation = false;
    isHidden = false;
    stateDelay = 0.3f;

    maxItems_ = maxItems;
    if (maxItems == 0)
    {
        maxItems_ = covise::coCoviseConfig::getInt("COVER.VRUI.MenuMaxItems", 0);
    }

    upItem_ = NULL;
    downItem_ = NULL;
    startPos_ = 0;
    if (maxItems_ != 0)
    {
        // create button menu items for scrolling
        upItem_ = new coIconButtonToolboxItem("UI/scrollUp");
        upItem_->setMenuListener(this);
        upItem_->setActive(false);

        downItem_ = new coIconButtonToolboxItem("UI/scrollDown");
        downItem_->setActive(false);
        downItem_->setMenuListener(this);
        itemsContainer->addElement(upItem_->getUIElement());
        itemsContainer->addElement(downItem_->getUIElement());
    }
}

/// Destructor.
coRowMenu::~coRowMenu()
{
    vruiIntersection *intersection = vruiIntersection::getIntersectorForAction("coAction");
    if (intersection)
        intersection->remove(this);

    removeAll();

    delete itemsContainer;
    delete itemsFrame;
    delete handle;
}

int coRowMenu::hit(vruiHit *hit)
{
    coMenu::hit(hit);

    if (moveInteraction->wasStarted())
        handle->hit(hit);

    if (moveInteraction->wasStopped())
        handle->miss();

    return ACTION_CALL_ON_MISS;
}

void coRowMenu::miss()
{
    coMenu::miss();
    handle->miss();
}

/** 
  * select is called if the menu is selected via joystick
  * highlight the handle
  */
void coRowMenu::selected(bool select)
{
    handle->highlight(select);
}

void coRowMenu::makeVisible(coMenuItem *item)
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

void coRowMenu::add(coMenuItem *item)
{
    if (maxItems_ != 0)
    {
        if (ssize_t(items.size()) <= startPos_ + maxItems_ - 1)
            itemsContainer->insertElement(item->getUIElement(), itemsContainer->getSize() - 1);

        if (startPos_ > 0 && ssize_t(items.size()) > maxItems_)
            upItem_->setActive(true);
        if (startPos_ + maxItems_ < ssize_t(items.size()))
            downItem_->setActive(true);
    }
    else
        itemsContainer->addElement(item->getUIElement());
    coMenu::add(item);
}

void coRowMenu::insert(coMenuItem *item, int pos)
{
    if (maxItems_ != 0)
    {
        if ((pos >= startPos_) && (pos <= startPos_ + maxItems_ - 1))
            itemsContainer->insertElement(item->getUIElement(), pos + 1);
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
    coMenu::insert(item, pos);
}

void coRowMenu::remove(coMenuItem *item)
{
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
            startPos_--;
            int i = 0;
            for (std::list<coMenuItem *>::iterator it = items.begin(); it != items.end(); it++)
            {
                i++;
                if (i == startPos_) // insert already happened
                    itemsContainer->insertElement((*it)->getUIElement(), 1);
            }
        }
        if (startPos_ == 0)
            upItem_->setActive(false);
        if (startPos_ + maxItems_ == ssize_t(items.size()))
            downItem_->setActive(false);
    }
    else
    {
        if (item)
        {
            if (item->getUIElement())
                itemsContainer->removeElement(item->getUIElement());
            coMenu::remove(item);
            item->setParentMenu(0);
        }
    }
}

void coRowMenu::setTransformMatrix(vruiMatrix *matrix)
{
    handle->setTransformMatrix(matrix);
}

void coRowMenu::setTransformMatrix(vruiMatrix *matrix, float scale)
{
    handle->setTransformMatrix(matrix, scale);
}

void coRowMenu::setScale(float scale)
{
    handle->setScale(scale);
}

float coRowMenu::getScale() const
{
    return handle->getScale();
}

void coRowMenu::setVisible(bool newState)
{
    if (visible == newState)
        return; // state is already ok
    visible = newState;
    if (visible)
    {
        if (!inScene_)
            vruiRendererInterface::the()->getMenuGroup()->addChild(getDCS());
        else
        {
            if (vruiRendererInterface::the()->getScene())
                vruiRendererInterface::the()->getScene()->addChild(getDCS());
        }
    }
    else
    {
        getDCS()->removeAllParents();
    }
    if (vruiRendererInterface::the()->getJoystickManager() && visible)
        vruiRendererInterface::the()->getJoystickManager()->openedSubMenu(NULL, this);
    else if (vruiRendererInterface::the()->getJoystickManager() && !visible)
        vruiRendererInterface::the()->getJoystickManager()->closedMenu(this, parent);
}

vruiTransformNode *coRowMenu::getDCS()
{
    return handle->getDCS();
}

/// This function frequently updates the menu scale when activated
bool coRowMenu::update()
{

    double t_i;
    float scale_i;

    if (stateChangeRunning)
    {
        t_now = vruiRendererInterface::the()->getFrameTime();
        t_i = (t_now - t_start) / (t_end - t_start);

        //VRUILOG("coRowMenu::update info: menu sliding (" << (t_i * 100) << "%)");

        // slide up or down linear and stay below 100%
        if (t_i < 0.99)
        {
            switch (showMode)
            {
            case MENU_SLIDE:
                if (showOperation)
                {
                    scale_i = (float)t_i;

                    itemsFrame->getDCS()->setScale(1.0f, scale_i, 1.0f);

                    itemsFrame->setPos(itemsFrame->getXpos(),
                                       itemsFrame->getHeight() * (1.0f - scale_i)
                                       + handle->getVgap(),
                                       itemsFrame->getZpos());
                }
                else
                { // hide operation
                    scale_i = 1.0f - (float)t_i;

                    itemsFrame->getDCS()->setScale(1.0f, scale_i, 1.0f);

                    itemsFrame->setPos(itemsFrame->getXpos(),
                                       itemsFrame->getHeight() * (1.0f - scale_i)
                                       + handle->getVgap(),
                                       itemsFrame->getZpos());
                }

                break;

            default:
                // just switch on or off
                if (showOperation)
                {
                    itemsFrame->getDCS()->setScale(1.0f, 1.0f, 1.0f);
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
                handle->getDCS()->removeChild(itemsFrame->getDCS());
                //itemsFrame->getDCS()->setScale(1.0, 0.001, 0.001);
                isHidden = true;
            }
            else
            {
                itemsFrame->getDCS()->setScale(1.0f, 1.0f, 1.0f);

                itemsFrame->setPos(itemsFrame->getXpos(),
                                   handle->getVgap(),
                                   itemsFrame->getZpos());
            }

            stateChangeRunning = false;

        } // eo 99% boundary

    } // eo stateChangeRunning

    handle->update();

    return true;
}

void coRowMenu::show()
{

    //VRUILOG("coRowMenu::show info: called")

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
    if (vruiRendererInterface::the()->getJoystickManager())
        vruiRendererInterface::the()->getJoystickManager()->openedSubMenu(NULL, this);
}

void coRowMenu::hide()
{

    //VRUILOG("coRowMenu::hide info: called")

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

/// return the actual UI Element that represents this menu.
coUIElement *coRowMenu::getUIElement()
{
    return handle;
}

const char *coRowMenu::getClassName() const
{
    return "coRowMenu";
}

bool coRowMenu::isOfClassName(const char *classname) const
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

void coRowMenu::updateTitle(const char *newTitle)
{
    handle->updateTitle(newTitle);
}

void coRowMenu::buttonEvent(coButton *)
{
}

void
coRowMenu::menuEvent(coMenuItem *item)
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
            {
                itemsContainer->insertElement((*it)->getUIElement(), 1);
            }
            i++;
        }

        // reposition submenus
        i = 0;
        for (std::list<coMenuItem *>::iterator it = items.begin(); it != items.end(); it++)
        {
            if ((*it)->isOfClassName("coSubMenuItem"))
            {
                if ((i >= startPos_) && (i < startPos_ + maxItems_))
                {
                    ((coSubMenuItem *)(*it))->positionSubmenu();
                }
                else
                {
                    ((coSubMenuItem *)(*it))->closeSubmenu();
                }
            }
            i++;
        }
    }
    else if (item == downItem_)
    {
        //if (startPos_+maxItems < items.size()-1)
        //deactivate down button

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
            }
        }

        // reposition submenus
        i = 0;
        for (std::list<coMenuItem *>::iterator it = items.begin(); it != items.end(); it++)
        {
            if ((*it)->isOfClassName("coSubMenuItem"))
            {
                if ((i >= startPos_) && (i < startPos_ + maxItems_))
                {
                    ((coSubMenuItem *)(*it))->positionSubmenu();
                }
                else
                {
                    ((coSubMenuItem *)(*it))->closeSubmenu();
                }
            }
            i++;
        }
    }
}
}
