/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "ModuleInteraction.h"
#ifdef VRUI
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coSubMenuItem.h>
#else
#include <cover/coVRPluginSupport.h>
#include <cover/ui/Button.h>
#include <cover/ui/Menu.h>
#endif
#include <cover/coVRConfig.h>
#include "FeedbackManager.h"

using namespace opencover;

ModuleInteraction::ModuleInteraction(const RenderObject *container, coInteractor *inter, const char *pluginName)
    : ModuleFeedbackManager(container, inter, pluginName)
    , showPickInteractor_(false)
    , showDirectInteractor_(false)
    , showPickInteractorCheckbox_(NULL)
    , showDirectInteractorCheckbox_(NULL)
{
    FeedbackManager::instance()->registerFeedback(this, inter);

#ifdef VRUI
    int menuItemCounter = 0;
    showPickInteractorCheckbox_ = new coCheckboxMenuItem("Pick Interactor", false);
    showPickInteractorCheckbox_->setMenuListener(this);
    menu_->insert(showPickInteractorCheckbox_, menuItemCounter);
    menuItemCounter++;
#else
    showPickInteractorCheckbox_ = new ui::Button(menu_, "PickInteractor");
    showPickInteractorCheckbox_->setText("Pick interactor");
    showPickInteractorCheckbox_->setState(showPickInteractor_);
    showPickInteractorCheckbox_->setCallback([this](bool state){
        showPickInteractor_ = state;
        updatePickInteractors(state);
    });
#endif

#ifdef VRUI
    coSubMenuItem *parent = dynamic_cast<coSubMenuItem *>(menu_->getSubMenuItem());
    if (parent)
        parent->setSecondaryItem(showPickInteractorCheckbox_);
#endif

    if (coVRConfig::instance()->has6DoFInput())
    {
#ifdef VRUI
        showDirectInteractorCheckbox_ = new coCheckboxMenuItem("Direct Interactor", false, groupPointerArray[0]);
        showDirectInteractorCheckbox_->setMenuListener(this);
        menu_->insert(showDirectInteractorCheckbox_, menuItemCounter);
        menuItemCounter++;
#else
        showDirectInteractorCheckbox_ = new ui::Button(menu_, "DirectInteractor");
        showDirectInteractorCheckbox_->setText("Direct interactor");
        showDirectInteractorCheckbox_->setState(showDirectInteractor_);
        showDirectInteractorCheckbox_->setGroup(cover->navGroup());
        showDirectInteractorCheckbox_->setCallback([this](bool state){
            showDirectInteractor_ = state;
            updateDirectInteractors(state);
        });
#endif
#ifdef VRUI
        if (parent)
            parent->setSecondaryItem(showDirectInteractorCheckbox_);
#endif
    }
}

ModuleInteraction::~ModuleInteraction()
{
    FeedbackManager::instance()->unregisterFeedback(this);

#ifdef VRUI
    delete showPickInteractorCheckbox_;
    delete showDirectInteractorCheckbox_;
#endif
}

void
ModuleInteraction::update(const RenderObject *container, coInteractor *inter)
{
    FeedbackManager::instance()->registerFeedback(this, inter);

    // base class updates the item in the COVISE menu
    // and the title of the Tracer menu
    ModuleFeedbackManager::update(container, inter);
}

void
ModuleInteraction::preFrame()
{
    //fprintf(stderr,"ModuleFeedbackManager::preFrame for object=%s plugin=%s\n", initialObjectName_.c_str(), pName_.c_str());
}

#ifdef VRUI
void
ModuleInteraction::menuEvent(coMenuItem *item)
{

    //fprintf(stderr,"ModuleInteraction::menuEvent %s\n", item->getName());
    ModuleFeedbackManager::menuEvent(item);

    if (item == showPickInteractorCheckbox_)
    {
        showPickInteractor_ = showPickInteractorCheckbox_->getState();
        updatePickInteractors(showPickInteractor_);
    }
    else if (item == showDirectInteractorCheckbox_)
    {
        showDirectInteractor_ = showDirectInteractorCheckbox_->getState();
        updateDirectInteractors(showDirectInteractor_);
    }
    else if (item == hideCheckbox_)
    {
        if (!hideCheckbox_->getState() && showPickInteractor_)
            updatePickInteractors(true);
        else
            updatePickInteractors(false);

        if (!hideCheckbox_->getState() && showDirectInteractor_)
            updateDirectInteractors(true);
        else
            updateDirectInteractors(false);
    }
}
#endif

void
ModuleInteraction::setShowInteractorFromGui(bool state)
{
    showPickInteractor_ = state;
    if (state)
    {
#ifdef VRUI
        if (!showPickInteractorCheckbox_->getState())
        {
            showPickInteractorCheckbox_->setState(true, true);
        }
#else
        if (!showPickInteractorCheckbox_->state())
        {
            showPickInteractorCheckbox_->setState(true);
        }
#endif
    }
    else
    {
#ifdef VRUI
        if (showPickInteractorCheckbox_->getState())
        {
            showPickInteractorCheckbox_->setState(false, true);
        }
#else
        if (showPickInteractorCheckbox_->state())
        {
            showPickInteractorCheckbox_->setState(false);
        }
#endif
    }
    updatePickInteractors(showPickInteractor_);
}

void ModuleInteraction::enableDirectInteractorFromGui(bool state)
{
    showDirectInteractor_ = false;
    if (showDirectInteractorCheckbox_)
    {
        showDirectInteractor_ = state;
        showDirectInteractorCheckbox_->setState(state);
    }
    updateDirectInteractors(showDirectInteractor_);
}

void ModuleInteraction::triggerHide(bool state)
{
    ModuleFeedbackManager::triggerHide(state);

    if (!state && showPickInteractor_)
        updatePickInteractors(true);
    else
        updatePickInteractors(false);

    if (!state  && showDirectInteractor_)
        updateDirectInteractors(true);
    else
        updateDirectInteractors(false);
}
