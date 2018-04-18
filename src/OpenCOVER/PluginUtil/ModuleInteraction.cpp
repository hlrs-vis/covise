/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "ModuleInteraction.h"
#include <cover/coVRPluginSupport.h>
#include <cover/ui/Button.h>
#include <cover/ui/Menu.h>
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

    showPickInteractorCheckbox_ = new ui::Button(menu_, "PickInteractor");
    showPickInteractorCheckbox_->setText("Pick interactor");
    showPickInteractorCheckbox_->setState(showPickInteractor_);
    showPickInteractorCheckbox_->setCallback([this](bool state){
        showPickInteractor_ = state;
        updatePickInteractors(state);
    });

    if (coVRConfig::instance()->has6DoFInput())
    {
        showDirectInteractorCheckbox_ = new ui::Button(menu_, "DirectInteractor");
        showDirectInteractorCheckbox_->setText("Direct interactor");
        showDirectInteractorCheckbox_->setState(showDirectInteractor_);
        showDirectInteractorCheckbox_->setGroup(cover->navGroup());
        showDirectInteractorCheckbox_->setCallback([this](bool state){
            showDirectInteractor_ = state;
            updateDirectInteractors(state);
        });
#ifdef VRUI
        if (parent)
            parent->setSecondaryItem(showDirectInteractorCheckbox_);
#endif
    }
}

ModuleInteraction::~ModuleInteraction()
{
    FeedbackManager::instance()->unregisterFeedback(this);
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

void
ModuleInteraction::setShowInteractorFromGui(bool state)
{
    showPickInteractor_ = state;
    showPickInteractorCheckbox_->setState(state);
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
