/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "ModuleInteraction.h"
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coSubMenuItem.h>
#include <cover/coVRConfig.h>
#include <cover/VRPinboard.h>

using namespace vrui;
using namespace opencover;

ModuleInteraction::ModuleInteraction(const RenderObject *container, coInteractor *inter, const char *pluginName)
    : ModuleFeedbackManager(container, inter, pluginName)
    , showPickInteractor_(false)
    , showDirectInteractor_(false)
    , showPickInteractorCheckbox_(NULL)
    , showDirectInteractorCheckbox_(NULL)
{

    int menuItemCounter = 0;
    showPickInteractorCheckbox_ = new coCheckboxMenuItem("Pick Interactor", false);
    showPickInteractorCheckbox_->setMenuListener(this);
    menu_->insert(showPickInteractorCheckbox_, menuItemCounter);
    menuItemCounter++;

    coSubMenuItem *parent = dynamic_cast<coSubMenuItem *>(menu_->getSubMenuItem());
    if (parent)
        parent->setSecondaryItem(showPickInteractorCheckbox_);

    if (coVRConfig::instance()->has6DoFInput())
    {
        showDirectInteractorCheckbox_ = new coCheckboxMenuItem("Direct Interactor", false, groupPointerArray[0]);
        showDirectInteractorCheckbox_->setMenuListener(this);
        menu_->insert(showDirectInteractorCheckbox_, menuItemCounter);
        menuItemCounter++;
        if (parent)
            parent->setSecondaryItem(showDirectInteractorCheckbox_);
    }
}

ModuleInteraction::~ModuleInteraction()
{
    delete showPickInteractorCheckbox_;
    delete showDirectInteractorCheckbox_;
}

void
ModuleInteraction::update(const RenderObject *container, coInteractor *inter)
{

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

void
ModuleInteraction::setShowInteractorFromGui(bool state)
{
    showPickInteractor_ = state;
    if (state)
    {

        if (!showPickInteractorCheckbox_->getState())
        {
            showPickInteractorCheckbox_->setState(true, true);
        }
    }
    else
    {
        if (showPickInteractorCheckbox_->getState())
        {
            showPickInteractorCheckbox_->setState(false, true);
        }
    }
    updatePickInteractors(showPickInteractor_);
}
