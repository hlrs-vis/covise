/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <util/common.h>

#include "IsoSurfaceInteraction.h"
#include "IsoSurfacePoint.h"
#include "IsoSurfacePlugin.h"

#include <cover/coInteractor.h>
#include <cover/coVRPluginSupport.h>

//#include <OpenVRUI/coMenu.h>
//#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coSliderMenuItem.h>
#include <OpenVRUI/coCheckboxMenuItem.h>

using namespace vrui;
using namespace opencover;

const char *IsoSurfaceInteraction::ISOVALUE = "isovalue";
const char *IsoSurfaceInteraction::ISOPOINT = "isopoint";

IsoSurfaceInteraction::IsoSurfaceInteraction(RenderObject *container, coInteractor *inter, const char *pluginName, IsoSurfacePlugin *p)
    : ModuleInteraction(container, inter, pluginName)
{
    inter->getFloatSliderParam(ISOVALUE, minValue_, maxValue_, isoValue_);
    plugin = p;

    isoPoint_ = new IsoSurfacePoint(inter, p);

    createMenu();
    updateMenu();
}

IsoSurfaceInteraction::~IsoSurfaceInteraction()
{
    delete isoPoint_;
    deleteMenu();
}

// called when a new object has arrived:
// keep menu widgets updated
void
IsoSurfaceInteraction::update(RenderObject *container, coInteractor *inter)
{
    ModuleFeedbackManager::update(container, inter);

    inter->getFloatSliderParam(ISOVALUE, minValue_, maxValue_, isoValue_);
    updateMenu();
    isoPoint_->update(inter);
}

void IsoSurfaceInteraction::updateInteractorVisibility()
{
    // if geometry is hidden, hide also interactor
    if (!hideCheckbox_->getState() && showPickInteractorCheckbox_->getState())
        isoPoint_->showPickInteractor();
    else
        isoPoint_->hidePickInteractor();

    if (!hideCheckbox_->getState() && showDirectInteractorCheckbox_ && showDirectInteractorCheckbox_->getState())
        isoPoint_->showDirectInteractor();
    else
        isoPoint_->hideDirectInteractor();
}

void
IsoSurfaceInteraction::preFrame()
{
    isoPoint_->preFrame();
}

void
IsoSurfaceInteraction::menuEvent(coMenuItem *menuItem)
{
    ModuleInteraction::menuEvent(menuItem);

    updateInteractorVisibility();
}

void
IsoSurfaceInteraction::menuReleaseEvent(coMenuItem *menuItem)
{
    if (menuItem == valueSlider_)
    {
        inter_->getFloatSliderParam(ISOVALUE, minValue_, maxValue_, isoValue_);
        isoValue_ = valueSlider_->getValue();
        plugin->getSyncInteractors(inter_);
        plugin->setSliderParam("isovalue", minValue_, maxValue_, isoValue_);
        plugin->executeModule();
    }
    else
    {
        ModuleInteraction::menuReleaseEvent(menuItem);
    }
}

void
IsoSurfaceInteraction::createMenu()
{
    // create the value poti
    inter_->getFloatSliderParam(ISOVALUE, minValue_, maxValue_, isoValue_);
    valueSlider_ = new coSliderMenuItem("IsoValue", minValue_, maxValue_, isoValue_);
    valueSlider_->setMenuListener(this);
    menu_->add(valueSlider_);
}

void
IsoSurfaceInteraction::deleteMenu()
{
    delete valueSlider_;
}

void
IsoSurfaceInteraction::updateMenu()
{
    // now set menu items accordingly
    valueSlider_->setMin(minValue_);
    valueSlider_->setMax(maxValue_);
    valueSlider_->setValue(isoValue_);
}

void
IsoSurfaceInteraction::updatePickInteractors(bool show)
{
    updateInteractorVisibility();
}

void
IsoSurfaceInteraction::updateDirectInteractors(bool show)
{
    updateInteractorVisibility();
}
