/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _MODULE_INTERACTION_H
#define _MODULE_INTERACTION_H

#include <PluginUtil/ModuleFeedbackManager.h>

#ifdef VRUI
namespace vrui
{
class coCheckboxMenuItem;
}
#endif

namespace opencover
{
class RenderObject;
}

namespace opencover
{
class PLUGIN_UTILEXPORT ModuleInteraction : public ModuleFeedbackManager
{
public:
    ModuleInteraction(const opencover::RenderObject *container, opencover::coInteractor *inter, const char *pluginName);
    virtual ~ModuleInteraction();
    virtual void update(const opencover::RenderObject *container, opencover::coInteractor *inter) override;
    virtual void preFrame() override;
#ifdef VRUI
    virtual void menuEvent(vrui::coMenuItem *menuItem) override;
#endif
    virtual void updatePickInteractors(bool) = 0;
    virtual void updateDirectInteractors(bool) = 0;
    virtual void setShowInteractorFromGui(bool state);
    virtual void enableDirectInteractorFromGui(bool state);

    void triggerHide(bool state) override;

protected:
    bool showPickInteractor_;
    bool showDirectInteractor_;
#ifdef VRUI
    vrui::coCheckboxMenuItem *showPickInteractorCheckbox_;
    vrui::coCheckboxMenuItem *showDirectInteractorCheckbox_;
#else
    ui::Button *showPickInteractorCheckbox_ = nullptr;
    ui::Button *showDirectInteractorCheckbox_ = nullptr;
#endif
};
}
#endif
