/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _MODULE_INTERACTION_H
#define _MODULE_INTERACTION_H

#include <PluginUtil/ModuleFeedbackManager.h>

namespace vrui
{
class coCheckboxMenuItem;
}

namespace opencover
{
class RenderObject;
}

namespace opencover
{
class PLUGIN_UTILEXPORT ModuleInteraction : public ModuleFeedbackManager
{
public:
    ModuleInteraction(opencover::RenderObject *container, opencover::coInteractor *inter, const char *pluginName);
    virtual ~ModuleInteraction();
    virtual void update(opencover::RenderObject *container, opencover::coInteractor *inter);
    virtual void preFrame();
    virtual void menuEvent(vrui::coMenuItem *menuItem);
    virtual void updatePickInteractors(bool) = 0;
    virtual void updateDirectInteractors(bool) = 0;
    virtual void setShowInteractorFromGui(bool state);

protected:
    bool showPickInteractor_;
    bool showDirectInteractor_;
    vrui::coCheckboxMenuItem *showPickInteractorCheckbox_;
    vrui::coCheckboxMenuItem *showDirectInteractorCheckbox_;
};
}
#endif
